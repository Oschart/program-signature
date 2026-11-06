import os
import json
import pickle
import random

import h5py
import numpy as np
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

from karel_env.dsl import dsl_version_for
from common.utils import pad_tensor, pad_repeat_tensor


def _load_pickle_dataset(pickle_path):
    """Load a processed_dataset_*.pkl: a pickled dict {'program_data': [...], 'max_demo_length': ...,
    'max_program_len': ..., 'num_program_tokens': ...}."""
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    if not (isinstance(dataset, dict) and 'program_data' in dataset):
        raise RuntimeError(f"Unrecognized pickle format at {pickle_path}")
    return dataset


def _check_matches_split(program_data, marked_ids, split_name, pickle_path):
    """A cached pickle must hold exactly the marked ids, in marked order. A train pickle may
    be a prefix of them (size-capped dump, e.g. trace2code_bs2). Anything else means the
    pickle was re-split or renamed: fail instead of training/evaluating on the wrong split."""
    got = [d['program_id'] for d in program_data]
    ok = got == marked_ids or (split_name == 'train' and got == marked_ids[:len(got)])
    if not ok:
        raise ValueError(f"{pickle_path} does not match the marked {split_name} split "
                         f"({len(got)} vs {len(marked_ids)} ids); delete it to rebuild from data.hdf5.")


def read_marked_split(datadir):
    """The {'train': ids, 'test': ids} split marked in data/<datadir>/dataset_info.json (Karel / List
    generators), or None when the dataset has none (RASP: its generator writes the split pickles directly)."""
    info_path = os.path.join("data", datadir, 'dataset_info.json')
    if not os.path.exists(info_path):
        return None
    with open(info_path) as f:
        split = json.load(f).get('split')
    if split is None:
        return None
    ids = {}
    for name in ('train', 'test'):
        with open(os.path.join("data", datadir, split[name])) as f:
            ids[name] = [line.strip() for line in f if line.strip()]
        if len(ids[name]) != split[f'n_{name}']:
            raise ValueError(f"{split[name]} has {len(ids[name])} ids, dataset_info says {split[f'n_{name}']}")
    if set(ids['train']) & set(ids['test']):
        raise ValueError(f"{datadir}: marked train and test ids overlap")
    return ids


def build_split_pickles(datadir, split_ids, dsl_max_program_len):
    """Convert data/<datadir>/data.hdf5 (written by the Karel / List generators) into one
    processed_dataset_<split>.pkl per marked split, programs in marked order. Training does this the
    first time it loads a dataset; karel_env/rewrite.py does it when no run has."""
    program_data = []
    hdf5_file = h5py.File(os.path.join("data", datadir, 'data.hdf5'), 'r')
    id_file = open(os.path.join("data", datadir, 'id.txt'), 'r')
    max_demo_length = hdf5_file['data_info']['max_demo_length'][()]
    max_program_len = max(hdf5_file['data_info']['max_program_len'][()], dsl_max_program_len + 1)  # +1 for DEF token
    num_program_tokens = hdf5_file['data_info']['num_program_tokens'][()]

    id_list = id_file.readlines()
    for program_id in tqdm(id_list):
        program_id = program_id.strip()
        program = hdf5_file[program_id]['program'][()]
        program = th.tensor(program)
        program_len = program.shape[0]
        s_h = th.tensor(hdf5_file[program_id]['s_h'])
        a_h = th.tensor(hdf5_file[program_id]['a_h'])
        s_h_len = th.tensor(hdf5_file[program_id]['s_h_len'])

        program_dict = {
            'program_id': program_id,
            'program': program,
            'program_len': program_len,
            's_h': s_h,
            'a_h': a_h,
            's_h_len': s_h_len
        }
        program_data.append(program_dict)

    hdf5_file.close()
    id_file.close()
    dataset = {
        "program_data": program_data,
        "max_demo_length": max_demo_length,
        "max_program_len": max_program_len,
        "num_program_tokens": num_program_tokens
    }
    by_id = {d['program_id']: d for d in program_data}
    for name, ids in split_ids.items():
        mode_dataset = {**dataset, 'program_data': [by_id[i] for i in ids]}
        mode_pickle_path = os.path.join("data", datadir, f'processed_dataset_{name}.pkl')
        print(f"Saving pickled dataset to {mode_pickle_path}")
        # Atomic write: when many jobs launch against an un-cached dataset they
        # race to build this pickle. A bare open('wb')+dump lets concurrent (or
        # interrupted) writes leave a truncated/interleaved file that later reads
        # as `invalid load key`. Write to a per-process temp then os.replace, so
        # the final path only ever appears as a complete pickle.
        tmp_path = f"{mode_pickle_path}.tmp.{os.getpid()}"
        with open(tmp_path, 'wb') as f:
            pickle.dump(mode_dataset, f)
        os.replace(tmp_path, mode_pickle_path)


class CustomKarelDataset(Dataset):
    """Programs and their traces of one split (Karel, List or RASP)."""

    def __init__(self, config, mode='train', share_program_data_from=None):
        """
        Parameters:
            :param config (dict): all configs in dict format
            :param mode: 'train', 'valid' (the marked test split) or 'bootstrap' (training programs with all
                their stored demos, for trace2code_bs2)
            :param share_program_data_from: another CustomKarelDataset whose program_data (and
                derived attributes) this instance should reuse by reference instead of loading
                from disk. Used by the bootstrap loader so writebacks to program_data are
                visible to the train loader.
        """
        self.config = config
        self.pad_token = config['tokenizer'].pad_token
        self.mode = mode
        self.num_demos = config['num_demos'] if mode == 'train' else config.get('num_demos_test', config['num_demos'])
        self.datadir = config['datadir']
        self.size_cap = config['size_cap']
        self.defer_state_expansion = config.get('defer_state_expansion', False)
        if share_program_data_from is not None:
            self.env_spec = self.config["env_spec"]
            self.env_spec.parse_data_dims(self.datadir)
            self.img_size = self.env_spec.state_shape
            self.program_data = share_program_data_from.program_data
            self.max_demo_length = share_program_data_from.max_demo_length
            self.max_program_len = share_program_data_from.max_program_len
            self.num_program_tokens = share_program_data_from.num_program_tokens
            self.print_data_stats()
            return
        self.load_dataset(self.datadir)
        if self.mode == "train":
            print(f"Applying size cap: {self.size_cap}")
            self.program_data = self.program_data[:self.size_cap]
        self.print_data_stats()

    def load_dataset(self, datadir):
        self.env_spec = self.config["env_spec"]
        self.env_spec.parse_data_dims(datadir)
        self.img_size = self.env_spec.state_shape
        split_ids = self._marked_split(datadir)  # None for RASP
        file_mode = self.mode
        if split_ids is not None:
            file_mode = 'test' if self.mode == 'valid' else self.mode  # validate on the test split

        pickle_path = os.path.join("data", datadir, f'processed_dataset_{file_mode}.pkl')
        if not os.path.exists(pickle_path):
            if split_ids is None:
                raise FileNotFoundError(f"{pickle_path} does not exist")
            build_split_pickles(datadir, split_ids, self.config['dsl']['max_program_len'])
        cache_key = f"{self.mode}_dataset::{datadir}"
        if self.config.get(cache_key, None) is not None:
            print(f"Using cached dataset for {datadir} ({self.mode})")
            dataset = self.config[cache_key]
        else:
            print(f"Loading pickled dataset from {pickle_path}")
            dataset = _load_pickle_dataset(pickle_path)
            self.config[cache_key] = dataset
        if split_ids is not None:
            _check_matches_split(dataset["program_data"], split_ids[file_mode], file_mode, pickle_path)
        self.program_data = dataset["program_data"]
        self.max_demo_length = dataset["max_demo_length"]
        self.max_program_len = max(dataset["max_program_len"], self.config['dsl']['max_program_len'] + 1)  # +1 for DEF token
        self.num_program_tokens = dataset["num_program_tokens"]

    def _marked_split(self, datadir):
        """Check data/<datadir>/dataset_info.json against this run and return the marked
        {'train': ids, 'test': ids} split, or None for datasets without one."""
        data_version, run_version = dsl_version_for(datadir), self.config.get('dsl_version', 1)
        if data_version != run_version:
            raise ValueError(f"{datadir} was generated with DSL v{data_version} but this run uses "
                             f"v{run_version}: token ids differ, so the datasets cannot be mixed.")
        info_path = os.path.join("data", datadir, 'dataset_info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                data_env = json.load(f).get('environment')
            run_env = self.config.get('environment')
            if data_env and run_env and data_env != run_env:
                raise ValueError(f"{datadir} is a '{data_env}' dataset but this run uses the {run_env} environment.")
        return read_marked_split(datadir)

    def print_data_stats(self):
        print(20*'-')
        print(f"Dataset split: {self.mode}")
        print(f"Dataset size: {len(self.program_data)}")
        print(f"Max program length: {self.max_program_len}")
        print(f"Max program length (actual): {max([sample['program_len'] for sample in self.program_data])}")
        print(f"Max demo length: {self.max_demo_length}")
        print(f"Max demo length (actual): {max([sample['s_h_len'].float().max() for sample in self.program_data])}")
        # Mean program length
        mean_prog_len = np.mean([sample['program_len'] for sample in self.program_data])
        print(f"Mean program length: {mean_prog_len}")
        # Mean demo length
        mean_demo_len = np.mean([sample['s_h_len'].float().mean() for sample in self.program_data])
        print(f"Mean demo length: {mean_demo_len}")
        print(20*'-')

    def __len__(self):
        return len(self.program_data)

    def __getitem__(self, idx):
        sample = self.program_data[idx]
        program_len = sample["program_len"]
        program = sample["program"].long()

        mask = th.zeros((self.max_program_len, 1), dtype=th.bool)
        mask[:program_len] = 1
        K_avail = sample["s_h"].shape[0]
        if self.mode == 'bootstrap':
            # All stored demos, deterministic order — used by trace2code_bs2
            # to verify writebacks against every demo, not just K.
            selected_indices = list(range(K_avail))
        elif self.mode == 'train':
            num_demos = min(self.num_demos, K_avail)
            selected_indices = random.sample(range(K_avail), num_demos)
        else:
            # Deterministic first-K selection for reproducibility
            num_demos = min(self.num_demos, K_avail)
            selected_indices = list(range(num_demos))

        # load exec data
        s_h = sample["s_h"][selected_indices]
        a_h = sample["a_h"][selected_indices]
        s_h_len = sample["s_h_len"][selected_indices]

        # Expand state: grid descriptors → images (Karel) or raw → float (List, RASP).
        # When --defer_state_expansion is set, the trainer's move_batch will run
        # the equivalent batched expansion on GPU instead — we keep s_h compact
        # here to shrink host→device transfer.
        if not self.defer_state_expansion:
            s_h = self.env_spec.expand_state(s_h)

        a_h = a_h.to(th.int32)       # (K, T-1)
        s_h_len = s_h_len.to(th.int32)  # (K,)

        sample_dict = {
            'program': program,
            'mask': mask,
            'program_len': program_len,
            's_h': s_h,
            'a_h': a_h,
            's_h_len': s_h_len,
            'idx': idx,
        }
        return sample_dict


# @profile
def _pad_last_dim_repeat(t: th.Tensor, target_d: int) -> th.Tensor:
    """Pad t's last dim up to target_d by repeating the last value.
    Idempotent for compact one-hot scatter indices: re-writing a position
    to 1.0 is a no-op, so this is safe to apply before expand_state_batched."""
    cur_d = t.shape[-1]
    if cur_d >= target_d:
        return t
    pad_amt = target_d - cur_d
    last = t[..., -1:].expand(*t.shape[:-1], pad_amt)
    return th.cat([t, last], dim=-1)


def collate_custom_karel_batch(data_list, pad_token):
    max_prog_len = max(sample['program_len'] for sample in data_list)
    # .item() up front so the outer max() compares Python ints rather than
    # 0-dim tensors — tensor-vs-tensor comparisons are ~10x slower per call.
    max_demo_len = max(sample['s_h_len'].max().item() for sample in data_list)
    # When samples carry compact grid descriptors (deferred expansion mode),
    # the converter pads D within each sample but D still varies across
    # samples — equalize it here so torch.stack succeeds.
    needs_d_pad = data_list[0]['s_h'].dim() == 3
    max_state_d = (max(s['s_h'].shape[-1] for s in data_list)
                   if needs_d_pad else None)

    batch = {
        'program': [],
        'program_len': [],
        'masks': [],
        's_h': [],
        'a_h': [],
        's_h_len': [],
        'idx': [],
    }

    for sample in data_list:
        program = sample['program']
        program_len = sample['program_len']
        s_h = sample['s_h']
        a_h = sample['a_h']
        s_h_len = sample['s_h_len']
        mask = sample['mask']

        # Pad program
        program = pad_tensor(program, max_prog_len, dim=0, pad_value=pad_token)
        mask = pad_tensor(mask, max_prog_len, dim=0, pad_value=0)
        s_h = pad_repeat_tensor(s_h, s_h_len, max_demo_len, pad_dim=1)
        if max_state_d is not None:
            s_h = _pad_last_dim_repeat(s_h, max_state_d)
        a_h = pad_repeat_tensor(a_h, s_h_len-1, max_demo_len-1, pad_dim=1)

        batch['program'].append(program)
        batch['program_len'].append(program_len)
        batch['masks'].append(mask)
        batch['s_h'].append(s_h)
        batch['a_h'].append(a_h)
        batch['s_h_len'].append(s_h_len)
        batch['idx'].append(sample['idx'])

    # Stack up
    for key in batch:
        if key == "program_len":
            batch[key] = th.tensor(batch[key])
            continue
        if key == "idx":
            batch[key] = th.tensor(batch[key], dtype=th.long)
            continue
        batch[key] = th.stack(batch[key])

    return batch
