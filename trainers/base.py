import json
import os
from karel_env.program_dataset import CustomKarelDataset, collate_custom_karel_batch
from torch.utils.data import DataLoader
import torch.nn as nn
from time import time
from functools import partial

from karel_env.dsl import get_DSL, dsl_version_for
from common.tokenizer import Tokenizer
from common.csv_logger import CSVLogger
from common.env_spec import get_env_spec


# Keys excluded from the saved config.json snapshot: objects added by trainer setup, secrets, values derived
# at runtime, logging knobs and test-time flags. Anything else is kept if it is JSON-serializable.
_CONFIG_JSON_EXCLUDE = frozenset({
    "tokenizer", "dsl-env", "env_spec", "csv_logger",
    "wandb_key",
    "state_shape", "img_size", "log_dir", "dsl_tokens",
    "no_wandb", "log_every",
    "checkpoint", "test_only",
})


def save_config_json(config: dict, dirpath: str, filename: str = "config.json",
                     extra: dict = None):
    """Write a JSON snapshot of the training config into dirpath/filename.

    Excludes non-serializable objects, secrets, and clearly irrelevant fields
    (see _CONFIG_JSON_EXCLUDE). Any remaining values that fail to serialize are
    silently skipped so a single weird object can't block the whole dump.

    `extra` is merged into the output verbatim (overriding config keys on
    collision). Use it to record the checkpoint-deciding metrics alongside
    the config that produced them."""
    cleaned = {}
    for k, v in {**config, **(extra or {})}.items():
        if k in _CONFIG_JSON_EXCLUDE:
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        cleaned[k] = v
    out_path = os.path.join(dirpath, filename)
    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)


class BaseTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _make_meta(self, config):
        # run_name is normally set in BaseOptions.parse; ensure it exists for
        # configs rebuilt from a checkpoint's config.json (latent_view.py).
        config.setdefault("run_name", f"{config['exp_name']}-r{config['seed']}")
        env = config["environment"]
        config.setdefault("dsl_version", dsl_version_for(config["datadir"]))
        self.config["env_spec"] = get_env_spec(env)
        self.tokenizer = Tokenizer(config)
        self.config["tokenizer"] = self.tokenizer
        self.config["dsl-env"] = get_DSL(dsl_type='prob', seed=config["seed"], environment=env,
                                         version=config["dsl_version"])
        self.csv_logger = CSVLogger(self.config)

    def _make_datasets(self, config):
        start = time()
        # CSV columns of the test metrics carry this tag, e.g. ID[15-20-v2]-program_total_match.
        self.test_tag = f"ID[{config['datadir'].split('len-')[-1]}]"

        num_workers = config.get("num_workers", 0)
        loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

        if not self.config["test_only"]:
            print("Creating datasets...")
            train_dataset = CustomKarelDataset(config, mode='train')
            self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                           collate_fn=partial(collate_custom_karel_batch, pad_token=train_dataset.pad_token),
                                           shuffle=True, **loader_kwargs)
            max_demo_length = train_dataset.max_demo_length
            max_program_length = train_dataset.max_program_len
        else:
            print(f"Skipping creation of train dataset")
            max_demo_length = 0
            max_program_length = 0

        dataset = CustomKarelDataset(config, mode='valid')
        self.test_loader = DataLoader(dataset, batch_size=config['batch_size'],
                                      collate_fn=partial(collate_custom_karel_batch, pad_token=dataset.pad_token),
                                      shuffle=False, **loader_kwargs)

        self.config["max_demo_length"] = max(max_demo_length, dataset.max_demo_length)
        self.config["max_program_len"] = max(max_program_length, dataset.max_program_len)
        self.config["img_size"] = dataset.img_size
        self.config["state_shape"] = dataset.img_size
        print(f"Datasets created in {time() - start:.2f} seconds.")
