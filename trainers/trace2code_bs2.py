"""Trace2CodeBS2 — trace2code with bootstrapping (--solver=trace2code_bs2, Finding 2).

Training is plain trace2code. Every --bs2_freq epochs (from --bs2_warmup) a bootstrap pass runs:
  - a full no-grad pass over the training set, decoding every program greedily;
  - a decoded program replaces its label when it is strictly shorter and reproduces ALL stored demos of that
    program (read directly from program_data, not just the K demos of a batch);
  - all execution is sequential — no multiprocessing.
The reduced dataset is dumped to data/<datadir>-bs-<model_tag>/ after every pass; the tag distinguishes runs
in flight.
"""

import os
import pickle
import shutil
from functools import partial
from glob import glob
from time import time

import numpy as np
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainers.trace2code import Trace2Code
from common.config_utils import TensorDict
from karel_env.dsl import get_DSL
from karel_env.program_dataset import CustomKarelDataset, collate_custom_karel_batch
import karel_env.karel as _karel_env


_DSL_CACHE = {}


def _trim_at_eos(program_seq, dsl):
    """Cut a decoded token sequence after its first EOS (`m)`) token."""
    eos_positions = (program_seq == dsl.token2int["m)"]).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        program_seq = program_seq[: eos_positions[0].item() + 1]
    return program_seq


def _get_dsl(env: str):
    dsl = _DSL_CACHE.get(env)
    if dsl is None:
        dsl = get_DSL(environment=env)
        _DSL_CACHE[env] = dsl
    return dsl


def _exec_program(env: str, env_kwargs: dict, prog_str: str, s_0_np: np.ndarray,
                  monitor_a=None):
    """Execute prog_str on a single initial state. Returns (s_h_list, a_h_list,
    s_h_len) on success, None on parse/runtime failure. Sequential — no pool.

    When `monitor_a` is supplied, the world raises on the first action that
    deviates from the monitor (over-emission or in-bounds mismatch), so doomed
    candidates short-circuit instead of running to MAX_FUNC_CALL.
    """
    dsl = _get_dsl(env)
    try:
        parsed = dsl.parse(prog_str)
        if parsed is None:
            return None
        if env == "list":
            from list_env.list_world import ListWorld
            L_max = env_kwargs["L_max"]
            mv = env_kwargs["max_val"]
            length = int(s_0_np[-2])
            cursor = int(s_0_np[-1])
            values = s_0_np[:L_max][:length].astype(np.int8)
            world = ListWorld(L_max=L_max, max_val=mv, monitor_a=monitor_a)
            world.set_new_state(values, cursor)
        else:
            world = _karel_env.Karel_world(make_error=False, monitor_a=monitor_a)
            world.set_new_state(s_0_np)
        dsl.run_parsed(world, parsed)
        return list(world.s_h), list(world.a_h), len(world.s_h)
    except Exception:
        return None


class Trace2CodeBS2(Trace2Code):
    def __init__(self, config: dict):
        super().__init__(config)
        self.bs_freq = config["bs2_freq"]
        self.bs_warmup = config["bs2_warmup"]
        self.bs_model_tag = config["bs2_model_tag"]
        self.bs_batch_size = config["batch_size"]
        if not self.bs_model_tag:
            raise ValueError("--bs2_model_tag is required for --solver=trace2code_bs2")

        self._make_bootstrap_loader()
        self._writeback_indices: set = set()
        self._tokens_saved: int = 0
        self._original_total_prog_len: int = sum(
            int(p["program_len"]) for p in self.train_loader.dataset.program_data
        )
        self._dataset_size: int = len(self.train_loader.dataset.program_data)
        self._original_mean_prog_len: float = self._original_total_prog_len / max(self._dataset_size, 1)
        self._n_bs_passes: int = 0
        self._mini_sibling_pickles_copied: bool = False
        self._rasp_train_io = None  # RASP: rasp_io_train.pkl, loaded on first use

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _make_bootstrap_loader(self):
        train_dataset = self.train_loader.dataset
        bs_dataset = CustomKarelDataset(
            self.config,
            mode="bootstrap",
            share_program_data_from=train_dataset,
        )
        pad_token = bs_dataset.pad_token
        # No shuffle: we want determinism, and the pass touches every program
        # exactly once anyway. Single-process loading: avoids re-pickling
        # mutated program_data into worker processes after writebacks.
        self.bootstrap_loader = DataLoader(
            bs_dataset,
            batch_size=self.bs_batch_size,
            collate_fn=partial(collate_custom_karel_batch, pad_token=pad_token),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def _env_kwargs(self):
        env = self.config["environment"]
        if env == "list":
            es = self.config["env_spec"]
            return {"L_max": es.L_max, "max_val": es.max_val}
        return {}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, callback=None):
        print("Starting Training Loop (Bootstrapped v2)...")
        self.iterations = 0
        self.model.train()
        self.best_stats = {}
        log_every = self.config["log_every"]

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs} started..")

            # Bootstrap pass at the *start* of qualifying epochs. The reduced
            # dataset takes effect immediately for this epoch's training.
            if epoch >= self.bs_warmup and (epoch - self.bs_warmup) % self.bs_freq == 0:
                self._bootstrap_pass(epoch)
                self._save_reduced_dataset()

            start = time()
            for batch in tqdm(self.train_loader):
                batch = self.move_batch(batch)
                batch = self.solve(batch, mode="train")

                is_log_step = (self.iterations % log_every == 0)
                loss_dict = self.update(batch, sync_loss=is_log_step)
                if is_log_step:
                    metrics = self.compute_metrics(batch)
                    log_dict = {
                        "LR": self.optimizer.param_groups[0]["lr"],
                        "loss": loss_dict,
                        **metrics,
                    }
                    self.logger.record(log_dict, "train", tag="", iteration=self.iterations, roll_window=20)
                self.reset()
                self.iterations += 1

            elapsed = time() - start
            print(f"Epoch {epoch} ended! Time elapsed: {elapsed:.2f} seconds.")
            self.on_epoch_end(epoch, tag="train")
            avg_solved = self.validate(epoch=epoch)
            if callback is not None:
                callback(epoch, avg_solved)

    # ------------------------------------------------------------------
    # Bootstrap pass
    # ------------------------------------------------------------------

    def _bootstrap_pass(self, epoch: int):
        dsl = self.config["dsl-env"]
        env = self.config["environment"]
        env_kwargs = self._env_kwargs()
        is_flat_state = env == "list"
        program_data = self.train_loader.dataset.program_data

        n_attempted = 0
        n_committed = 0
        n_decoded_no_shorter = 0  # decoded program was not shorter than GT — skipped
        tokens_saved_this_pass = 0
        start = time()

        self.model.eval()
        for batch in tqdm(self.bootstrap_loader, desc=f"BS pass [epoch {epoch+1}]"):
            batch = self.move_batch(batch)
            with th.inference_mode():
                decoded = self.model(batch)
            gen_programs = decoded["generated_programs"].cpu()  # (B, T_gen)
            indices = batch["idx"].cpu().tolist()

            B = gen_programs.shape[0]
            for b in range(B):
                n_attempted += 1
                ds_idx = int(indices[b])
                gt_len = int(program_data[ds_idx]["program_len"])

                trimmed = _trim_at_eos(gen_programs[b], dsl)
                decoded_len = int(trimmed.shape[0])
                if decoded_len >= gt_len:
                    n_decoded_no_shorter += 1
                    continue

                prog_str = dsl.intseq2str(trimmed)
                if not prog_str.startswith("DEF"):
                    prog_str = f"DEF {prog_str}"

                if self._verify_one(program_data[ds_idx], prog_str, env, env_kwargs, is_flat_state):
                    program_data[ds_idx]["program"] = trimmed.cpu()
                    program_data[ds_idx]["program_len"] = decoded_len
                    self._writeback_indices.add(ds_idx)
                    saved = gt_len - decoded_len
                    self._tokens_saved += saved
                    tokens_saved_this_pass += saved
                    n_committed += 1
        self.model.train()

        elapsed = time() - start
        self._n_bs_passes += 1

        n_unique_reduced = len(self._writeback_indices)
        cum_tokens_current = self._original_total_prog_len - self._tokens_saved
        cum_mean_prog_len = cum_tokens_current / max(self._dataset_size, 1)
        cum_pct_samples_reduced = 100.0 * n_unique_reduced / max(self._dataset_size, 1)
        cum_pct_tokens_reduced = 100.0 * self._tokens_saved / max(self._original_total_prog_len, 1)
        pct_committed = 100.0 * n_committed / max(n_attempted, 1)

        bar = "=" * 70
        dash = "-" * 70
        print(
            f"\n{bar}\n"
            f"  Bootstrap pass #{self._n_bs_passes} (epoch {epoch+1})  |  elapsed: {elapsed:.1f}s\n"
            f"{bar}\n"
            f"  This pass:\n"
            f"    attempted             : {n_attempted}\n"
            f"    committed             : {n_committed}  ({pct_committed:.2f}%)\n"
            f"    decoded not shorter   : {n_decoded_no_shorter}\n"
            f"    tokens saved          : {tokens_saved_this_pass}\n"
            f"{dash}\n"
            f"  Cumulative (over {self._n_bs_passes} pass{'es' if self._n_bs_passes != 1 else ''}):\n"
            f"    unique samples reduced: {n_unique_reduced} / {self._dataset_size}  ({cum_pct_samples_reduced:.2f}%)\n"
            f"    tokens saved          : {self._tokens_saved} / {self._original_total_prog_len}  ({cum_pct_tokens_reduced:.2f}%)\n"
            f"    total prog tokens     : {cum_tokens_current}  (was {self._original_total_prog_len})\n"
            f"    mean prog length      : {cum_mean_prog_len:.3f}  (was {self._original_mean_prog_len:.3f})\n"
            f"{bar}\n"
        )

        bs_log = {
            # Per-pass
            "bs_attempted": n_attempted,
            "bs_committed": n_committed,
            "bs_decoded_not_shorter": n_decoded_no_shorter,
            "bs_tokens_saved_pass": tokens_saved_this_pass,
            "bs_pct_committed": pct_committed,
            # Cumulative
            "bs_n_passes": self._n_bs_passes,
            "bs_cum_unique_reduced": n_unique_reduced,
            "bs_cum_pct_samples_reduced": cum_pct_samples_reduced,
            "bs_cum_tokens_saved": self._tokens_saved,
            "bs_cum_pct_tokens_reduced": cum_pct_tokens_reduced,
            "bs_cum_total_tokens": cum_tokens_current,
            "bs_cum_mean_prog_len": cum_mean_prog_len,
        }
        self.logger.record(bs_log, "train", tag="bs", iteration=self.iterations, roll_window=1)
        # Bundle into the next CSV row (flush=False — the last validate() of
        # this epoch will flush the accumulated dict).
        self.log_csv(bs_log, tag="bs", epoch=epoch, flush=False)

    def _verify_one(self, sample: dict, prog_str: str, env: str, env_kwargs: dict, is_flat_state: bool) -> bool:
        """Re-execute prog_str on EVERY stored initial state of `sample` and
        return True iff (a) every execution succeeds and (b) every produced
        trace matches GT (length + actions) exactly. Only then is the
        candidate safe to commit as a replacement.

        The monitor in the world catches in-bounds mismatches and over-emission
        eagerly (raises → `_exec_program` returns None). The post-exec length
        check below handles under-emission (candidate halts cleanly as a strict
        prefix, where the monitor stays silent).
        """
        if env == "rasp":
            return self._verify_rasp(sample, prog_str)
        K_avail = sample["s_h"].shape[0]
        for k in range(K_avail):
            s_0_compact = sample["s_h"][k, 0]
            if is_flat_state:
                # List: int8 [buffer, length, cursor] snapshot.
                s_0_np = s_0_compact.numpy().astype(np.int8)
            else:
                # Karel: compact grid descriptor → dense (W, H, C) bool grid.
                s_0_kt = s_0_compact.view(1, 1, -1)
                s_0_exp = self.config["env_spec"].expand_state(s_0_kt)  # (1,1,C,W,H)
                s_0_np = s_0_exp[0, 0].permute(1, 2, 0).numpy().astype(bool)

            gt_len = int(sample["s_h_len"][k].item())
            gt_actions = sample["a_h"][k, : gt_len - 1].tolist()

            result = _exec_program(env, env_kwargs, prog_str, s_0_np,
                                   monitor_a=gt_actions)
            if result is None:
                return False
            _, _, s_h_len_k = result
            if s_h_len_k != gt_len:
                return False
        return True

    def _verify_rasp(self, sample: dict, prog_str: str) -> bool:
        """RASP has no world to replay: the candidate must reproduce the stored output of every stored input
        (rasp_io_train.pkl; on the sweep view, every frame of every sweep)."""
        from rasp_env import rasp_grammar as G
        if self._rasp_train_io is None:
            path = os.path.join("data", self.train_loader.dataset.datadir, "rasp_io_train.pkl")
            with open(path, "rb") as f:
                self._rasp_train_io = pickle.load(f)
        try:
            spec = G.tokens_to_spec(prog_str.split())
        except (ValueError, IndexError):
            return False
        return self._rasp_behavioral_ratio(G, spec, self._rasp_train_io[sample["program_id"]]) == 1.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_reduced_dataset(self):
        dataset = self.train_loader.dataset
        src_dir = os.path.join("data", dataset.datadir)
        dst_dir = os.path.join("data", f"{dataset.datadir}-bs-{self.bs_model_tag}")
        os.makedirs(dst_dir, exist_ok=True)

        train_pkl = os.path.join(dst_dir, "processed_dataset_train.pkl")
        payload = {
            "program_data": dataset.program_data,
            "max_demo_length": dataset.max_demo_length,
            "max_program_len": dataset.max_program_len,
            "num_program_tokens": dataset.num_program_tokens,
        }
        with open(train_pkl, "wb") as f:
            pickle.dump(payload, f)

        if not self._mini_sibling_pickles_copied:
            # dataset_info.json + split id files carry the DSL version and the
            # marked train/test split; without them the dump would load as DSL v1.
            # RASP: the IO sidecars (metric + verification inputs; their gt_token_str
            # stays the pre-reduction label, the current one is in program_data) and meta.
            siblings = glob(os.path.join(src_dir, "processed_dataset_*.pkl")) + \
                glob(os.path.join(src_dir, "rasp_io_*.pkl")) + [
                os.path.join(src_dir, f) for f in ("dataset_info.json", "train_ids.txt", "test_ids.txt",
                                                   "rasp_data_info.json")
                if os.path.exists(os.path.join(src_dir, f))]
            for src in siblings:
                base = os.path.basename(src)
                if base == "processed_dataset_train.pkl":
                    continue
                dst = os.path.join(dst_dir, base)
                if not os.path.exists(dst):
                    shutil.copyfile(src, dst)
            self._mini_sibling_pickles_copied = True

        print(f"[BS save] wrote {len(dataset.program_data)} programs to {train_pkl}")
