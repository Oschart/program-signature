import torch as th
import torch.nn as nn
import torch.optim as optim
from models.prog_decoder import KarelSynthesizer
from trainers.base import BaseTrainer, save_config_json
from common.config_utils import safer_mean, TensorDict
from tqdm import tqdm
import os
from common.logger import Logger
from karel_env.karel_utils import verify_programs
from time import time

class Trace2Code(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.device = config["device"]
        self.logger = Logger(config)
        self.log_accum = {}
        self._make_meta(config)
        self._make_datasets(self.config)
        self._make_aliases()
        self._make_model()
        self._setup_losses()
        self.load_checkpoint(config["checkpoint"])
        self._make_optimizer()

        self.batch_size = config["batch_size"]
        self.iterations = 0
        self.fitness_metrics = ["program_total_match"]

    def _make_aliases(self):
        self.n_epochs = self.config["n_epochs"]
        self.max_grad_norm = self.config["max_grad_norm"]
        self.pad_token = self.tokenizer.pad_token
        self.num_program_tokens = self.tokenizer.num_program_tokens
        self.base_lr = self.config["base_lr"]

        self.model_name = self.config["run_name"]

    def _make_model(self):
        self.model = KarelSynthesizer(self.config).to(self.device)
        self._pending_opt_state = None

    def _make_optimizer(self):
        params = list(self.model.parameters())
        print(f"Optimizer over {sum(p.numel() for p in params)} trainable params")
        wd = self.config.get("weight_decay", 0.0)
        if wd > 0:
            # AdamW on the weight matrices only (biases and norm gains are not decayed)
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            print(f"[weight_decay] AdamW lambda={wd:g}: "
                  f"{sum(p.numel() for p in decay)} decayed / {sum(p.numel() for p in no_decay)} undecayed params")
            groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
            self.optimizer = optim.AdamW([g for g in groups if g["params"]], lr=self.base_lr, eps=1e-5)
        else:
            self.optimizer = optim.Adam(params, lr=self.base_lr, eps=1e-5)

        if self._pending_opt_state is not None:
            try:
                self.optimizer.load_state_dict(self._pending_opt_state)
                print("Restored optimizer state from checkpoint")
            except Exception as e:
                print(f"Could not load optimizer state (param set may differ): {e}")
        self._pending_opt_state = None

    def _setup_losses(self):
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token, reduction='none')
        self.loss_funcs = {
            "ce_loss": (self.ce_loss, 1.0),
        }

    def ce_loss(self, batch: TensorDict):
        targets = batch["program"][:, 1:]  # (B, T-1)
        mask = (targets != self.pad_token).float()  # (B, T-1)
        gt_len = targets.shape[1]
        logits = batch["logits"][:, :gt_len]  # (B, T-1, V)

        ce_loss = self.ce_loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))  # (B*T,)
        ce_loss = ce_loss.view(targets.size())  # (B, T-1)
        ce_loss = (ce_loss * mask).sum() / mask.sum()

        return ce_loss
    
    # @profile
    def compute_semantic_accr(self, batch: TensorDict):
        if self.config["environment"] == "rasp":
            # RASP has no imperative world to re-execute (env_spec.make_world
            # raises), but behavioral equivalence is well-defined: the decoded
            # program must reproduce GT outputs on the K stored inputs. We compute
            # it here (via evaluate_spec_py + the rasp_io sidecar) so program_total_match
            # IS behavioral-eq -- parity with the imperative envs, where it is
            # re-execution equivalence -- and best_model is selected on the metric
            # the paper reports. Falls back to exact token match if the sidecar is absent.
            return self._compute_rasp_semantic(batch)

        a_gt = batch["a_h"].cpu()
        a_h_len = batch["s_h_len"] - 1 # (B, K)
        a_h_len = a_h_len.cpu()
        program_seqs = batch["generated_programs"]  # (B, T)

        s_h = batch["s_h"]
        s_h = self.config["env_spec"].prepare_state_for_verification(s_h)
        s_h = s_h.cpu().numpy()
        a_gt = a_gt.numpy()
        a_h_len = a_h_len.numpy()
        accr_dict = verify_programs(self.config, program_seqs, s_h, a_gt, a_h_len)

        return {
            "program_total_match": accr_dict["total_match"],
            "semantic_match_score": accr_dict["match_ratio"],
            "io_match_score": accr_dict["io_match_ratio"]
        }

    def _rasp_exact_match(self, preds, targets):
        """Fraction of rows where preds exactly equals targets over the GT's
        non-pad prefix. preds/targets both start with DEF (SOS) and end with m)
        (EOS) — so this is full-program exact token match."""
        pad = self.pad_token
        gt_len = targets.shape[1]
        if preds.shape[1] < gt_len:
            preds = th.nn.functional.pad(preds, (0, gt_len - preds.shape[1]), value=pad)
        else:
            preds = preds[:, :gt_len]
        mask = (targets != pad).float()                       # (B, T)
        correct = (preds == targets).float() * mask
        program_match = th.all(correct + (1 - mask), dim=1).float().mean()
        return program_match.item()

    def _compute_rasp_token_match(self, batch: TensorDict):
        match = self._rasp_exact_match(batch["generated_programs"], batch["program"])
        return {
            "program_total_match": match,
            "semantic_match_score": match,
            "io_match_score": 0.0,
        }

    def _rasp_dsl_cached(self):
        """Lazily build + cache a RaspDSL (int<->token maps for trim/parse)."""
        if getattr(self, "_rasp_dsl", None) is None:
            from rasp_env.rasp_grammar import RaspDSL
            self._rasp_dsl = RaspDSL()
        return self._rasp_dsl

    def _rasp_io_side(self, loader):
        """Load + cache the rasp_io_valid.pkl sidecar for `loader`'s datadir
        (inputs + GT outputs keyed by program_id). Returns None (warned once per
        datadir) when the loader or sidecar is unavailable, so the caller falls
        back to exact-token-match fitness. Every test loader is built mode='valid'
        (base.py), so the sidecar name is fixed."""
        import pickle
        cache = self.__dict__.setdefault("_rasp_io_cache", {})
        datadir = getattr(getattr(loader, "dataset", None), "datadir", None)
        if datadir is None:
            return None
        if datadir not in cache:
            path = os.path.join("data", datadir, "rasp_io_valid.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    cache[datadir] = pickle.load(f)
            else:
                print(f"[rasp] WARNING: missing IO sidecar {path}; falling back "
                      f"to exact-token-match fitness for '{datadir}'.")
                cache[datadir] = None
        return cache[datadir]

    @staticmethod
    def _rasp_behavioral_ratio(G, spec, io_entry):
        """Fraction of the K stored inputs `spec` reproduces (0..1) — the
        partial-credit analog of the imperative verify_program's match_ratio.
        All-K (ratio == 1.0) is behavioral equivalence."""
        inputs, gts = io_entry["inputs"], io_entry["gt_outputs"]
        if not inputs:
            return 0.0
        ok = 0
        for x, gt_out in zip(inputs, gts):
            try:
                pred = G.evaluate_spec_py(spec, x)
            except Exception:
                continue
            if list(pred) == list(gt_out):
                ok += 1
        return ok / len(inputs)

    def _compute_rasp_semantic(self, batch: TensorDict):
        """RASP semantic metrics, defined to mirror the imperative envs (Karel/List)
        so the columns mean the same thing everywhere:
          - program_total_match : behavioral equivalence = decoded program reproduces
                                  GT outputs on ALL K stored inputs (== imperative
                                  total_match; the fitness metric best_model selects on).
          - semantic_match_score: mean FRACTION of the K inputs reproduced (partial
                                  credit == imperative match_ratio; always >= total).
          - io_match_score      : parseable-program rate (1 - parse_fail).
        Exact token match is already logged as (greedy) prog_synt_accr, so it is not
        duplicated here. Requires the test set's IO sidecar; falls back to exact-token-match
        when it is absent."""
        from rasp_env import rasp_grammar as G

        loader = self.test_loader
        io_side = self._rasp_io_side(loader)
        if io_side is None:
            return self._compute_rasp_token_match(batch)

        program_data = loader.dataset.program_data
        dsl = self._rasp_dsl_cached()
        idxs = batch["idx"].cpu().tolist()
        programs = batch["generated_programs"].cpu().tolist()

        B = len(idxs)
        full_ok = ratio_sum = parse_ok = 0.0
        for i in range(B):
            io_entry = io_side[program_data[idxs[i]]["program_id"]]
            ratio = 0.0
            row_parse = False
            gen_tokens = G.trim_program_ids(programs[i], dsl)
            if gen_tokens is not None:
                try:
                    spec = G.tokens_to_spec(gen_tokens)
                    row_parse = True
                except Exception:
                    pass
            if row_parse:
                ratio = self._rasp_behavioral_ratio(G, spec, io_entry)
            full_ok += (ratio == 1.0)          # all-K behavioral (fitness)
            ratio_sum += ratio                  # partial behavioral
            parse_ok += row_parse
        return {
            "program_total_match": full_ok / B,     # all-K behavioral; == imperative total_match
            "semantic_match_score": ratio_sum / B,  # partial behavioral; == imperative match_ratio (>= total)
            "io_match_score": parse_ok / B,         # parseable-program rate (1 - parse_fail)
        }

    def compute_syntactic_accr(self, preds, targets, mask):
        gt_len = targets.shape[1]
        if preds.shape[1] < gt_len:
            preds = th.nn.functional.pad(preds, (0, gt_len - preds.shape[1]), value=self.pad_token)
        else:
            preds = preds[:, :gt_len]
        correct = (preds == targets).float() * mask  # (B, T)
        token_accr = correct.sum() / mask.sum()
        program_accr = th.all(correct + (1 - mask), dim=1).float().mean()
        return {
            "token_synt_accr": token_accr.item(),
            "prog_synt_accr": program_accr.item(),
        }

    # @profile
    @th.inference_mode()
    def compute_metrics(self, batch: TensorDict):
        targets = batch["program"]  # (B, T)
        gt_len = targets.shape[1]

        logits = batch["logits"][:, :gt_len-1]  # (B, T-1, V)
        tf_preds = th.argmax(logits, dim=-1)  # (B, T-1)

        modes = {"tf": (tf_preds, targets[:, 1:])}
        if "generated_programs" in batch:
            modes["greedy"] = (batch["generated_programs"], targets)
        accr_dict = {}
        for mode in modes:
            pred, label = modes[mode]
            mask = (label != self.pad_token).float()  # (B, T)
            out_dict = self.compute_syntactic_accr(pred, label, mask)
            prefix = "tf_" if mode == "tf" else ""
            accr_dict.update({f"{prefix}{k}": v for k, v in out_dict.items()})

        if not self.model.training:
            sem_accr_dict = self.compute_semantic_accr(batch)
            accr_dict.update(sem_accr_dict)
        return accr_dict

    def move_batch(self, batch: TensorDict):
        for k, v in batch.items():
            if isinstance(v, th.Tensor):
                batch[k] = v.to(device=self.device, non_blocking=True)
        # Run grid one-hot expansion on GPU (compact ints came over the bus).
        # The dataset honors the same flag and skips its CPU-side expansion.
        # `needs_batched_expansion` lets each env_spec decide whether the
        # incoming s_h is still raw (Karel/List use a dtype heuristic).
        if self.config.get("defer_state_expansion") and "s_h" in batch:
            env_spec = self.config["env_spec"]
            if env_spec.needs_batched_expansion(batch["s_h"]):
                batch["s_h"] = env_spec.expand_state_batched(batch["s_h"])
        return batch

    # @profile
    def train(self, callback=None):

        print("Starting Training Loop...")
        self.iterations = 0
        self.model.train()
        self.best_stats = {}
        log_every = self.config["log_every"]
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs} started..")
            start = time()
            for batch in tqdm(self.train_loader):
                batch = self.move_batch(batch)
                batch = self.solve(batch, mode="train")

                # Only compute scalar metrics on logging iterations — avoids a
                # GPU→CPU sync (.item() in compute_metrics) on every step.
                is_log_step = (self.iterations % log_every == 0)
                loss_dict = self.update(batch, sync_loss=is_log_step)
                if is_log_step:
                    metrics = self.compute_metrics(batch)
                    log_dict = {
                        'LR': self.optimizer.param_groups[0]['lr'],
                        "loss": loss_dict,
                        **metrics
                    }
                    self.logger.record(log_dict, "train", tag="", iteration=self.iterations, roll_window=20)
                self.reset()
                self.iterations += 1

            elapsed = time() - start
            print(f"Epoch {epoch} ended! Time elapsed: {elapsed:.2f} seconds.")
            if th.cuda.is_available():
                peak_mb = th.cuda.max_memory_allocated() / 1024**2
                print(f"[mem] epoch {epoch} peak GPU alloc: {peak_mb:.1f} MB")
                th.cuda.reset_peak_memory_stats()
            self.on_epoch_end(epoch, tag="train")
            avg_solved = self.validate(epoch=epoch)
            if callback is not None:
                callback(epoch, avg_solved)

        # best_model is the best-by-fitness (exact program_total_match) checkpoint;
        # when exact-match plateaus at 0 (e.g. a weak encoder on RASP) it can freeze
        # an early, undertrained epoch. last_model is the final-epoch network so eval
        # can measure looser metrics (RASP behavioral-eq / parse-fail) on it.
        if self.config.get("save_last"):
            self.save_checkpoint("last_model")

    def log_csv(self, log_dict: dict, tag: str, epoch, flush=True):
        log_dict = {f"{tag}-{k}": v for k, v in log_dict.items()}
        if flush:
            log_dict = {**self.log_accum, **log_dict}
            self.csv_logger.log({**log_dict, "epoch": epoch})
            self.log_accum = {}
            return
        self.log_accum = {**self.log_accum, **log_dict}
    
    @th.inference_mode()
    def validate(self, epoch=None, save_model=True):
        tag = self.test_tag
        print(f"Validating on {tag} test set..")
        start = time()
        self.model.eval()
        stats = []
        for batch in tqdm(self.test_loader):
            batch = self.move_batch(batch)
            batch = self.solve(batch, mode="test")
            loss_dict = self.compute_loss(batch)
            loss_dict = {k: v.item() for k, v in loss_dict.items()}
            metrics = self.compute_metrics(batch)
            stats.append({"loss": loss_dict, "metrics": metrics})

        log_dict = self.parse_test_stats(stats)
        elapsed = time() - start
        print(f"Validation on {tag} test set ended! Elapsed time: {elapsed:.2f} seconds.")
        print(10*"=")
        self.logger.record(log_dict, "test", tag, iteration=self.iterations, roll_window=1)
        if not self.config["test_only"]:  # an evaluation leaves the run's training log alone
            self.log_csv(log_dict, tag, epoch)
        if save_model:
            for metric in self.fitness_metrics:
                assert metric in log_dict, f"Metric {metric} not found in log_dict"
            if len(self.best_stats) == 0:
                self.best_stats = log_dict
                self.save_checkpoint("best_model")
            else:
                for metric in self.fitness_metrics:
                    if log_dict[metric] > self.best_stats[metric]:
                        print(f"[INFO] Reached new best for {metric}: new={log_dict[metric]}, old={self.best_stats[metric]}")
                        self.best_stats = log_dict
                        self.save_checkpoint("best_model")
                        break
                    elif log_dict[metric] < self.best_stats[metric]:
                        break
        self.on_epoch_end(epoch, tag="test")
        self.model.train()
        return log_dict
    
    def load_checkpoint(self, path):
        if path is None:
            print("No checkpoint path provided. Skipping load.")
            return
        print(f"Loading checkpoint from {path}")
        checkpoint = th.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"]

        ckpt_vocab_w = state_dict.get("model.prog_decoder.fc_out.weight")
        if ckpt_vocab_w is not None and ckpt_vocab_w.size(0) != self.tokenizer.num_program_tokens:
            raise RuntimeError(
                f"Vocab size mismatch in prog_decoder.fc_out: ckpt={ckpt_vocab_w.size(0)}, "
                f"current={self.tokenizer.num_program_tokens}. "
                f"Did the tokenizer/environment change between phases?"
            )

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys from checkpoint: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")

        self._pending_opt_state = checkpoint.get("optimizer")

    def save_checkpoint(self, tag):
        dirpath = f'checkpoints/{self.config["run_name"]}'
        os.makedirs(dirpath, exist_ok=True)
        kpis = {m: self.best_stats[m] for m in self.fitness_metrics if m in self.best_stats}
        save_config_json(self.config, dirpath, extra={"_best_stats": kpis})
        mpath = f"{dirpath}/{tag}.pth"
        save_dict = {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        th.save(save_dict, mpath)
        print(f"Checkpoint saved to {mpath}")

    def parse_test_stats(self, stats):
        metrics = {
            k: [x["metrics"][k] for x in stats] for k in stats[0]["metrics"]
        }
        losses = {
            f"{k}": [x["loss"][k] for x in stats] for k in stats[0]["loss"]
        }
        log_dict_pre = {"Metric": metrics, "Loss": losses}
        log_dict = {}
        print(f"Test stats [{self.model_name}]:")
        for k in log_dict_pre:
            for k_ in log_dict_pre[k]:
                if len(log_dict_pre[k][k_]) > 0:
                    log_dict[f"{k_}"] = safer_mean(log_dict_pre[k][k_])
                    print(f"{k}/{k_}: {log_dict[f'{k_}']}")
        return log_dict

    def compute_loss(self, batch: TensorDict):
        loss_dict = {}
        total_loss = 0.0
        for k, loss_func in self.loss_funcs.items():
            loss_dict[k] = loss_func[0](batch)
            total_loss += loss_dict[k] * loss_func[1]
        loss_dict["total_loss"] = total_loss
        return loss_dict
    
    def reset_buffers(self):
        pass

    def update(self, batch: TensorDict, sync_loss: bool = True):
        loss_dict = self.compute_loss(batch)
        total_loss = loss_dict["total_loss"]
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm > 0:
            th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        if th.isnan(total_loss):
            print("="*50)
            print("NaN detected in loss. Skipping update.")
            print("="*50)
        else:
            self.optimizer.step()

        if sync_loss:
            for k, v in loss_dict.items():
                loss_dict[k] = v.item()
        self.reset_buffers()
        return loss_dict
    
    def reset(self, train: bool = True):
        pass

    def solve(self, batch: TensorDict, mode="train"):
        out_dict = self.model(batch)
        return out_dict
    
    def on_epoch_end(self, epoch, tag=""):
        pass

