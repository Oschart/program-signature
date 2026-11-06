from typing import Dict
import torch as th
import numpy as np
import argparse
import json
import os
import yaml


def pretty_print_config(args: dict, title: str = "Run configs") -> None:
    """Print the final merged config as sorted, indented JSON. Redacts wandb_key."""
    redacted = dict(args)
    if redacted.get("wandb_key"):
        redacted["wandb_key"] = "***REDACTED***"
    body = json.dumps(redacted, indent=2, sort_keys=True, default=str)
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}\n{body}\n{bar}\n")

TensorDict = Dict[str, th.Tensor]


def parse_lr(lr_str):
    """"constant(1e-4)" -> 1e-4. The learning rate is constant over training."""
    schedule_type, base_lr = lr_str.split("(")
    if schedule_type != "constant":
        raise ValueError(f"--lr_schedule={lr_str}: only constant(<lr>) is supported")
    return float(base_lr[:-1])


def safer_mean(arr, default=0):
    # Check if object is iterable
    if not hasattr(arr, '__len__'):
        return arr
    if len(arr) == 0:
        return default
    vals = []
    for v in arr:
        if isinstance(v, th.Tensor):
            v = v.cpu().tolist()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        if not isinstance(v, list):
            v = [v]
        vals.extend(v)
    return np.mean(vals)


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Run identity and logging
        self.parser.add_argument("--project_name", type=str, default="ProgSig")
        self.parser.add_argument("--exp_name", type=str, default="default_exp")
        self.parser.add_argument("--exp_desc", type=str, default="")
        self.parser.add_argument("--seed", type=int, default=2023)
        self.parser.add_argument("--no_wandb", action='store_true')
        self.parser.add_argument("--wandb_key", type=str, default=None)
        self.parser.add_argument("--log_every", type=int, default=20)
        self.parser.add_argument("--overwrite_logs", action='store_true')
        self.parser.add_argument("--device", type=str, default="cuda")

        # Training
        self.parser.add_argument("--solver", type=str, default="trace2code", choices=["trace2code", "trace2code_bs2"])
        self.parser.add_argument('--configfile', help='Input file for parameters, constants and initial settings')
        self.parser.add_argument("--n_epochs", type=int, default=50)
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--lr_schedule", type=str, default="constant(1e-4)")
        self.parser.add_argument("--max_grad_norm", type=float, default=0.5)
        self.parser.add_argument("--weight_decay", type=float, default=0.0,
                                 help="Decoupled (AdamW) weight decay lambda on every weight matrix; 0 = plain Adam.")
        self.parser.add_argument("--save_last", action='store_true',
                                 help="Also save last_model.pth (final-epoch checkpoint) at end of "
                                      "training, alongside best_model.pth (best-by-fitness).")
        self.parser.add_argument("--test_only", action='store_true')
        self.parser.add_argument("--checkpoint", type=str, default=None)

        # Data
        self.parser.add_argument('--datadir', help='dataset directory under data/')
        self.parser.add_argument("--num_demos", type=int, default=5)
        self.parser.add_argument("--num_demos_test", type=int, default=None,
                                 help="Number of demos for validation/test (defaults to --num_demos)")
        self.parser.add_argument("--size_cap", type=int, default=None)
        self.parser.add_argument("--defer_state_expansion", action='store_true',
                                 help="Defer Karel grid one-hot expansion from dataloader workers to GPU (in trainer.move_batch). "
                                      "Shrinks host→device transfer ~16x at the cost of one batched scatter per step on GPU.")
        self.parser.add_argument("--num_workers", type=int, default=4,
                                 help="DataLoader worker processes. Set to 0 on macOS (spawn cannot pickle the DSL).")
        self.parser.add_argument("--prefetch_factor", type=int, default=2,
                                 help="DataLoader prefetch_factor (only used when --num_workers > 0).")

        # Model
        self.parser.add_argument("--hidden_dim", type=int, default=256)
        self.parser.add_argument(
            "--state_encoder_type", type=str, default="mlp", choices=["mlp", "linear"],
            help="Per-frame encoder for flat-state envs (List/RASP). 'mlp' (default) = 2-layer Linear-ReLU-Linear; 'linear' = single Linear (preserves sign info, used for RASP's float states). No effect on grid envs (Karel)."
        )
        self.parser.add_argument("--trace_encoder", type=str, default="logsig",
                                 choices=["logsig", "sig", "lstm", "transformer", "fourier"])
        self.parser.add_argument("--use_set_trans", action='store_true',
                                 help="Attention across the K trace embeddings of a program before decoding.")
        self.parser.add_argument("--num_heads", type=int, default=8,
                                 help="Attention heads of the Transformer encoder, set encoder and decoder.")
        self.parser.add_argument("--num_layers", type=int, default=2,
                                 help="Layers of the Transformer encoder, set encoder and decoder.")
        self.parser.add_argument("--fourier_modes", type=int, default=5)
        self.parser.add_argument("--sig_depth", type=int, default=2)
        self.parser.add_argument("--sig_step_norm", type=str, default="none", choices=["none", "unit"],
                                 help="sig/logsig only: 'unit' rebuilds the embedded path from its steps with every "
                                      "step's length set to 1, direction kept, before the time code.")
        self.parser.add_argument("--pre_enc_dim", type=int, default=None,
                                 help="Width of the embedded path (default: --hidden_dim).")
        self.parser.add_argument("--pre_enc_layers", type=int, default=0,
                                 help="0 (default): pre_enc is one Linear. N > 0: a learned per-position network "
                                      "(FrameNet: input layer, N residual MLP blocks of width --pre_enc_hidden, "
                                      "output layer of width --pre_enc_dim) applied before the trace encoder.")
        self.parser.add_argument("--pre_enc_hidden", type=int, default=512,
                                 help="Hidden width of the --pre_enc_layers network.")
        self.parser.add_argument("--include_time", action='store_true')
        self.parser.add_argument("--time_aug_mode", type=str, default="add_sin", choices=["add_sin", "concat_onehot"],
                                 help="Time augmentation (only consulted when --include_time is set). "
                                      "add_sin: sinusoidal PE added to the path channels. "
                                      "concat_onehot: fixed one-hot(t) channels concatenated (T extra channels).")

        # Bootstrapping — used by --solver=trace2code_bs2 only.
        self.parser.add_argument("--bs2_freq", type=int, default=5,
                                 help="Run a bootstrap pass every N epochs (trace2code_bs2 only).")
        self.parser.add_argument("--bs2_warmup", type=int, default=5,
                                 help="Skip bootstrap until epoch >= bs2_warmup, then trigger when (epoch - bs2_warmup) %% bs2_freq == 0.")
        self.parser.add_argument("--bs2_model_tag", type=str, default=None,
                                 help="Short tag (e.g. logsig) appended to the saved-to-disk reduced dataset directory: data/<datadir>-bs-<tag>/. Required when --solver=trace2code_bs2.")

    def expand_config(self, args: dict):
        # Load config yaml file
        cfg_path = f"configs/{args['configfile']}.yml"
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file {cfg_path} does not exist.")
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        # Override with command line args
        for key, value in config.items():
            if key not in args or args[key] is None:
                args[key] = value
            else:
                print(f"Overriding config value for {key}: {value} -> {args[key]}")
        return args

    def parse(self):
        args, unknown = self.parser.parse_known_args()
        if unknown:
            print(f"Warning: unknown arguments will be ignored: {unknown}")
        # Convert to dictionary
        args = vars(args)
        # Expand config
        args = self.expand_config(args)

        if args["num_demos_test"] is None:
            args["num_demos_test"] = args["num_demos"]

        args["base_lr"] = parse_lr(args["lr_schedule"])
        if args["test_only"]:
            args["overwrite_logs"] = False  # an evaluation must not wipe the run's training log

        if args["wandb_key"] is None and os.path.exists("wandb_api_key.txt"):
            # Read wandb API key from file (else wandb uses WANDB_API_KEY / its own login)
            with open("wandb_api_key.txt", "r") as f:
                args["wandb_key"] = f.read().strip()

        args["run_name"] = f"{args['exp_name']}-r{args['seed']}"
        args["log_dir"] = f"logs/{args['run_name']}"
        os.makedirs(args["log_dir"], exist_ok=True)

        datadir = args["datadir"]
        if datadir.startswith("list"):
            args["environment"] = "list"
        elif datadir.startswith("rasp"):
            args["environment"] = "rasp"
        else:
            args["environment"] = "karel"

        from karel_env.dsl import get_DSL, dsl_version_for
        args["dsl_version"] = dsl_version_for(args["datadir"])
        _dsl_tmp = get_DSL(environment=args["environment"], version=args["dsl_version"])
        args["dsl_tokens"] = list(_dsl_tmp.int2token)

        pretty_print_config(args, title="Run configs")
        print(f"Logging to {args['log_dir']}")
        self.config = args
        return args
