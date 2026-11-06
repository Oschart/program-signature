import os
# os.environ["LINE_PROFILE"] = "1"
from common.config_utils import BaseOptions
from trainers.trace2code import Trace2Code
from trainers.trace2code_bs2 import Trace2CodeBS2

import torch as th
import numpy as np
import random

# Workaround for PyTorch's fused SDPA backward producing NaN
# (ScaledDotProductEfficientAttentionBackward0). Force the math kernel.
# Opt out by setting SDPA_FUSED=1.
if os.environ.get("SDPA_FUSED", "1") != "1" and th.cuda.is_available():
    th.backends.cuda.enable_flash_sdp(False)
    th.backends.cuda.enable_mem_efficient_sdp(False)

if __name__ == "__main__":
    print("Starting training script...")
    config = BaseOptions().parse()
    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Print torch and cuda version
    print(f"Torch version: {th.__version__}")

    if config["solver"] == "trace2code":
        trainer = Trace2Code(config)
    elif config["solver"] == "trace2code_bs2":
        trainer = Trace2CodeBS2(config)
    else:
        raise NotImplementedError(f"Solver {config['solver']} not implemented.")

    def callback(epoch, avg_solved):
        if avg_solved[trainer.fitness_metrics[0]] == 1.0:
            print(f"All problems solved at epoch {epoch}. Stopping training.")
            exit(0)

    print("=" * 80)
    print("Final model:")
    print(trainer.model)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Params: {n_train:,} trainable / {n_total:,} total")
    print("=" * 80)

    if config["test_only"]:
        trainer.validate(epoch=0, save_model=False)  # evaluate --checkpoint on the test set (metrics printed)
    else:
        trainer.train(callback=callback)