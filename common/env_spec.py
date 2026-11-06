"""
Environment specification strategy for multi-environment support (Karel, List, RASP).

Each EnvSpec encapsulates environment-specific behavior for:
- Dataset dimension parsing from datadir names / HDF5 metadata
- State tensor expansion (sparse grid descriptors vs raw vectors)
- State preparation for semantic verification
- World object creation for program re-execution
"""

import numpy as np
import torch


def _expand_grid_desc(grid_desc, img_dims):
    """Expand sparse grid descriptors to dense one-hot images."""
    img_feat = img_dims[0] * img_dims[1] * img_dims[2]
    K, T, D = grid_desc.shape
    grid = torch.zeros((K, T, img_feat), dtype=torch.float32)
    # scatter_ requires int64 indices; tolerate int16/int32 storage from the converter.
    grid.scatter_(2, grid_desc.long(), 1.0)
    grid = grid.view(K, T, *img_dims)
    return grid


class EnvSpec:
    """Abstract environment specification."""

    name: str

    def parse_data_dims(self, datadir):
        raise NotImplementedError

    @property
    def state_shape(self):
        raise NotImplementedError

    def expand_state(self, s_h):
        raise NotImplementedError

    def needs_batched_expansion(self, s_h) -> bool:
        # Default heuristic: raw on-disk states are integer-typed (Karel uint8
        # grid descriptors, List int8); expanded states are float.
        return not s_h.is_floating_point()

    def prepare_state_for_verification(self, s_h):
        raise NotImplementedError

    def make_world(self, s_0, monitor_a=None):
        raise NotImplementedError


class KarelEnvSpec(EnvSpec):
    name = "karel"

    def parse_data_dims(self, datadir):
        data_spec = datadir.split('-')[0]
        data_dims = data_spec[5:].split('x')  # "karel8x8" -> ["8", "8"]
        self.width = int(data_dims[0])
        self.height = int(data_dims[1])
        self._state_shape = (self.width, self.height, 16)

    @property
    def state_shape(self):
        return self._state_shape

    def expand_state(self, s_h):
        s_h = _expand_grid_desc(s_h, self._state_shape)  # (K, T, W, H, 16)
        s_h = s_h.permute(0, 1, 4, 2, 3)  # (K, T, 16, W, H)
        return s_h

    def expand_state_batched(self, s_h):
        # Batched, device-preserving variant of expand_state.
        # s_h: (B, K, T, D) int → (B, K, T, C, W, H) float32 on the same device.
        # Use this from the trainer's move_batch so the heavy scatter happens
        # on GPU instead of in dataloader workers (and host→device transfer
        # carries the compact int tensor instead of the 16× larger float volume).
        img_dims = self._state_shape  # (W, H, C)
        img_feat = img_dims[0] * img_dims[1] * img_dims[2]
        B, K, T, D = s_h.shape
        grid = torch.zeros((B, K, T, img_feat), dtype=torch.float32, device=s_h.device)
        grid.scatter_(3, s_h.long(), 1.0)
        grid = grid.view(B, K, T, *img_dims)         # (B, K, T, W, H, C)
        grid = grid.permute(0, 1, 2, 5, 3, 4)        # (B, K, T, C, W, H)
        return grid.contiguous()

    def prepare_state_for_verification(self, s_h):
        # (B, K, T, C, H, W) -> (B, K, T, H, W, C)
        return s_h.permute(0, 1, 2, 4, 5, 3)

    def make_world(self, s_0, monitor_a=None):
        import karel_env.karel as karel
        world = karel.Karel_world(monitor_a=monitor_a)
        world.set_new_state(s_0)
        return world


class ListEnvSpec(EnvSpec):
    """A state is the stored [buffer (L_max), length, cursor] ints, read as floats."""
    name = "list"

    def parse_data_dims(self, datadir):
        # Parse from datadir name: "list_l20v19_...".
        # Match parts like "l20v19" — leading 'l', digits, 'v', digits.
        import re
        prefix = datadir.split('-')[0]
        parts = prefix.split('_')
        for part in parts:
            m = re.fullmatch(r'l(\d+)v(\d+)', part)
            if m:
                self.L_max = int(m.group(1))
                self.max_val = int(m.group(2))
                break
        else:
            raise ValueError(f"Cannot parse list dimensions from datadir: {datadir}")
        self._state_shape = (self.L_max + 2,)

    @property
    def state_shape(self):
        return self._state_shape

    def expand_state(self, s_h):
        return s_h.float()

    def expand_state_batched(self, s_h):
        return self.expand_state(s_h)

    def prepare_state_for_verification(self, s_h):
        return s_h

    def make_world(self, s_0, monitor_a=None):
        from list_env.list_world import ListWorld
        # s_0 layout: [buffer (L_max), length, cursor]
        buffer = s_0[:self.L_max]
        length = int(s_0[-2])
        cursor = int(s_0[-1])
        values = np.asarray(buffer[:length], dtype=np.int8)
        world = ListWorld(L_max=self.L_max, max_val=self.max_val, monitor_a=monitor_a)
        world.set_new_state(values, cursor)
        return world


class RaspEnvSpec(EnvSpec):
    """RASP env. A state is a flat vector of width d_model: the input and output sequences of a sweep
    frame, d_model = 2L, read from data/<datadir>/rasp_data_info.json (written by the generator)."""
    name = "rasp"

    def parse_data_dims(self, datadir):
        import json
        import os
        meta_path = os.path.join("data", datadir, "rasp_data_info.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} is missing (written by rasp_env.sweep_generator)")
        with open(meta_path) as f:
            meta = json.load(f)
        d_model = int(meta["d_model"])
        max_seq_len = int(meta["max_seq_len"])
        vocab_size = int(meta["vocab_size"])
        from rasp_env.rasp_grammar import RASP_FAMILY   # the token vocab is built from it at import
        if meta.get("family", "base") != RASP_FAMILY:
            raise ValueError(f"{datadir} holds the {meta.get('family', 'base')!r} RASP family but "
                             f"RASP_FAMILY={RASP_FAMILY!r}: export RASP_FAMILY before starting.")
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self._state_shape = (d_model,)

    @property
    def state_shape(self):
        return self._state_shape

    def expand_state(self, s_h):
        return s_h.float()

    def expand_state_batched(self, s_h):
        return s_h.float()

    def needs_batched_expansion(self, s_h) -> bool:
        # Float-on-disk and expand_state is a no-op cast, so this is always
        # False (trailing dim already == d_model). Explicit for clarity.
        return s_h.shape[-1] != self._state_shape[0]

    def prepare_state_for_verification(self, s_h):
        return s_h

    def make_world(self, s_0, monitor_a=None):
        raise NotImplementedError(
            "RASP has no imperative world to re-execute: a decoded program is checked by evaluating it "
            "on the stored inputs (Trace2Code._compute_rasp_semantic).")


def get_env_spec(environment):
    specs = {
        "karel": KarelEnvSpec,
        "list": ListEnvSpec,
        "rasp": RaspEnvSpec,
    }
    if environment not in specs:
        raise ValueError(f"Unknown environment: {environment}. Must be one of {list(specs.keys())}")
    return specs[environment]()
