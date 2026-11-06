import torch as th
import torch.nn as nn
from common.config_utils import TensorDict
from models.aux_nets import GridEncoder, ArrayStateEncoder

from common.utils import repeat_last_valid


class Tracer(nn.Module):
    """Embeds every state of every trace into hidden_dim channels: batch["traces"] (B, K, T, hidden_dim) and
    batch["trace_lens"] (B, K). Karel grids go through a CNN; List and RASP states through an MLP or a single
    Linear (--state_encoder_type)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config["hidden_dim"]
        self.is_grid = config["environment"] not in ("list", "rasp")
        if self.is_grid:
            self._make_grid_encoder()
        else:
            state_dim = self.config["state_shape"][0]
            encoder_type = self.config.get("state_encoder_type", "mlp")
            self.state_encoder = ArrayStateEncoder(state_dim, self.hidden_dim, encoder_type=encoder_type)

    def _make_grid_encoder(self):
        img_size = self.config["img_size"]  # (W, H, channels): 16 for Karel
        kernel_size = 3
        conv_stack = (64,64,64)
        fc_stack = [self.hidden_dim]
        cnn = GridEncoder(kernel_size, conv_stack, fc_stack, IMG_SIZE=img_size)
        self.state_encoder = nn.Sequential(
            nn.Conv2d(img_size[2], 64, kernel_size=kernel_size, padding=(kernel_size -1)//2), nn.ReLU(inplace=True),
            cnn)

    def _encode_states(self, s_h, lens):
        """Encode state tensors through the state encoder.
        Grid input: (B, K, T, C, H, W) -> (B, K, T, hidden_dim)
        Flat input: (B, K, T, D) -> (B, K, T, hidden_dim)
        The collate pads each trace by repeating its last state, so padded frames encode to copies
        of the last valid frame. For grid states only the valid frames go through the CNN and the last
        one is repeated into the tail: same output, ~11% less CNN time on len-15-20 Karel. Flat states
        encode every frame: the MLP is cheaper than selecting the valid frames.
        """
        trailing = s_h.shape[-3:] if self.is_grid else s_h.shape[-1:]
        batch_dims = s_h.shape[:-len(trailing)]
        if not self.is_grid:
            flat_out = self.state_encoder(s_h.reshape(-1, *trailing))
            return flat_out.view(*batch_dims, -1)
        valid = th.arange(batch_dims[-1], device=s_h.device) < lens.unsqueeze(-1)  # (B, K, T)
        enc = self.state_encoder(s_h[valid])  # (num_valid, hidden_dim)
        out = enc.new_zeros(*batch_dims, enc.shape[-1])
        out[valid] = enc
        return repeat_last_valid(out, lens, dim=len(batch_dims) - 1)

    def forward(self, batch: TensorDict):
        batch["traces"] = self._encode_states(batch["s_h"], batch["s_h_len"])  # (B, K, T, hidden_dim)
        batch["trace_lens"] = batch["s_h_len"]
        return batch["traces"]
