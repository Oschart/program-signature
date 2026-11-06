import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from common.config_utils import TensorDict
from common import signatures as custom_sig
from common.utils import repeat_last_valid
from models.aux_nets import PositionalEncoding, OneHotTimeConcat, FrameNet, compute_path_dim
from typing import Dict


class TraceEncoder(nn.Module):
    """Encodes each of the K traces of a program: batch["traces"] (B, K, T, hidden_dim) -> (B, K, hidden_dim).
    Shared front end: pre_enc (one Linear, or FrameNet with --pre_enc_layers), the unit-step rebuild
    (--sig_step_norm=unit, sig/logsig only) and the time code (--include_time), giving batch["pre_enc_traces"]."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config["hidden_dim"]
        # Width of batch["traces"]: the tracer embeds every state into hidden_dim channels.
        self.feat_dim = self.hidden_dim

        self._make_pre_enc()
        self._make_time_aug(config)

        self.sig_step_norm = config.get("sig_step_norm", "none") or "none"
        if self.sig_step_norm != "none" and not isinstance(self, SigEncoder):
            raise ValueError(f"--sig_step_norm is for the sig/logsig encoders, not {type(self).__name__}")

    def _make_pre_enc(self):
        if self.config.get("pre_enc_dim", None) is not None:
            self.pre_enc_dim = self.config["pre_enc_dim"]
        else:
            self.pre_enc_dim = self.config["hidden_dim"]
        layers = self.config.get("pre_enc_layers", 0) or 0
        if layers > 0:      # learned per-position network (FrameNet) instead of the single Linear
            self.pre_enc = FrameNet(self.feat_dim, self.config.get("pre_enc_hidden", 512), self.pre_enc_dim, layers)
        else:
            self.pre_enc = nn.Linear(self.feat_dim, self.pre_enc_dim)

    def _make_time_aug(self, config):
        max_len = config["max_demo_length"]
        if config.get("include_time", False) and config.get("time_aug_mode", "add_sin") == "concat_onehot":
            self.pos_enc = OneHotTimeConcat(max_len)
        else:
            # Also the no-time case: PositionalEncoding is a no-op unless include_time,
            # and the Transformer encoder forces it on.
            self.pos_enc = PositionalEncoding(config, self.pre_enc_dim, max_len)
        self.path_dim = compute_path_dim(config, self.pre_enc_dim)

    def encode(self, batch: TensorDict):
        raise NotImplementedError

    _STEP_EPS = 1e-3  # smooths |step| so a zero step stays zero with a finite gradient

    def _normalize_steps(self, traces, lens):
        """--sig_step_norm=unit: rebuild the path from its steps with each step's length set to 1, direction
        kept, so a few long steps (List push/pop) do not dominate the signature. Steps past each trace's
        length become zero (flat tail). No-op for none.
        traces: (B, K, T, D), lens: (B, K)."""
        if self.sig_step_norm == "none":
            return traces
        steps = traces[..., 1:, :] - traces[..., :-1, :]
        valid = th.arange(steps.shape[-2], device=traces.device) < (lens - 1).unsqueeze(-1)  # (B, K, T-1)
        length = (steps.pow(2).sum(-1, keepdim=True) + self._STEP_EPS ** 2).sqrt()
        steps = steps * length.pow(-1.0) * valid.unsqueeze(-1).to(steps.dtype)
        return th.cat([traces[..., :1, :], traces[..., :1, :] + steps.cumsum(dim=-2)], dim=-2)

    def forward(self, batch: TensorDict):
        traces = self.pre_enc(batch["traces"])  # (B, K, T, D)
        traces = self._normalize_steps(traces, batch["trace_lens"])  # before time, so time stays a clock
        traces = self.pos_enc(traces)  # (B, K, T, D)
        batch["pre_enc_traces"] = traces
        return self.encode(batch)  # (B, K, D)


class LstmEncoder(TraceEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(self.path_dim, self.hidden_dim, batch_first=True)
        self.pre_decode = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(self, batch: TensorDict):
        traces = batch["pre_enc_traces"]  # (B, K, T, D)
        t_lens = batch["trace_lens"]  # (B, K)
        B, K, T = traces.shape[:3]

        # Unfold for LSTM
        lstm_in = traces.view(B * K, T, -1) # (B * K, T, D)
        flat_lens = t_lens.view(-1).cpu()
        # Pack needs the sequences sorted by length, longest first
        lengths_sorted, sorted_idx = flat_lens.sort(descending=True)
        x_sorted = lstm_in[sorted_idx]
        packed_input = pack_padded_sequence(
            x_sorted,
            lengths_sorted,
            batch_first=True,
            enforce_sorted=True
        )
        _, (h_n, _) = self.lstm(packed_input)
        # Restore the original order
        _, original_idx = sorted_idx.sort()
        h_last = h_n[-1, original_idx]  # (B*K, D)

        trace_reps = self.pre_decode(h_last)
        trace_reps = trace_reps.view(B, K, -1)
        return trace_reps


class TransformerEncoder(TraceEncoder):
    def __init__(self, config):
        super().__init__(config)
        if isinstance(self.pos_enc, PositionalEncoding) and not self.pos_enc.include_time:
            print(f"Warning: TransformerEncoder forces include_time=True. Overriding config setting.")
            self.pos_enc.include_time = True  # Force positional encoding

        temp_layer = nn.TransformerEncoderLayer(d_model=self.path_dim, nhead=config["num_heads"], batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temp_layer, num_layers=config["num_layers"])
        self.post_enc = nn.Linear(self.path_dim, self.hidden_dim)

    def _create_padding_mask(self, lengths, max_len):
        """Boolean mask, True at padding. Shape: (Batch_Size, Max_Len)."""
        range_tensor = th.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_tensor >= lengths.unsqueeze(1)

    def masked_mean_pooling(self, x, mask):
        """
        Pools over the time dimension (T), ignoring padded elements.
        x: (B*K, T, D)
        mask: (B*K, T) - True where padding exists
        """
        valid_mask_expanded = (~mask).unsqueeze(-1).float()  # (B*K, T, 1)
        sum_embeddings = th.sum(x * valid_mask_expanded, dim=1)
        # Count valid lengths (avoid division by zero with clamp)
        valid_counts = valid_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / valid_counts

    def encode(self, batch: TensorDict):
        traces = batch["pre_enc_traces"]  # (B, K, T, D)
        t_lens = batch["trace_lens"]  # (B, K)
        B, K, T, D = traces.shape

        traces_flat = traces.view(B * K, T, D)
        lengths_flat = t_lens.view(B * K)

        # PyTorch transformers expect True for PADDING: (B*K, T)
        temporal_mask = self._create_padding_mask(lengths_flat, T)
        x = self.temporal_encoder(traces_flat, src_key_padding_mask=temporal_mask)
        x_pooled = self.masked_mean_pooling(x, temporal_mask)  # (B*K, D)

        trace_reps = x_pooled.view(B, K, -1)
        trace_reps = self.post_enc(trace_reps)
        return trace_reps


class SigEncoder(TraceEncoder):
    """Truncated signature of the embedded path (level --sig_depth), read by one Linear (post_sig)."""
    def __init__(self, config):
        super().__init__(config)
        self.depth = self.config["sig_depth"]
        self._make_transform()
        self.post_sig = nn.Linear(self.sig_dim, self.hidden_dim)

    def _make_transform(self):
        self.sig_fn = custom_sig.Signature(depth=self.depth)
        self.sig_dim = custom_sig.signature_channels(self.path_dim, self.depth)

    def encode(self, batch: TensorDict):
        traces = batch["pre_enc_traces"] # trace: (B, K, T, D)
        B, K, T, D = traces.shape
        traces = traces.view(B * K, T, D)
        # Pad by repeating the last state: a zero step leaves the signature unchanged, a jump to zero would not.
        traces = repeat_last_valid(traces, batch["trace_lens"].view(-1), dim=1)
        if traces.shape[1] < 2:
            # Signature of a single point is not well-defined, so we duplicate the single point to create a valid path
            traces = traces.repeat(1, 2, 1)
        sig_out = self.sig_fn(traces).view(B, K, -1)  # (B, K, sig_dim)
        return self.post_sig(sig_out)                # (B, K, D)


class LogsigEncoder(SigEncoder):
    def _make_transform(self):
        self.sig_fn = custom_sig.LogSignature(depth=self.depth)
        self.sig_dim = custom_sig.logsignature_channels(self.path_dim, self.depth)


class FourierEncoder(TraceEncoder):
    """The first --fourier_modes frequencies of the orthonormal real DFT of the embedded path, real and
    imaginary parts stacked, read by one Linear."""
    def __init__(self, config):
        super().__init__(config)
        self.modes = config["fourier_modes"]
        self.signal_dim = self.modes * self.path_dim * 2
        self.min_T = (self.modes - 1)*2  # shortest path with that many rfft modes
        self.post_fourier = nn.Linear(self.signal_dim, self.hidden_dim)

    def encode(self, batch: TensorDict):
        traces = batch["pre_enc_traces"] # trace: (B, K, T, D)
        B, K, T, D = traces.shape

        target_T = max(T, self.min_T)
        pad_amount = target_T - T
        traces = F.pad(traces, (0, 0, 0, pad_amount), mode='constant', value=0)
        traces = traces.view(B * K, target_T, D)

        x_freq = th.fft.rfft(traces, dim=1, norm='ortho')
        x_freq = x_freq[:, :self.modes]
        sig = th.cat([x_freq.real, x_freq.imag], dim=-1) # signal
        sig = sig.view(B, K, -1)
        return self.post_fourier(sig)


encoder_map: Dict[str, TraceEncoder] = {
    "lstm": LstmEncoder,
    "transformer": TransformerEncoder,
    "sig": SigEncoder,
    "logsig": LogsigEncoder,
    "fourier": FourierEncoder,
}
