import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, kernel_size, in_feats):
        '''
        kernel_size: width of the kernels
        in_feats: number of channels in inputs
        '''
        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        out = self.relu(out)

        return out

class GridEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack, IMG_SIZE=(8,8,16)):
        '''
        kernel_size: width of the kernels
        conv_stack: Number of channels at each point of the convolutional part of
                    the network (includes the input)
        fc_stack: number of channels in the fully connected part of the network
        IMG_SIZE: (height, width, channels)
        '''
        super(GridEncoder, self).__init__()
        self.conv_layers = []
        for i in range(1, len(conv_stack)):
            if conv_stack[i-1] != conv_stack[i]:
                block = nn.Sequential(
                    ResBlock(kernel_size, conv_stack[i-1]),
                    nn.Conv2d(conv_stack[i-1], conv_stack[i],
                              kernel_size=kernel_size, padding=(kernel_size-1)//2 ),
                    nn.ReLU(inplace=True)
                )
            else:
                block = ResBlock(kernel_size, conv_stack[i-1])
            self.conv_layers.append(block)
            self.add_module("ConvBlock-" + str(i-1), self.conv_layers[-1])

        # We have operated so far to preserve all of the spatial dimensions so
        # we can estimate the flattened dimension.
        first_fc_dim = conv_stack[-1] * IMG_SIZE[0]* IMG_SIZE[1]
        adjusted_fc_stack = [first_fc_dim] + fc_stack
        self.fc_layers = []
        for i in range(1, len(adjusted_fc_stack)):
            self.fc_layers.append(nn.Linear(adjusted_fc_stack[i-1],
                                            adjusted_fc_stack[i]))
            self.add_module("FC-" + str(i-1), self.fc_layers[-1])
        # TODO: init?

    def forward(self, x):
        '''
        x: batch_size x channels x Height x Width
        '''
        batch_size = x.size(0)

        # Convolutional part
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten for the fully connected part
        x = x.view(batch_size, -1)
        # Fully connected part
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)

        return x


class ArrayStateEncoder(nn.Module):
    """Encoder for flat state vectors (List / RASP envs).

    `encoder_type='mlp'` (default): 2-layer Linear-ReLU-Linear (original behavior).
    `encoder_type='linear'`: single Linear projection. Useful for
    sign-bearing float states, where ReLU after the first projection can
    dead-zone half the input distribution before the linear has learned to
    fold signs.
    """
    def __init__(self, state_dim, hidden_dim, encoder_type='mlp'):
        super().__init__()
        if encoder_type == 'linear':
            self.net = nn.Linear(state_dim, hidden_dim)
        elif encoder_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}, expected 'mlp' or 'linear'")

    def forward(self, x):
        """x: (batch, state_dim) -> (batch, hidden_dim)"""
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, config, d_model, max_len=100):
        super().__init__()
        self.include_time = config["include_time"]
        self.d_model = d_model
        self.register_buffer("pe", self._make_pos_enc(max_len, d_model), persistent=False)

    @staticmethod
    def _make_pos_enc(length, d_model, device=None):
        pe = th.zeros(length, d_model, device=device)
        position = th.arange(0, length, dtype=th.float32, device=device).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2, dtype=th.float32, device=device) *
                          -(math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        return pe

    def _grow(self, length: int):
        new_len = max(length, self.pe.size(0) * 2)
        self.pe = self._make_pos_enc(new_len, self.d_model, device=self.pe.device).to(self.pe.dtype)

    def forward(self, x: th.Tensor):
        if not self.include_time:
            return x
        T = x.shape[-2]
        if T > self.pe.size(0):
            self._grow(T)
        return x + self.pe[:T]


class OneHotTimeConcat(nn.Module):
    """Concatenate fixed one-hot time channels to the path (--time_aug_mode=concat_onehot, RASP sig/logsig).

    Input x: (..., T, D) -> (..., T, D + max_len), e_t at step t. At depth 2 the time-feature Levy areas then
    contain every single increment of every feature channel, so the depth-2 (log)signature is a lossless
    encoding of the discrete path.
    """
    def __init__(self, max_len: int):
        super().__init__()
        self.register_buffer("table", th.eye(max_len), persistent=False)
        self.extra_dim = max_len

    def forward(self, x: th.Tensor) -> th.Tensor:
        T = x.shape[-2]
        if T > self.table.size(0):
            raise ValueError(f"Time table has {self.table.size(0)} positions but the trace has T={T}.")
        time_ch = self.table[:T].expand(*x.shape[:-2], T, self.extra_dim)
        return th.cat([x, time_ch], dim=-1)


class FrameNet(nn.Module):
    """Learned map applied identically at every position of a trace (--pre_enc_layers > 0, the RASP front end):
    an input layer in -> hidden, `layers` residual blocks [LayerNorm -> Linear -> ReLU -> Linear], then
    hidden -> out. x: (..., T, in_dim) -> (..., T, out_dim)."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, hidden, 1)  # a per-position Linear, stored as a width-1 conv
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        lead, (T, D) = x.shape[:-2], x.shape[-2:]
        h = self.inp(x.reshape(-1, T, D).transpose(1, 2)).transpose(1, 2)     # (N, T, hidden)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h).reshape(*lead, T, -1)


def compute_path_dim(config, pre_enc_dim: int) -> int:
    """Width of the time-augmented path the trace encoders read."""
    if config.get("include_time", False) and config.get("time_aug_mode", "add_sin") == "concat_onehot":
        return pre_enc_dim + config["max_demo_length"]
    return pre_enc_dim
