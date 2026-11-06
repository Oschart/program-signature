import torch.nn as nn
from common.config_utils import TensorDict
from models.tracers import Tracer
from models.encoders import encoder_map
from models.decoders import TransformerDecoder


class SetEncoder(nn.Module):
    """Attention across the K trace embeddings of a program (--use_set_trans): (B, K, D) -> (B, K, D).
    Identity without --use_set_trans."""

    def __init__(self, config):
        super().__init__()
        if config["use_set_trans"]:
            layer = nn.TransformerEncoderLayer(config["hidden_dim"], config["num_heads"], batch_first=True)
            self.inner = nn.TransformerEncoder(layer, config["num_layers"])
        else:
            self.inner = nn.Identity()

    def forward(self, z):
        return self.inner(z)


class KarelSynthesizer(nn.Module):
    """Tracer (state encoder) -> trace encoder -> set encoder -> program decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.tracer = Tracer(config)
        # A list of one, so parameter names (trace_encoders.0.*) match the paper's checkpoints.
        self.trace_encoders = nn.ModuleList([encoder_map[config["trace_encoder"]](config)])
        self.set_encoder = SetEncoder(config)
        self.prog_decoder = TransformerDecoder(config)
        self.to(self.device)

    def forward(self, batch: TensorDict):
        self.tracer(batch)  # batch["traces"]: (B, K, T, D)
        trace_reps = self.trace_encoders[0](batch)  # (B, K, D)
        batch["trace_reps"] = self.set_encoder(trace_reps)
        batch.update(self.prog_decoder(batch))
        return batch
