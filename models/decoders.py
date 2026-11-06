import torch as th
import torch.nn as nn
from common.config_utils import TensorDict
from models.aux_nets import PositionalEncoding


class TransformerDecoder(nn.Module):
    """Writes the program token by token, cross-attending to the K trace embeddings (batch["trace_reps"]).
    Training: teacher forcing. Evaluation: greedy decoding, each step masked to the tokens the grammar allows."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        tokenizer = config["tokenizer"]
        self.syntax_checker = tokenizer.syntax_checker
        self.sos_token = tokenizer.T2I["DEF"]
        self.eos_token = tokenizer.T2I["m)"]
        self.pad_token = tokenizer.T2I["<pad>"]
        self.vocab_size = tokenizer.num_program_tokens
        self.len_leeway = 1.2
        self.max_program_len = int(self.len_leeway*self.config["max_program_len"])

        d_model = config["hidden_dim"]
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoder_prog = PositionalEncoding(self.config, d_model, max_len=self.max_program_len)
        # Always on, whatever --include_time says (that flag is about the traces).
        self.pos_encoder_prog.include_time = True
        # Cross-attention between the program (query) and the traces (key/value)
        dec_layer = nn.TransformerDecoderLayer(d_model, config["num_heads"], batch_first=True)
        self.program_decoder = nn.TransformerDecoder(dec_layer, config["num_layers"])
        self.fc_out = nn.Linear(d_model, self.vocab_size)

    def _get_syntax_mask(self, batch_size, current_tokens, mask_size, grammar_state):
        device = current_tokens.device
        neg_inf = -th.finfo(th.float32).max
        # Single host sync to pull all tokens at once, then advance grammar
        # states in Python (no per-element .item() / per-element .to(device)).
        tokens_list = current_tokens.tolist()
        masks = []
        for i, tok in enumerate(tokens_list):
            self.syntax_checker.forward(grammar_state[i], tok)
            # allowed_tokens returns the precomputed (1,1,V) bool mask cached
            # on the syntax checker (already on the right device when
            # use_cuda=True was passed at construction).
            masks.append(self.syntax_checker.allowed_tokens(grammar_state[i]).view(-1))
        infeasible = th.stack(masks).to(device, non_blocking=True)  # (B, V)
        syntax_mask = th.zeros((batch_size, mask_size), dtype=th.float32, device=device)
        syntax_mask.masked_fill_(infeasible, neg_inf)
        return syntax_mask, grammar_state

    def _generate_square_subsequent_mask(self, sz):
        # Creates a causal mask (triangular) so decoder can't look ahead
        mask = (th.triu(th.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _tf_decode(self, batch: TensorDict):
        trace_reps = batch["trace_reps"]
        program_tgt = batch["program"].int()
        tgt_emb = self.embedding(program_tgt)
        tgt_emb = self.pos_encoder_prog(tgt_emb)

        # Causal mask for the sequence (so index i can only see 0...i)
        L_prog = program_tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(L_prog).to(trace_reps.device)
        # Padding mask for the program (ignore pad tokens in calculation)
        tgt_key_padding_mask = (program_tgt == self.pad_token)

        output = self.program_decoder(
            tgt=tgt_emb,
            memory=trace_reps,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )
        return self.fc_out(output)

    @th.inference_mode()
    def _greedy_decode(self, batch: TensorDict):
        trace_reps = batch["trace_reps"]
        B = trace_reps.size(0)
        # Start every sequence with the start token (SOS)
        generated = th.full((B, 1), self.sos_token, dtype=th.long, device=trace_reps.device)
        grammar_state = [self.syntax_checker.get_initial_checker_state() for _ in range(B)]
        max_program_len = int(self.len_leeway*batch["program"].shape[1])
        for _ in range(max_program_len):
            tgt_emb = self.embedding(generated)
            tgt_emb = self.pos_encoder_prog(tgt_emb)
            # The causal mask is required even though only the last token is read: with >1 layer, earlier
            # positions would otherwise attend to later tokens, a computation never seen in training.
            tgt_mask = self._generate_square_subsequent_mask(generated.size(1)).to(trace_reps.device)
            out = self.program_decoder(tgt=tgt_emb, memory=trace_reps, tgt_mask=tgt_mask,
                                       memory_key_padding_mask=None)
            # Get logits for the last token only
            last_token_logits = self.fc_out(out[:, -1, :])
            last_token_logits = th.nan_to_num(last_token_logits, nan=0.0)
            syntax_mask, grammar_state = self._get_syntax_mask(B, generated[:, -1], self.vocab_size, grammar_state)
            last_token_logits = last_token_logits + syntax_mask

            next_token = th.argmax(last_token_logits, dim=-1).unsqueeze(1)
            generated = th.cat([generated, next_token], dim=1)

            if (next_token.squeeze(1) == self.eos_token).all():
                break

        return generated

    def forward(self, batch: TensorDict):
        ret_dict = {"logits": self._tf_decode(batch)}
        if not self.training:
            ret_dict["generated_programs"] = self._greedy_decode(batch)
        return ret_dict
