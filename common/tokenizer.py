
import copy

from karel_env.dsl import get_DSL
from karel_env.tool.syntax_checker import PySyntaxChecker, acts as checker_acts, bool_check as checker_bool_check


class Tokenizer:
    def __init__(self, config):
        self.config = config
        self.dsl = get_DSL(seed=config["seed"], environment=config["environment"],
                           version=config.get("dsl_version", 1))
        self.use_simplified_dsl = config["dsl"]["use_simplified_dsl"]

        self.syntax_checker_tokens = copy.copy(config['dsl_tokens'])
        self.T2I = {token: i for i, token in enumerate(self.syntax_checker_tokens)}
        self.T2I['<pad>'] = len(self.syntax_checker_tokens)
        self.I2T = {i: token for token, i in self.T2I.items()}

        self.pad_token = self.T2I['<pad>']

        self.num_agent_actions = len(self.dsl.action_functions) + 1      # +1 for a no-op action, just for filling
        self.max_program_len = config["dsl"]["max_program_len"] + 1
        self.num_program_tokens = len(self.T2I)
        # Add syntax checker to mask out invalid tokens
        # SOS is 0:DEF, EOS is m), padding is <pad>
        if config["environment"] == "rasp":
            # RASP is functional — no imperative Karel-style grammar state
            # machine. Mask decoding to prefixes of generator-family programs, so
            # only programs the generator could have produced can be emitted.
            from rasp_env.rasp_grammar import RaspGrammarChecker
            self.syntax_checker = RaspGrammarChecker(self.T2I)
        else:
            # The checker's token tables are hard-coded: a DSL action or condition missing from them is
            # masked out of every decode (the program can never be emitted), so fail at construction.
            missing = [t for t in self.dsl.action_functions if t not in checker_acts] + \
                      [t for t in self.dsl.conditional_functions if t not in checker_bool_check]
            if missing:
                raise ValueError(f"karel_env/tool/syntax_checker.py does not know the {config['environment']} tokens "
                                 f"{missing}, so the decoder could never emit them: add them to acts / bool_check.")
            self.syntax_checker = PySyntaxChecker(self.T2I, use_cuda='cuda' in config['device'],
                                                      use_simplified_dsl=self.use_simplified_dsl,
                                                      new_tokens=self.syntax_checker_tokens)


    def encode_program(self, prog_node):
        tokens = ["<SOS>", "DEF", "main", "m("] + prog_node.to_tokens() + [")", "<EOS>"]
        return [self.token2idx.get(t, 0) for t in tokens]

    def encode(self, tokens):
        # Check unknown tokens
        for t in tokens:
            if t not in self.token2idx:
                raise ValueError(f"Unknown token: {t}")
        return [self.token2idx[t] for t in tokens]
    
    def decode(self, indices):
        return [self.I2T[i] for i in indices]