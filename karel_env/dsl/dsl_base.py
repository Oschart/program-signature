"""
Domain specific language for Karel Environment

Code is adapted from https://github.com/carpedm20/karel
"""

import numpy as np
import ply.lex as lex
from functools import wraps

from third_party import yacc

MIN_INT = 0
MAX_INT = 19
INT_PREFIX = 'R='


class DSLBase(object):

    def get_yacc(self):
        # Per-env tabmodule so concurrent jobs (e.g. karel + list generators
        # on the same NFS-mounted repo) don't thrash a shared _parsetab.py.
        # Each grammar signature differs across envs; sharing one cache file
        # causes mid-write corruption when one job rewrites it under another's
        # workers (race produces "module '_parsetab' has no attribute
        # '_tabversion'" on import).
        env = getattr(self, 'environment', None)
        if env is None:
            tabmodule = "_parsetab"
        else:
            tabmodule = "_parsetab_" + env.replace('-', '_').replace('.', '_')
            if getattr(self, 'version', 1) != 1:  # different grammar -> own cache file
                tabmodule += f"_v{self.version}"
        self.yacc, self.grammar = yacc.yacc(
            module=self,
            tabmodule=tabmodule,
            with_grammar=True)

    def __init__(self, seed=None):
        self.lexer = lex.lex(module=self)
        self.get_yacc()

        self.prodnames = self.grammar.Prodnames
        self.call_counter = [0]
        self.max_func_call = 220
        self.rng = np.random.RandomState(seed)

        self.construct_vocab()

        def callout(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                if self.call_counter[0] > self.max_func_call:
                    raise RuntimeError("Program execution timeout.")
                r = f(*args, **kwargs)
                self.call_counter[0] += 1
                return r
            return wrapped

        self.callout = callout

    def construct_vocab(self):
        self.token2int = []
        self.int2token = []
        for term in self.tokens:
            token = getattr(self, 't_{}'.format(term))
            if callable(token):
                if token == self.t_INT:
                    for i in range(MIN_INT, MAX_INT + 1):
                        self.int2token.append("{}{}".format(INT_PREFIX, i))
            else:
                self.int2token.append(str(token).replace('\\', ''))
        self.token2int = {v: i for i, v in enumerate(self.int2token)}

    def strseq2intseq(self, code):
        return [self.token2int[t] for t in code]
    
    def str2intseq(self, code):
        return [self.token2int[t] for t in code.split()]

    def code2intseq(self, code):
        return [self.token2int[t] for t in code.split()]

    def intseq2str(self, intseq):
        if max(intseq) < len(self.int2token):
            return ' '.join([self.int2token[i] for i in intseq])
        else:
            # intseq contains a termination token
            program_str = []
            for i in intseq:
                if i < len(self.int2token):
                    program_str.append(self.int2token[i])
                else:
                    break
            return ' '.join(program_str)
    
    def intseq2tkns(self, intseq):
        return [self.int2token[i] for i in intseq]

    conditional_functions = []

    action_functions = []

    #########
    # lexer
    #########

    def t_error(self, t):
        t.lexer.skip(1)
        raise RuntimeError('Syntax Error')

    #########
    # parser
    #########

    def p_error(self, p):
        raise RuntimeError('Syntax Error')

    def random_code(self, start_token="prog", depth=0, max_depth=6, nesting_depth=0, max_nesting_depth=4):
        code = " ".join(self.random_tokens(start_token, depth, max_depth, nesting_depth, max_nesting_depth))

        return code

    def parse(self, code, **kwargs):
        self.call_counter = [0]
        self.error = False
        program = self.yacc.parse(code, **kwargs)
        return program

    def run(self, world, code, **kwargs):
        self.call_counter = [0]
        program = self.parse(code, **kwargs)

        # run program
        world.clear_history()
        program(world)
        return world.s_h
    
    def run_parsed(self, world, program):
        self.call_counter = [0]
        # run program
        world.clear_history()
        program(world)
        return world.s_h
