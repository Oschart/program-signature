import numpy as np


class ListStateGenerator(object):
    def __init__(self, seed=None, L_max=20, max_val=19, min_init_len=1):
        self.rng = np.random.RandomState(seed)
        self.L_max = L_max
        self.max_val = max_val
        self.min_init_len = min_init_len

    def generate_single_state(self):
        init_len = self.rng.randint(self.min_init_len, self.L_max + 1)
        values = self.rng.randint(0, self.max_val + 1, size=init_len).astype(np.int8)
        cursor = 0
        return values, cursor
