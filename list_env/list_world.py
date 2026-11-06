import numpy as np


action_table = {
    0: 'Cursor Left',
    1: 'Cursor Right',
    2: 'Push Zero',
    3: 'Pop',
    4: 'Swap Next',
    5: 'Increment Element',
    6: 'Decrement Element',
}
LIST_ACTIONS = ['cursorLeft', 'cursorRight', 'pushZero', 'pop', 'swapNext', 'incElem', 'decElem']  # ids as above


class ListWorld(object):
    """Variable-length list with a single cursor.

    Snapshot encoding: flat int8 vector of length (L_max + 2):
      positions [0, L_max)   -> list values; unused slots padded with `pad_value` = max_val + 1
      position  L_max        -> current length
      position  L_max + 1    -> cursor
    """
    def __init__(self, values=None, cursor=0, L_max=20, max_val=19, monitor_a=None):
        self.L_max = L_max
        self.max_val = max_val
        self.pad_value = max_val + 1
        self.num_actions = len(LIST_ACTIONS)
        self.monitor_a = monitor_a
        if values is not None:
            self.set_new_state(values, cursor)

    def set_new_state(self, values, cursor=0):
        values = np.asarray(values, dtype=np.int8).copy()
        assert values.ndim == 1, "values must be 1D"
        assert values.shape[0] <= self.L_max, f"initial length exceeds L_max={self.L_max}"
        self.buffer = np.full(self.L_max, self.pad_value, dtype=np.int8)
        self.length = int(values.shape[0])
        self.buffer[:self.length] = values
        self.cursor = max(0, min(int(cursor), max(0, self.length - 1)))
        self.s_h = [self._snapshot()]
        self.a_h = []

    def clear_history(self):
        self.s_h = [self._snapshot()]
        self.a_h = []

    def _snapshot(self):
        return np.concatenate(
            [self.buffer.copy(), [self.length, self.cursor]]
        ).astype(np.int8)

    def add_to_history(self, a_idx):
        self.s_h.append(self._snapshot())
        self.a_h.append(a_idx)

    def state_transition(self, a):
        a_idx = int(np.argmax(a))

        if self.monitor_a is not None:
            step = len(self.a_h)
            if step >= len(self.monitor_a):
                raise RuntimeError(f"Action overrun: monitor exhausted at step {step}, got extra action {a_idx}")
            if a_idx != self.monitor_a[step]:
                raise RuntimeError(f"Action mismatch: expected {self.monitor_a[step]}, got {a_idx}")

        self._apply(a_idx)
        self.add_to_history(a_idx)

    def _apply(self, a_idx):
        if a_idx == 0:      # cursorLeft
            self.cursor = max(self.cursor - 1, 0)
        elif a_idx == 1:    # cursorRight
            if self.length > 0:
                self.cursor = min(self.cursor + 1, self.length - 1)
        elif a_idx == 2:    # pushZero — insert 0 at cursor
            if self.length < self.L_max:
                self.buffer[self.cursor + 1:self.length + 1] = self.buffer[self.cursor:self.length]
                self.buffer[self.cursor] = 0
                self.length += 1
        elif a_idx == 3:    # pop — remove element at cursor
            if self.length > 0:
                self.buffer[self.cursor:self.length - 1] = self.buffer[self.cursor + 1:self.length]
                self.buffer[self.length - 1] = self.pad_value
                self.length -= 1
                self.cursor = max(0, min(self.cursor, max(0, self.length - 1)))
        elif a_idx == 4:    # swapNext
            if self.cursor + 1 < self.length:
                tmp = int(self.buffer[self.cursor])
                self.buffer[self.cursor] = self.buffer[self.cursor + 1]
                self.buffer[self.cursor + 1] = tmp
        elif a_idx == 5:    # incElem
            if self.length > 0:
                self.buffer[self.cursor] = min(int(self.buffer[self.cursor]) + 1, self.max_val)
        elif a_idx == 6:    # decElem
            if self.length > 0:
                self.buffer[self.cursor] = max(int(self.buffer[self.cursor]) - 1, 0)

    # ---- Condition methods (snake_case; called by DSL via lambdas) ----

    def cursor_at_start(self):
        return self.cursor == 0

    def cursor_at_end(self):
        return self.length == 0 or self.cursor == self.length - 1

    def list_empty(self):
        return self.length == 0

    def elem_is_zero(self):
        return self.length > 0 and int(self.buffer[self.cursor]) == 0

    def elem_lt_next(self):
        return (self.cursor + 1 < self.length and
                int(self.buffer[self.cursor]) < int(self.buffer[self.cursor + 1]))

    # ---- Perception API (Karel-compatible interface) ----

    def get_perception_vector(self):
        return np.array([
            self.cursor_at_start(),
            self.cursor_at_end(),
            self.list_empty(),
            self.elem_is_zero(),
            self.elem_lt_next(),
        ], dtype=bool)

    def get_perception_list(self):
        return ['cursorAtStart', 'cursorAtEnd', 'listEmpty', 'elemIsZero', 'elemLtNext']
