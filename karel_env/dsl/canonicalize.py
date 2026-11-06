"""Partial canonicalization of DSL programs (shared imperative grammar).

Every rewrite is TRACE-equivalent: the program emits exactly the same action
sequence (and crashes at the same point) on every start state. End-state-only
equivalences such as `turnLeft turnLeft turnLeft` == `turnRight` are deliberately
NOT rewritten — they produce different traces, so a model can tell them apart.

Rules (applied bottom-up until nothing changes):
  REPEAT R=0 (X)                    -> (nothing)
  REPEAT R=1 (X)                    -> X
  REPEAT R=a (REPEAT R=b X)         -> REPEAT R=a*b X          if a*b <= 19
  REPEAT R=n (X)                    -> X ... X (n copies)      if not longer in tokens
  REPEAT R=n / IF c / WHILE c (WHILE c X), same c for IF/WHILE -> WHILE c X
  IFELSE not c (A) ELSE (B)         -> IFELSE c (B) ELSE (A)
  IFELSE c (A) ELSE (A)             -> A
  IFELSE c (A X) ELSE (B X)         -> IFELSE c (A) ELSE (B) X   (shared tail hoisted)
  IFELSE c () ELSE (B)              -> IF not c (B);   IFELSE c (A) ELSE () -> IF c (A)
  IF c () / REPEAT R=n ()           -> (nothing)
A program with no canonical form (empty program, WHILE with an empty body) maps to None.

Nodes: ('A', tok) | ('REPEAT', n, body) | ('IF', cond, body) | ('WHILE', cond, body)
       | ('IFELSE', cond, then, else); cond = ('x',) or ('not', 'x'); body = list of nodes.
"""

MAX_REPEAT = 19
_OPEN = {'IF': 'i(', 'WHILE': 'w('}


def _block(toks, i):
    """toks[i] is an opening bracket; return (inner tokens, index after its match)."""
    depth = 0
    for j in range(i, len(toks)):
        if toks[j].endswith('('):
            depth += 1
        elif toks[j].endswith(')'):
            depth -= 1
            if depth == 0:
                return toks[i + 1:j], j + 1
    raise ValueError(f"unbalanced brackets at {i}: {' '.join(toks)}")


def _cond(toks):
    return tuple(t for t in toks if not t.endswith(('(', ')')))


def parse_body(toks):
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t == 'REPEAT':
            n = int(toks[i + 1][2:])
            body, i = _block(toks, i + 2)
            out.append(('REPEAT', n, parse_body(body)))
        elif t in ('IF', 'WHILE'):
            cond, j = _block(toks, i + 1)
            body, i = _block(toks, j)
            out.append((t, _cond(cond), parse_body(body)))
        elif t == 'IFELSE':
            cond, j = _block(toks, i + 1)
            then, j = _block(toks, j)
            els, i = _block(toks, j + 1)  # j points at ELSE
            out.append(('IFELSE', _cond(cond), parse_body(then), parse_body(els)))
        else:
            out.append(('A', t))
            i += 1
    return out


def _cond_toks(cond):
    inner = [cond[0]] if len(cond) == 1 else ['not', 'c(', cond[1], 'c)']
    return ['c('] + inner + ['c)']


def body_tokens(body):
    out = []
    for n in body:
        k = n[0]
        if k == 'A':
            out.append(n[1])
        elif k == 'REPEAT':
            out += ['REPEAT', f'R={n[1]}', 'r('] + body_tokens(n[2]) + ['r)']
        elif k in _OPEN:
            close = _OPEN[k][0] + ')'
            out += [k] + _cond_toks(n[1]) + [_OPEN[k]] + body_tokens(n[2]) + [close]
        else:
            out += (['IFELSE'] + _cond_toks(n[1]) + ['i('] + body_tokens(n[2]) + ['i)', 'ELSE', 'e(']
                    + body_tokens(n[3]) + ['e)'])
    return out


def parse_program(code):
    toks = code.split()
    if toks[:3] != ['DEF', 'run', 'm('] or toks[-1] != 'm)':
        raise ValueError(f"not a program: {code}")
    return parse_body(toks[3:-1])


def program_code(body):
    return ' '.join(['DEF', 'run', 'm('] + body_tokens(body) + ['m)'])


class _NoCanonicalForm(Exception):
    pass


def _negate(cond):
    return (cond[1],) if cond[0] == 'not' else ('not', cond[0])


def _rewrite_node(n):
    """Canonical replacement for one node, as a (possibly empty) list of nodes."""
    k = n[0]
    if k == 'A':
        return [n]
    if k == 'IFELSE':
        cond, then, els = n[1], _rewrite_body(n[2]), _rewrite_body(n[3])
        if cond[0] == 'not':
            cond, then, els = cond[1:], els, then
        if then == els:
            return then
        tail = []
        while then and els and then[-1] == els[-1]:
            tail.insert(0, then.pop())
            els.pop()
        if not then:
            return [('IF', _negate(cond), els)] + tail
        if not els:
            return [('IF', cond, then)] + tail
        return [('IFELSE', cond, then, els)] + tail
    body = _rewrite_body(n[2])
    if k == 'WHILE':
        if not body:
            raise _NoCanonicalForm('WHILE with empty body')
        if len(body) == 1 and body[0][0] == 'WHILE' and body[0][1] == n[1]:
            return body
        return [('WHILE', n[1], body)]
    if not body:  # IF c () and REPEAT R=n () do nothing
        return []
    if k == 'IF':
        if len(body) == 1 and body[0][0] == 'WHILE' and body[0][1] == n[1]:
            return body
        return [('IF', n[1], body)]
    reps = n[1]  # REPEAT
    if reps == 0:
        return []
    if reps == 1:
        return body
    if len(body) == 1 and body[0][0] == 'WHILE':
        return body
    if len(body) == 1 and body[0][0] == 'REPEAT' and reps * body[0][1] <= MAX_REPEAT:
        return [('REPEAT', reps * body[0][1], body[0][2])]
    size = len(body_tokens(body))
    if reps * size <= size + 4:  # unrolled form is not longer than the REPEAT
        return body * reps
    return [('REPEAT', reps, body)]


def _rewrite_body(body):
    return [m for n in body for m in _rewrite_node(n)]


def canonicalize(code, max_passes=20):
    """Trace-equivalent canonical form of `code`, or None if it has none."""
    body = parse_program(code)
    try:
        for _ in range(max_passes):
            new = _rewrite_body(body)
            if new == body:
                break
            body = new
        else:
            raise RuntimeError(f"canonicalize did not converge: {code}")
    except _NoCanonicalForm:
        return None
    return program_code(body) if body else None
