"""
Task generator for the State Surgery substrate.

Design constraint (non-negotiable, see research log Sec. 8.3):
every task draws a FRESH random colour permutation. With 9 non-background
colours that is log2(9!) ~= 18.5 bits of task-specific content that cannot
have been memorised during training, because that exact permutation has
(almost surely) never been seen. This converts the claim from
"the model recognises the rule" to "the model binds a novel mapping".

A task rule is:      y = perm( op(x) )
  op   in {identity, hflip, vflip, rot180, transpose}   (structure)
  perm : a permutation of colours 1..9, with 0 (background) fixed  (binding)

The oracle is deterministic and ships with the generator, so ground truth
is always available beside the model estimate.
"""

import numpy as np

H = W = 6
N_COLORS = 10          # 0 = background, 1..9 paintable
N_PAINT = 9
OPS = ["identity", "hflip", "vflip", "rot180", "transpose"]
N_OPS = len(OPS)

ROLE_DEMO_IN, ROLE_DEMO_OUT, ROLE_QUERY = 0, 1, 2
MAX_DEMOS = 4


# ----------------------------------------------------------------------------
# structural operations
# ----------------------------------------------------------------------------

def apply_op(g, op):
    if op == "identity":
        return g.copy()
    if op == "hflip":
        return g[:, ::-1].copy()
    if op == "vflip":
        return g[::-1, :].copy()
    if op == "rot180":
        return g[::-1, ::-1].copy()
    if op == "transpose":
        return g.T.copy()
    raise ValueError(op)


def apply_rule(g, op, perm):
    """perm is an int array of length 10 with perm[0] == 0."""
    return perm[apply_op(g, op)]


def random_perm(rng):
    p = np.arange(N_COLORS, dtype=np.int64)
    p[1:] = rng.permutation(np.arange(1, N_COLORS))
    return p


# ----------------------------------------------------------------------------
# grids
# ----------------------------------------------------------------------------

def random_grid(rng, palette, fill=0.42):
    g = np.zeros((H, W), dtype=np.int64)
    m = rng.random((H, W)) < fill
    n = int(m.sum())
    if n < 3:                      # avoid near-empty grids
        idx = rng.choice(H * W, size=3, replace=False)
        m = np.zeros(H * W, dtype=bool)
        m[idx] = True
        m = m.reshape(H, W)
        n = 3
    g[m] = rng.choice(palette, size=n)
    return g


def _plant_missing(rng, grids, palette):
    """Guarantee every palette colour appears somewhere in `grids`."""
    present = set()
    for g in grids:
        present |= set(np.unique(g).tolist())
    for c in palette:
        if c not in present:
            gi = int(rng.integers(len(grids)))
            r, cc = int(rng.integers(H)), int(rng.integers(W))
            grids[gi][r, cc] = c
    return grids


# ----------------------------------------------------------------------------
# task construction
# ----------------------------------------------------------------------------

def make_task(rng, n_demos=3, n_queries=3, palette_size=None,
              extra_query_colors=0, op=None, perm=None, ops=None,
              palette=None):
    """
    extra_query_colors > 0 builds the UNCOVERED condition: the query grids
    contain colours that never appear in any demonstration input, so the
    binding for those colours cannot exist in the state. Used for the
    coverage-cliff experiment. Training always uses extra_query_colors=0.
    """
    if palette_size is None:
        palette_size = int(rng.integers(3, 7))
    palette_size = min(palette_size, N_PAINT - extra_query_colors)

    if palette is not None:
        palette = np.asarray(palette)
        rest = np.array([c for c in range(1, N_COLORS) if c not in set(palette.tolist())])
        extra = np.sort(rng.permutation(rest)[:extra_query_colors])
    else:
        all_paint = rng.permutation(np.arange(1, N_COLORS))
        palette = np.sort(all_paint[:palette_size])
        extra = np.sort(all_paint[palette_size:palette_size + extra_query_colors])

    if op is None:
        pool = ops if ops else OPS
        op = pool[int(rng.integers(len(pool)))]
    if perm is None:
        perm = random_perm(rng)

    demo_in = [random_grid(rng, palette) for _ in range(n_demos)]
    demo_in = _plant_missing(rng, demo_in, palette)
    demos = [(x, apply_rule(x, op, perm)) for x in demo_in]

    q_palette = np.concatenate([palette, extra]) if extra_query_colors else palette
    q_in = [random_grid(rng, q_palette) for _ in range(n_queries)]
    if extra_query_colors:
        q_in = _plant_missing(rng, q_in, extra)   # force the uncovered colour to appear
    queries = [(x, apply_rule(x, op, perm)) for x in q_in]

    return {
        "op": op,
        "perm": perm,
        "palette": palette,
        "extra": extra,
        "demos": demos,
        "queries": queries,
    }


# ----------------------------------------------------------------------------
# tensorisation
# ----------------------------------------------------------------------------

_RR, _CC = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
_RR = _RR.reshape(-1)
_CC = _CC.reshape(-1)
L = H * W


def _pair_tokens(x, y, demo_idx):
    """One token per cell carrying BOTH the input and output colour.

    This is a tokenisation choice, disclosed in the README: the two grids of a
    demonstration are presented cell-aligned, as two channels of one stream.
    It does not reveal the rule -- the permutation is fresh per task and must
    still be inferred from co-occurrence across cells. It restricts the model
    to shape- and position-preserving transformations, which is why the
    trained rule family is the colour-binding family (BDH-CQ Sec. 6.3).
    """
    return np.stack([x.reshape(-1), y.reshape(-1), _RR, _CC,
                     np.full(L, demo_idx, dtype=np.int64)], axis=1)


def _grid_tokens(g, role, demo_idx):
    return np.stack([g.reshape(-1),
                     np.zeros(L, dtype=np.int64),
                     _RR, _CC,
                     np.full(L, role, dtype=np.int64)], axis=1)


def memory_tokens(task, n_demos=None):
    """(T,5): colour_in, colour_out, row, col, demo_idx."""
    demos = task["demos"] if n_demos is None else task["demos"][:n_demos]
    if not demos:
        return np.zeros((0, 5), dtype=np.int64)
    return np.concatenate([_pair_tokens(x, y, i)
                           for i, (x, y) in enumerate(demos)], axis=0)


def query_tokens(grid):
    """(L,5): colour, 0, row, col, role."""
    return _grid_tokens(grid, ROLE_QUERY, 0)


def batch_from_tasks(tasks, n_demos=3, query_idx=0):
    mem = np.stack([memory_tokens(t, n_demos) for t in tasks]) if n_demos > 0 \
        else np.zeros((len(tasks), 0, 5), dtype=np.int64)
    qs = np.stack([query_tokens(t["queries"][query_idx][0]) for t in tasks])
    ys = np.stack([t["queries"][query_idx][1].reshape(-1) for t in tasks])
    return mem, qs, ys


def sample_batch(rng, bs, n_demos=3, n_queries=1, **kw):
    tasks = [make_task(rng, n_demos=max(n_demos, 1), n_queries=n_queries, **kw)
             for _ in range(bs)]
    return tasks, batch_from_tasks(tasks, n_demos=n_demos, query_idx=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = make_task(rng)
    print("op:", t["op"], "palette:", t["palette"])
    print("perm:", t["perm"])
    x, y = t["demos"][0]
    print(x, "\n->\n", y)
    assert np.array_equal(apply_rule(x, t["op"], t["perm"]), y)
    m, q, yy = batch_from_tasks([t])
    print("mem tokens", m.shape, "query", q.shape, "target", yy.shape)
    u = make_task(rng, extra_query_colors=2)
    seen = set()
    for a, b in u["demos"]:
        seen |= set(np.unique(a).tolist())
    qseen = set(np.unique(u["queries"][0][0]).tolist())
    print("uncovered colours in query:", sorted(qseen - seen))
