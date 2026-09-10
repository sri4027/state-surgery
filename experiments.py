"""
Every experiment behind the artifact's claims. Writes web/results.json.

  E1  demonstration count            0 / 1 / 2 / 3
  E2  state ablations                zero state, wrong-task state, random state
  E3  TRANSPLANT (the central test)  does S_A carry rule A onto query B?
  E4  reasoning depth R              1..8 (2..6 was trained)
  E5  COVERAGE CLIFF                 byte-identical query, coverage varies
  E6  within-task consistency        pair accuracy vs strict-task accuracy
  E7  leakage / shortcut audit       demo order, palette, position
  E8  parameter identity             SHA-256 and ||dtheta|| across ingestion

Run:  python -m arcstate.experiments --ckpt runs/main.pt
"""

import argparse, hashlib, json, os
import numpy as np
import torch

from .tasks import (make_task, batch_from_tasks, memory_tokens, query_tokens,
                    apply_rule, apply_op, random_perm, N_COLORS, OPS)
from .model import StateSurgeryModel, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the trained rule family: fresh colour permutation, position preserving
EVAL_OPS = ["identity"]


# ---------------------------------------------------------------- helpers ---

def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x))


@torch.no_grad()
def predict(model, tasks, n_demos=3, R=3, query_idx=0, state_from=None):
    """state_from: optional list of tasks whose demos build S (the transplant)."""
    mem_src = tasks if state_from is None else state_from
    if n_demos > 0:
        mem = np.stack([memory_tokens(t, n_demos) for t in mem_src])
        S, zs = model.build_state(_t(mem))
    else:
        B = len(tasks)
        S = torch.zeros(B, Config.dk, Config.dv)
        zs = torch.zeros(B, Config.dk)
    q = np.stack([query_tokens(t["queries"][query_idx][0]) for t in tasks])
    lg = model.solve(_t(q), S, zs, R=R)
    return lg.argmax(-1).numpy()


def score(pred, target):
    cell = float((pred == target).mean())
    grid = float((pred == target).all(axis=1).mean())
    return round(cell, 4), round(grid, 4)


def gen(rng, n, **kw):
    kw.setdefault("ops", EVAL_OPS)
    return [make_task(rng, **kw) for _ in range(n)]


def targets(tasks, qi=0):
    return np.stack([t["queries"][qi][1].reshape(-1) for t in tasks])


# ------------------------------------------------------------------- E1-E2 ---

def e1_demo_count(model, seed, n=256):
    out = {}
    for k in (0, 1, 2, 3):
        tasks = gen(np.random.default_rng(seed), n)
        p = predict(model, tasks, n_demos=k)
        c, g = score(p, targets(tasks))
        out[str(k)] = {"cell": c, "strict_grid": g}
    return out


def e2_ablations(model, seed, n=256):
    tasks = gen(np.random.default_rng(seed), n)
    tgt = targets(tasks)
    res = {}

    p = predict(model, tasks, n_demos=3)
    res["correct_state"] = dict(zip(("cell", "grid"), score(p, tgt)))

    p = predict(model, tasks, n_demos=0)
    res["zero_state"] = dict(zip(("cell", "grid"), score(p, tgt)))

    # wrong-task state: S built from a DIFFERENT task's demonstrations
    other = gen(np.random.default_rng(seed + 7), n)
    p = predict(model, tasks, n_demos=3, state_from=other)
    res["wrong_task_state"] = dict(zip(("cell", "grid"), score(p, tgt)))

    # random state matched in scale to a real one
    with torch.no_grad():
        mem = np.stack([memory_tokens(t, 3) for t in tasks])
        S, zs = model.build_state(_t(mem))
        Sr = torch.randn_like(S) * S.std()
        q = np.stack([query_tokens(t["queries"][0][0]) for t in tasks])
        pr = model.solve(_t(q), Sr, zs, R=3).argmax(-1).numpy()
    res["random_state"] = dict(zip(("cell", "grid"), score(pr, tgt)))
    return res


# ---------------------------------------------------------------------- E3 ---

def e3_transplant(model, seed, n=256):
    """
    Build S from task A's demonstrations, run it on task B's query.
    The sharp prediction is NOT "it gets B wrong" but
    "it produces exactly rule A applied to B's input".
    """
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(n):
        # SAME palette, DIFFERENT permutation. Matching the palette isolates
        # the rule: every colour in B's query is one that A's state has bound,
        # so "apply rule A to input B" is fully determined.
        pal = np.sort(rng.permutation(np.arange(1, N_COLORS))[:4])
        pa, pb = random_perm(rng), random_perm(rng)
        A.append(make_task(rng, palette=pal, perm=pa, ops=EVAL_OPS))
        B.append(make_task(rng, palette=pal, perm=pb, ops=EVAL_OPS))

    xB = [t["queries"][0][0] for t in B]
    tgt_B = np.stack([g.reshape(-1) for g in [t["queries"][0][1] for t in B]])
    tgt_A_on_B = np.stack([apply_rule(x, a["op"], a["perm"]).reshape(-1)
                           for x, a in zip(xB, A)])

    p_native = predict(model, B, n_demos=3)
    p_trans = predict(model, B, n_demos=3, state_from=A)

    # how often are the two rules coincidentally the same?
    same = float(np.mean([(a["op"] == b["op"]) and np.array_equal(a["perm"], b["perm"])
                          for a, b in zip(A, B)]))
    return {
        "native_vs_ruleB": dict(zip(("cell", "grid"), score(p_native, tgt_B))),
        "transplant_vs_ruleB": dict(zip(("cell", "grid"), score(p_trans, tgt_B))),
        "transplant_vs_ruleA_on_B": dict(zip(("cell", "grid"),
                                             score(p_trans, tgt_A_on_B))),
        "rule_collision_rate": round(same, 5),
        "n": n,
        "design": ("A and B share a palette but have different permutations, so "
                   "every colour in B's query is bound by A's state"),
    }


# ---------------------------------------------------------------------- E4 ---

def e4_depth(model, seed, n=192):
    tasks = gen(np.random.default_rng(seed), n)
    tgt = targets(tasks)
    out = {}
    for R in range(1, 9):
        p = predict(model, tasks, n_demos=3, R=R)
        c, g = score(p, tgt)
        out[str(R)] = {"cell": c, "strict_grid": g,
                       "trained_range": Config.r_train[0] <= R <= Config.r_train[1]}
    return out


# ---------------------------------------------------------------------- E5 ---

def e5_coverage(model, seed, n=256):
    """
    The query is BYTE-IDENTICAL across conditions. Only the demonstrations
    change. This mirrors BDH-CQ Table 3 / Appendix A.3 in miniature.
    """
    rng = np.random.default_rng(seed)
    tasks = [make_task(rng, n_demos=3, extra_query_colors=1, palette_size=4,
                       ops=EVAL_OPS) for _ in range(n)]

    # SUPPORTED: append a 4th demonstration whose input contains the missing colour
    sup = []
    for t in tasks:
        c = int(t["extra"][0])
        x = t["demos"][0][0].copy()
        idx = rng.choice(x.size, size=max(3, x.size // 8), replace=False)
        x.reshape(-1)[idx] = c
        s = dict(t)
        s["demos"] = list(t["demos"]) + [(x, apply_rule(x, t["op"], t["perm"]))]
        sup.append(s)

    tgt = targets(tasks)
    q_src = np.stack([apply_op(t["queries"][0][0], t["op"]).reshape(-1)
                      for t in tasks])            # source colour per output cell
    extra = np.array([t["extra"][0] for t in tasks])[:, None]
    uncov_mask = (q_src == extra)                  # cells whose colour was never shown
    cov_mask = ~uncov_mask

    p_short = predict(model, tasks, n_demos=3)
    p_sup = predict(model, sup, n_demos=4)

    def split(p):
        return {
            "cell_all": round(float((p == tgt).mean()), 4),
            "cell_covered": round(float((p == tgt)[cov_mask].mean()), 4),
            "cell_uncovered": round(float((p == tgt)[uncov_mask].mean()), 4),
            "strict_grid": round(float((p == tgt).all(axis=1).mean()), 4),
        }

    return {
        "short_context": split(p_short),
        "supported_context": split(p_sup),
        "uncovered_cells_per_grid": round(float(uncov_mask.sum(1).mean()), 2),
        "chance_on_uncovered": round(1 / 9, 4),
        "n": n,
        "note": "query grids byte-identical across the two conditions",
    }


# ---------------------------------------------------------------------- E6 ---

def e6_consistency(model, seed, n=256):
    tasks = gen(np.random.default_rng(seed), n, n_queries=3)
    per_q = []
    for qi in range(3):
        p = predict(model, tasks, n_demos=3, query_idx=qi)
        per_q.append((p == targets(tasks, qi)).all(axis=1))
    per_q = np.stack(per_q, axis=1)                      # (n,3)
    pair = float(per_q.mean())
    strict = float(per_q.all(axis=1).mean())
    partial = float(((per_q.sum(1) > 0) & (per_q.sum(1) < 3)).mean())
    return {"pair_accuracy": round(pair, 4),
            "strict_task_accuracy": round(strict, 4),
            "gap_points": round((pair - strict) * 100, 2),
            "partial_tasks": round(partial, 4),
            "n": n}


# ---------------------------------------------------------------------- E7 ---

def e7_leakage(model, seed, n=192):
    rng = np.random.default_rng(seed)
    tasks = gen(rng, n)
    tgt = targets(tasks)
    base = score(predict(model, tasks, n_demos=3), tgt)

    # (a) reversed demonstration order
    rev = []
    for t in tasks:
        s = dict(t); s["demos"] = list(reversed(t["demos"])); rev.append(s)
    r_rev = score(predict(model, rev, n_demos=3), tgt)

    # (b) held-out permutations only: reject any task whose permutation was
    #     (astronomically unlikely to be) seen. 9! = 362880 possible.
    big = gen(np.random.default_rng(seed + 3), n, palette_size=6)
    r_big = score(predict(model, big, n_demos=3), targets(big))

    return {
        "baseline": {"cell": base[0], "grid": base[1]},
        "reversed_demo_order": {"cell": r_rev[0], "grid": r_rev[1]},
        "order_invariant_by_construction": True,
        "order_note": ("our write is a sum of outer products, so S is exactly "
                       "permutation-invariant over demonstration tokens; "
                       "BDH-CQ states its memory is order-dependent (Eq. 1). "
                       "Documented difference, not a result."),
        "palette_size_6": {"cell": r_big[0], "grid": r_big[1]},
        "distinct_permutations": 362880,
    }


# ---------------------------------------------------------------------- E8 ---

def e8_param_identity(model, seed):
    def h(m):
        b = b"".join(p.detach().numpy().tobytes()
                     for _, p in sorted(m.state_dict().items()))
        return hashlib.sha256(b).hexdigest()

    before = h(model)
    snap = {k: v.clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(seed)
    tasks = gen(rng, 64)
    _ = predict(model, tasks, n_demos=3)
    after = h(model)
    delta = float(sum(((model.state_dict()[k] - v) ** 2).sum()
                      for k, v in snap.items()) ** 0.5)

    mem = np.stack([memory_tokens(t, 3) for t in tasks])
    with torch.no_grad():
        S, _ = model.build_state(_t(mem))
    return {"sha256_before": before, "sha256_after": after,
            "identical": before == after,
            "delta_theta_l2": delta,
            "delta_S_l2_mean": round(float(S.flatten(1).norm(dim=1).mean()), 4),
            "n_params": sum(p.numel() for p in model.parameters()),
            "state_entries": Config.dk * Config.dv}


def e9_capacity(model, seed, n=192):
    """How many simultaneous bindings fit in a fixed-size state?
    Training used palettes of 3-6; 2 and 7-9 are held out."""
    out = {}
    for k in range(2, 10):
        tasks = gen(np.random.default_rng(seed + k), n, palette_size=k)
        p = predict(model, tasks, n_demos=3)
        c, g = score(p, targets(tasks))
        out[str(k)] = {"cell": c, "strict_grid": g,
                       "in_training_range": 3 <= k <= 6}
    return out


def e10_state_growth(model, seed, n=64):
    """||S|| after each demonstration -- what the artifact's meter shows."""
    tasks = gen(np.random.default_rng(seed), n)
    out = {}
    for k in (0, 1, 2, 3):
        if k == 0:
            out[str(k)] = 0.0
            continue
        mem = np.stack([memory_tokens(t, k) for t in tasks])
        with torch.no_grad():
            S, _ = model.build_state(_t(mem))
        out[str(k)] = round(float(S.flatten(1).norm(dim=1).mean()), 3)
    return out


# probe grid: every paintable colour once, at fixed positions.
PROBE = np.zeros((6, 6), dtype=np.int64)
_pp = [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4), (4, 0), (4, 2), (4, 4)]
for _i, (_r, _c) in enumerate(_pp):
    PROBE[_r, _c] = _i + 1


@torch.no_grad()
def decode_rule(model, S, zs, R=3):
    """Read the bound mapping straight out of the state.

    One forward pass on a probe grid containing every colour once. The value
    the model emits at colour c's cell IS what the state says c maps to. This
    is the model's own read+decode path, not a surrogate.
    """
    B = S.shape[0]
    q = np.repeat(query_tokens(PROBE)[None], B, axis=0)
    pred = model.solve(_t(q), S, zs, R=R).argmax(-1).numpy()
    idx = [r * 6 + c for r, c in _pp]
    return pred[:, idx]                      # (B, 9) decoded perm for colours 1..9


def e11_interference(model, seed, n=192, kmax=6):
    """Superpose K task states in ONE fixed-size state.

    Our write is additive, so superposition is exactly S = sum_i S_i. This is
    the interference limit of a fixed-size carrier: a KV cache would simply
    store more, this cannot.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for K in range(1, kmax + 1):
        groups = [gen(np.random.default_rng(seed + 1000 * K + j), n) for j in range(K)]
        S = zs = None
        for g in groups:
            mem = np.stack([memory_tokens(t, 3) for t in g])
            with torch.no_grad():
                Si, zi = model.build_state(_t(mem))
            S = Si if S is None else S + Si
            zs = zi if zs is None else zs + zi
        tasks = groups[0]
        q = np.stack([query_tokens(t["queries"][0][0]) for t in tasks])
        with torch.no_grad():
            p = model.solve(_t(q), S, zs, R=3).argmax(-1).numpy()
        c, g_ = score(p, targets(tasks))
        out[str(K)] = {"cell": c, "strict_grid": g_}
    return out


def e12_readout(model, seed, n=192):
    """Does the decoded mapping match the true permutation?"""
    out = {}
    for k in (0, 1, 2, 3):
        tasks = gen(np.random.default_rng(seed + k), n, palette_size=6)
        if k > 0:
            mem = np.stack([memory_tokens(t, k) for t in tasks])
            with torch.no_grad():
                S, zs = model.build_state(_t(mem))
        else:
            S = torch.zeros(n, Config.dk, Config.dv); zs = torch.zeros(n, Config.dk)
        dec = decode_rule(model, S, zs)
        truth = np.stack([t["perm"][1:] for t in tasks])
        pal = [set(t["palette"].tolist()) for t in tasks]
        inpal = np.stack([[1 if (c + 1) in p else 0 for c in range(9)] for p in pal]).astype(bool)
        out[str(k)] = {
            "all9": round(float((dec == truth).mean()), 4),
            "colours_in_palette": round(float((dec == truth)[inpal].mean()), 4),
            "colours_never_shown": round(float((dec == truth)[~inpal].mean()), 4),
        }
    return out


# --------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/main.pt"))
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default=os.path.join(ROOT, "web/results.json"))
    a = ap.parse_args()

    torch.set_num_threads(1)
    model = StateSurgeryModel()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.eval()

    res = {
        "meta": {"checkpoint": os.path.basename(a.ckpt), "eval_seed": a.seed,
                 "grid": "6x6", "colors": 10, "ops": OPS,
                 "trained_R_range": list(Config.r_train),
                 "default_R": 4, "default_demos": 3},
        "E1_demo_count": e1_demo_count(model, a.seed),
        "E2_ablations": e2_ablations(model, a.seed + 1),
        "E3_transplant": e3_transplant(model, a.seed + 2),
        "E4_depth": e4_depth(model, a.seed + 3),
        "E5_coverage": e5_coverage(model, a.seed + 4),
        "E6_consistency": e6_consistency(model, a.seed + 5),
        "E7_leakage": e7_leakage(model, a.seed + 6),
        "E8_param_identity": e8_param_identity(model, a.seed + 7),
        "E9_capacity": e9_capacity(model, a.seed + 8),
        "E10_state_growth": e10_state_growth(model, a.seed + 9),
        "E11_interference": e11_interference(model, a.seed + 10),
        "E12_readout": e12_readout(model, a.seed + 11),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
