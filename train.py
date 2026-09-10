"""
Train the toy state-adaptation model.

Training distribution is deliberately narrow and fully COVERED: every colour
that appears in a query also appears in some demonstration input. The
coverage-cliff experiment (Act 4) then probes a condition that was never
trained on, so the failure is a genuine prediction of the mechanism rather
than a trained-in artifact.

Reasoning depth R is SAMPLED during training over cfg.r_train. This is the
same choice BDH-CQ made when training across effort levels, and it is exactly
why an accuracy-vs-R curve must not be read as pure inference-time scaling.
We report the curve and say so.

Run:  python -m arcstate.train --steps 12000
"""

import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .tasks import sample_batch, N_COLORS, make_task, batch_from_tasks
from .model import StateSurgeryModel, Config, n_params

OPSEL = None
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def evaluate(model, rng, n_tasks=128, n_demos=3, R=3, bs=32, ops=None):
    model.eval()
    cell_ok = tot = 0
    grid_ok = grids = 0
    with torch.no_grad():
        for i in range(0, n_tasks, bs):
            b = min(bs, n_tasks - i)
            _, (mem, q, y) = sample_batch(rng, b, n_demos=n_demos, ops=ops)
            lg = model(torch.from_numpy(mem), torch.from_numpy(q), R=R)
            pred = lg.argmax(-1).numpy()
            cell_ok += (pred == y).sum(); tot += y.size
            grid_ok += (pred == y).all(axis=1).sum(); grids += b
    model.train()
    return cell_ok / tot, grid_ok / grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ops", type=str, default="identity")
    a = ap.parse_args()

    global OPSEL
    OPSEL = None if a.ops == "all" else a.ops.split(",")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(a.seed)
    eval_rng_seed = 99991

    cfg = Config()
    model = StateSurgeryModel(cfg)
    if a.resume and os.path.exists(a.resume):
        model.load_state_dict(torch.load(a.resume, map_location="cpu"))
        print("resumed from", a.resume, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.15, div_factor=10,
        final_div_factor=20)

    print(json.dumps({"params": n_params(model), "state": cfg.dk * cfg.dv,
                      "steps": a.steps, "bs": a.bs, "seed": a.seed}), flush=True)

    t0 = time.time()
    best = -1.0
    log = []
    for step in range(1, a.steps + 1):
        R = int(rng.integers(cfg.r_train[0], cfg.r_train[1] + 1))
        nd = int(rng.integers(1, 5))   # 1..4 demos, so all demo slots are trained
        _, (mem, q, y) = sample_batch(rng, a.bs, n_demos=nd, ops=OPSEL)
        lg = model(torch.from_numpy(mem), torch.from_numpy(q), R=R)
        loss = F.cross_entropy(lg.reshape(-1, N_COLORS),
                               torch.from_numpy(y).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 250 == 0 or step == 20:
            c, g = evaluate(model, np.random.default_rng(eval_rng_seed), 96, ops=OPSEL)
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss.item(), 4),
                   "cell": round(float(c), 4), "grid": round(float(g), 4),
                   "sec": round(el, 1), "s_per_step": round(el / step, 3)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if g >= best:
                best = g
                torch.save(model.state_dict(), f"{OUT}/{a.tag}.pt")
            with open(f"{OUT}/{a.tag}_log.json", "w") as f:
                json.dump(log, f, indent=1)

    torch.save(model.state_dict(), f"{OUT}/{a.tag}_final.pt")
    print("done best_grid", best, flush=True)


if __name__ == "__main__":
    main()
