"""Export the trained checkpoint to a single JSON the browser can run.

Also writes a PARITY FIXTURE: a fixed task with the PyTorch logits, so the
hand-written JavaScript forward pass can verify itself against the reference
implementation at page load. If parity fails the artifact says so out loud.
"""
import base64, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcstate.model import StateSurgeryModel, Config
from arcstate.tasks import make_task, memory_tokens, query_tokens, N_COLORS, H, W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ck = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs/main.pt")

m = StateSurgeryModel(); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()

sd = m.state_dict()
names = sorted(sd.keys())
meta, buf, off = {}, [], 0
for nme in names:
    a = sd[nme].detach().numpy().astype(np.float32).ravel()
    meta[nme] = {"shape": list(sd[nme].shape), "off": off, "n": int(a.size)}
    buf.append(a); off += a.size
flat = np.concatenate(buf)

cfg = Config()
model_json = {
    "config": {"d": cfg.d, "n": cfg.n, "dk": cfg.dk, "dv": cfg.dv,
               "heads": cfg.heads, "H": H, "W": W, "NC": N_COLORS,
               "c_slice": cfg.c_slice, "o_slice": cfg.o_slice,
               "p_slice": cfg.p_slice, "m_slice": cfg.m_slice,
               "r_train": list(cfg.r_train), "default_R": 3},
    "n_params": int(flat.size),
    "meta": meta,
    "weights_b64": base64.b64encode(flat.tobytes()).decode("ascii"),
}

# ---- parity fixture ---------------------------------------------------------
rng = np.random.default_rng(4242)
t = make_task(rng, ops=["identity"], palette_size=4)
mem = memory_tokens(t, 3)[None]
q = query_tokens(t["queries"][0][0])[None]
with torch.no_grad():
    lg = m(torch.from_numpy(mem), torch.from_numpy(q), R=3)[0].numpy()
model_json["parity"] = {
    "mem": mem[0].tolist(), "query": q[0].tolist(),
    "target": t["queries"][0][1].reshape(-1).tolist(),
    "logits_sample": [round(float(x), 5) for x in lg[:4].ravel()],
    "argmax": lg.argmax(-1).tolist(),
}

out = os.path.join(ROOT, "web/model.json")
with open(out, "w") as f:
    json.dump(model_json, f)
print("wrote", out, round(os.path.getsize(out) / 1e6, 2), "MB;",
      flat.size, "params;", len(names), "tensors")
for nme in names:
    print("  ", nme, meta[nme]["shape"])
