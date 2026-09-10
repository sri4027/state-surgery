"""
A toy, BDH-family-motivated recurrent model for demonstration-conditioned
colour binding.

THIS IS NOT BDH-CQ. BDH-CQ's dimensions, exact update rules and implementation
details are proprietary (Engdahl et al. 2026, Sec. 3.3), so no faithful
reimplementation is possible by anyone, including us. See docs/BDH_DIFF.md.

Taken from the published BDH / BDH-CQ description:
  * high-dimensional, positive, ReLU-sparse activations             (BDH 4.1)
  * a "ReLU-lowrank" block  z -> relu(D (E z)) in neuron space      (BDH 5.2)
  * linear attention as a fixed-size associative state              (BDH 6.1)
  * demonstrations accumulate ADDITIVELY:  S_t = S_{t-1} + U(D_t),
    the special case explicitly named in BDH-CQ Eq. (1)
  * a reasoning workspace iterated R times under the task state,
    BDH-CQ Eqs. (2)-(4); intermediates are never decoded to symbols

FACTORED EMBEDDINGS. Colour, position and demonstration index occupy DISJOINT
coordinate blocks rather than being summed. With summed embeddings a linear
key projection cannot isolate colour from positional noise of equal magnitude,
and the normalised read degenerates to the mean of all values -- BDH's own
"state capacity vs. distinction capacity" problem (BDH Sec. 6.1). This is a
documented deviation from BDH, made for legibility at 10^5 parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import N_COLORS, H, W, L, MAX_DEMOS

EPS = 1e-5


class Config:
    d = 64
    n = 256
    dk = 128
    dv = 128
    heads = 2
    r_train = (1, 5)
    # factored coordinate blocks inside R^d
    c_slice = (0, 24)      # colour (input colour for demos, own colour for query)
    o_slice = (24, 48)     # output colour (demonstrations only)
    p_slice = (48, 60)     # position
    m_slice = (60, 64)     # demonstration index


def _lin(i, o, std=None):
    m = nn.Linear(i, o, bias=False)
    nn.init.normal_(m.weight, std=std if std else (1.0 / i) ** 0.5)
    return m


def read(state, z, q):
    """Normalised linear-attention read:  (q . S) / (q . z)."""
    num = torch.einsum("blk,bkv->blv", q, state)
    den = torch.einsum("blk,bk->bl", q, z).unsqueeze(-1)
    return num / (den + EPS)


class SharpKey(nn.Module):
    """Positive, sparse, contrast-enhanced keys:  k = relu(Wh + b)^2."""

    def __init__(self, din, dout, b0=-0.5):
        super().__init__()
        self.w = _lin(din, dout)
        self.b = nn.Parameter(torch.full((dout,), float(b0)))

    def forward(self, h):
        k = F.relu(self.w(h) + self.b)
        return k * k


def _emb(num, dim, std=1.0):
    e = nn.Embedding(num, dim)
    nn.init.normal_(e.weight, std=std)
    return e


class MemEmbed(nn.Module):
    """Demonstration cell -> R^d, factored: (colour_in, colour_out, pos, demo)."""

    def __init__(self, cfg):
        super().__init__()
        c, o, p, m = cfg.c_slice, cfg.o_slice, cfg.p_slice, cfg.m_slice
        self.cfg = cfg
        self.cin = _emb(N_COLORS, c[1] - c[0])
        self.cout = _emb(N_COLORS, o[1] - o[0])
        self.pos = _emb(H * W, p[1] - p[0], std=0.5)
        self.demo = _emb(MAX_DEMOS, m[1] - m[0], std=0.5)

    def forward(self, tok):
        pos = tok[..., 2] * W + tok[..., 3]
        return torch.cat([self.cin(tok[..., 0]), self.cout(tok[..., 1]),
                          self.pos(pos), self.demo(tok[..., 4])], dim=-1)


class QueryEmbed(nn.Module):
    """Query cell -> R^d, colour in the SAME block the memory keys on."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c, o, p, m = cfg.c_slice, cfg.o_slice, cfg.p_slice, cfg.m_slice
        self.col = _emb(N_COLORS, c[1] - c[0])
        self.pos = _emb(H * W, p[1] - p[0], std=0.5)
        self.zo = o[1] - o[0]
        self.zm = m[1] - m[0]

    def forward(self, tok):
        pos = tok[..., 2] * W + tok[..., 3]
        c = self.col(tok[..., 0])
        p = self.pos(pos)
        z1 = c.new_zeros(*c.shape[:-1], self.zo)
        z2 = c.new_zeros(*c.shape[:-1], self.zm)
        return torch.cat([c, z1, p, z2], dim=-1)


class ReluLowRank(nn.Module):
    """BDH's ReLU-lowrank block in neuron space R^n (positive, sparse)."""

    def __init__(self, cfg):
        super().__init__()
        self.up = _lin(cfg.d, cfg.n)
        self.e = _lin(cfg.n, cfg.d)
        self.dd = _lin(cfg.d, cfg.n)
        self.dn = _lin(cfg.n, cfg.d)

    def forward(self, v, ret_act=False):
        x = F.relu(self.up(v))
        y = F.relu(self.dd(self.e(x)))
        out = self.dn(x + y)
        return (out, x) if ret_act else out


class Writer(nn.Module):
    """U_theta: encode each demonstration element, then write additively."""

    def __init__(self, cfg, embed):
        super().__init__()
        self.emb = embed
        self.blk = ReluLowRank(cfg)
        self.wk = SharpKey(cfg.d, cfg.dk)
        self.wv = _lin(cfg.d, cfg.dv)

    def forward(self, tok, per_token=False):
        h = self.emb(tok)
        h = h + self.blk(h)                        # no norm: factoring survives
        k = self.wk(h)                             # positive sparse keys
        val = self.wv(h)
        S = torch.einsum("btk,btv->bkv", k, val)   # S_t = S_{t-1} + k v^T
        z = k.sum(dim=1)
        return (S, z, k, val) if per_token else (S, z)


class Reasoner(nn.Module):
    """BDH-CQ Eqs. (2)-(4): iterate the workspace under the task state."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hd = cfg.heads
        self.hd = hd
        self.qs = SharpKey(cfg.d, cfg.dk * hd)
        self.qq = SharpKey(cfg.d, cfg.dk * hd)
        self.os = _lin(cfg.dv * hd, cfg.d)
        self.oq = _lin(cfg.dv * hd, cfg.d)
        self.blk = ReluLowRank(cfg)
        self.norm = nn.LayerNorm(cfg.d)

    def _multi(self, proj, Hs, state, z):
        B, Ln, _ = Hs.shape
        q = proj(Hs).view(B, Ln * self.hd, self.cfg.dk)
        return read(state, z, q).reshape(B, Ln, self.hd * self.cfg.dv)

    def step(self, Hs, S, zs, Qm, zq, ret_act=False):
        a = self._multi(self.qs, Hs, S, zs)
        b = self._multi(self.qq, Hs, Qm, zq)
        u = Hs + self.os(a) + self.oq(b)
        f, act = self.blk(u, ret_act=True)
        h = self.norm(u + f)
        return (h, act) if ret_act else h


class StateSurgeryModel(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        self.cfg = cfg
        self.task_writer = Writer(cfg, MemEmbed(cfg))    # demonstrations -> S
        self.query_writer = Writer(cfg, QueryEmbed(cfg))  # query grid     -> Qm
        self.q_emb = QueryEmbed(cfg)
        self.reason = Reasoner(cfg)
        self.out = _lin(cfg.d, N_COLORS)

    def build_state(self, mem_tok):
        if mem_tok.shape[1] == 0:
            B = mem_tok.shape[0]
            dt = self.out.weight.dtype
            return (torch.zeros(B, self.cfg.dk, self.cfg.dv, dtype=dt),
                    torch.zeros(B, self.cfg.dk, dtype=dt))
        return self.task_writer(mem_tok)

    def solve(self, q_tok, S, zs, R=3, trace=False):
        Hs = self.q_emb(q_tok)
        Qm, zq = self.query_writer(q_tok)
        traj = []
        for _ in range(R):
            Hs = self.reason.step(Hs, S, zs, Qm, zq)
            if trace:
                traj.append(self.out(Hs))
        logits = self.out(Hs)
        return (logits, traj) if trace else logits

    def forward(self, mem_tok, q_tok, R=3):
        S, zs = self.build_state(mem_tok)
        return self.solve(q_tok, S, zs, R=R)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
