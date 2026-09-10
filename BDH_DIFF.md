# BDH / BDH-CQ: what we took, what we changed, what we must not claim

## Source discipline
Primary sources only: Kosowski et al. 2025 (arXiv:2509.26507) and Engdahl et al. 2026
(arXiv:2608.09888), plus Pathway's own research pages. No secondhand summaries.

## Taken from the published description
| Property | Source | In our toy |
|---|---|---|
| Sparse positive activations | BDH Sec. 4.1 (~5% active) | ReLU throughout; keys `relu(Wh+b)^2` |
| ReLU-lowrank block `z -> relu(D(Ez))` | BDH Sec. 5.2 | `ReluLowRank`, d=64 -> n=256 -> d |
| Linear attention as fixed-size associative state | BDH Sec. 6.1 | 128x128 state, normalised read |
| Additive write `S_t = S_{t-1} + U(D_t)` | BDH-CQ Eq. (1), named special case | `S = einsum('btk,btv->bkv')` |
| Iterated latent workspace, undecoded | BDH-CQ Eqs. (2)-(4) | `Reasoner.step` x R |
| Frozen weights at inference | BDH-CQ Sec. 3.2 | SHA-256 verified |
| Fresh-permutation binding probe | BDH-CQ Sec. 6.3 | our entire task family |
| Coverage / matched-support design | BDH-CQ Table 3, App. A.3 | Act 3 |
| Strict-task vs pair scoring | BDH-CQ Sec. 6.1, 6.4 | `e6_consistency` |

## Deviations (all disclosed in the artifact)
| | BDH-CQ | Ours | Why |
|---|---|---|---|
| Scale | 150M, full ARC | 298,720, 6x6 | must run in a browser and be visible |
| Update rule | `U_θ` order-dependent | additive sum, exactly order-invariant | the named special case is what we could implement |
| State | proprietary dims | dense 128x128 (~5% of params) | BDH targets parameter-scale `n<<m<<n^2`; we do not |
| `H_0` | `E_θ(x*, S_K)` | embedding only; state enters at every F step | simpler; disclosed |
| Embeddings | not published | **factored** blocks, not summed | summed embeddings made the mechanism impossible |
| Pipeline | input transforms, candidates, ranking | none | out of scope |
| Rule family | full ARC | colour binding only | structural ops did not train at this scale |
| Latent compute | effort tiers | R has no effect | reported as a negative result |

## What we must never claim
1. That this is BDH-CQ, or reproduces it. Dimensions and update rules are proprietary; no checkpoint
   is public; the evaluated system includes ranking we do not have.
2. That `lucidrains/bdh-cq` is a faithful implementation. It is WIP, is a language model with no ARC
   pipeline, and imports components from papers unrelated to BDH-CQ.
3. That BDH is an SSM in the Mamba sense. The PS explicitly warns against this; BDH-GPU is a separate
   formulation built from ReLU-lowrank transformations with linear attention.
4. That the 29.5% is an architecture result. It is a system result.
5. That the reproduction audit was fully external. It was conducted by co-authors, without weights
   access, under a documented protocol. Stronger than a developer claim; weaker than an external
   replication. Say exactly that.
6. That our order-invariance says anything about BDH-CQ. It is a difference between the systems.
