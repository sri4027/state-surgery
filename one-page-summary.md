# Skill Acquisition from Demonstrations: adaptation that happens in state, not in weights

**The design pressure.** A model meeting an unseen task has two ways to acquire its rule. It can
*optimize*: run gradient steps at inference. On ARC this dominates: Akyürek et al. (2024) fit a per-instance LoRA from the task's own
demonstrations for 53% on public validation, while HRM/TRM augment evaluation demonstrations into
training with a learned per-puzzle identity embedding — which ARC Prize's independent 2025 analysis
found means the inference data must be in the training set. Or it can *condition*: read the demonstrations and let internal state carry the rule. In a Transformer
that carrier is a key–value cache growing with every token. **BDH-CQ conditions into a fixed-size
carrier**, and that substitution is the concept.

**What changes technically.** BDH-CQ (Engdahl et al. 2026) publishes a system interface, not an
architecture. Demonstrations update a recurrent contextual memory, `S_t = U_θ(S_{t-1}, D_t)` with θ
fixed, and the paper names the additive special case `S_t = S_{t-1} + U_θ(D_t)`. The query is then
solved by iterating a separate latent workspace, `H_0 = E_θ(x*,S_K)`, `H_{r+1} = F_θ(H_r,S_K)`,
`ŷ = G_θ(H_R)`, with intermediates never decoded into language. The two states have different jobs:
`S` accumulates evidence, `H` computes the current answer. The lineage is Dragon Hatchling (Kosowski et al. 2025): fixed weights plus an evolving synaptic ruleset
σ sized `n ≪ m ≪ n²`, so the state has capacity to hold a rule. **Trade-off:** it cannot grow, so
coverage and interference bind where a KV cache would simply store more.

**Terminology.** Test-time *adaptation* is a goal; test-time *training* is one method. BDH-CQ adapts
with no optimizer, loss or backward pass. But "nothing was
learned, an activation changed" is equally wrong: linearised attention *is* fast-weight programming
(Schlag et al. 2021), and one linear self-attention layer implements one gradient-descent step on a
least-squares objective (von Oswald et al. 2023). The defensible statement is that **the trained
parameters are bit-identical before and after the demonstrations, and what changes is a fixed-size
state whose update rule is architectural.**

**Our contribution.** A 298,720-parameter recurrent model, frozen weights, 6×6 grids, where every task
invents a fresh colour permutation (9! = 362,880, so the rule was never trained on — BDH-CQ's own
binding probe, §6.3). Three live browser interventions. **Erase the state:** strict-grid accuracy goes
99.6% → 0%, with the SHA-256 of the parameter buffer recomputed live and unchanged, `||Δθ|| = 0`.
**Transplant it:** build `S` from task A's demonstrations and run it on task B's query with matched
palettes. Over 256 pairs with zero rule collisions the output scores 0.0% against rule B and **100%
against rule A applied to B's input** — a *predictably wrong* answer produced by moving one named
variable, which shows where the rule lives. **Starve the coverage:** holding the query byte-identical,
cells whose colour the demonstrations contain score 100% while cells whose colour they omit score
**0.0%**, below the 11.1% of guessing, so the model is confidently applying a binding that does not
exist. One supporting demonstration restores them to 99.4%. This is the miniature of BDH-CQ's cleanest
experiment, Table 3, where matched support takes depth-5 nesting from 19/24 to 24/24: **what reads as a
generalisation limit is often a demonstration-coverage artifact.**

**Read the rule out.** One probe grid containing every colour once decodes what the state has bound:
with three demonstrations the decoded mapping matches the true permutation on
99.7% of demonstrated colours and 0.0% of
colours never shown, against a chance rate of 11.1%. Superposing K rules into the one fixed-size state
drops strict-grid accuracy from 99.5% to 3.1% at K=6 —
the interference cost a growing key–value cache does not pay.

**Comparison.**

| | HRM / TRM | Test-time training (Akyürek et al.) | BDH-CQ |
|---|---|---|---|
| adapts by | optimization + puzzle embeddings | per-instance LoRA gradient steps | context written to recurrent state |
| params at inference | updated | updated | **frozen** |
| unseen task needs | a backward pass | a backward pass | a forward pass |
| reported ARC cost | $1.48 / $1.76 per task | high (8B + per-task training) | $0.00070 per task, 29.5% pass@2 |

**Evidence discipline.** BDH-CQ's 29.5% is a *system* result: the evaluated system includes input
transformations, candidate construction and ranking. The independent black-box audit that reproduced it
was conducted by **co-authors** from Bielik AI and NYU without weights access under a documented
protocol — stronger than a developer claim, weaker than a fully external replication. Dimensions and
update rules are proprietary and no checkpoint is public, so **no faithful reimplementation exists,
ours included**; our model is BDH-family-motivated and shares no weights or code.

**Open question, and our own negative results.** BDH-CQ's latent-compute evidence points two ways:
LOW/MEDIUM/HIGH gives 21/27/29.5%, but the model was *trained* across effort levels, and the separate
MIN-versus-STANDARD comparison is, in the authors' words, statistically unresolved (McNemar p = 0.167).
In our toy it is flat across R = 1…8, because the task needs one associative lookup. Our
pair-versus-strict-task gap is 0.3 points where BDH-CQ's is 18.5 on ConceptARC: the full system often
produces correct outputs without applying the rule to every input of a task, which under a
rule-induction account is the sharper limitation. Structural operations did not train at our scale; we
report that rather than dropping it.

**Continue learning.** BDH-CQ arXiv:2608.09888 §3, §6.2–6.3; Dragon Hatchling arXiv:2509.26507 §4–6;
ARC Prize 2025 HRM analysis; Hendel et al. 2023 (the extract-and-patch method our transplant ports to
a recurrent state).

*Live artifact: https://claude.ai/code/artifact/51fdc852-9592-4b6c-b2c1-9f107a19b02f — live in-browser, ~90 seconds.*
