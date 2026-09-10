# Technical defence + judge demo scripts

## A. 60-second script

> "Every task here invents a colour rule the model has never seen — there are 362,880 and this one was
> drawn a second ago. Three examples, then a held-out grid: solved, ground truth beside it.
> Now watch. **[press Forget]** Same weights — the parameter hash is unchanged, ||Δθ|| is exactly zero —
> same query, same code path. The only thing I removed was a fixed-size state, and the rule went with it.
> That's the claim: the rule doesn't live in the weights, it lives in the state."

## B. 2-minute script

Add Act 2. "Two tasks, same colours, different rules. Both solved. Now I move A's state into B.
**Predict what you'll see.** — It applies **A's rule to B's input**: wrong for B, exactly right as an
application of A, 100.0% strict-grid over 256 pairs with zero rule
collisions. A correct answer would only show the system works. A *predictably wrong* answer produced by
moving one named variable shows where the rule lives."

Then Act 3. "The query is byte-identical in both conditions. One colour never appears in the
demonstrations. Covered cells 100.0%, uncovered cells
0.0% — *below* the 11.1% of guessing, so it isn't
guessing, it's confidently applying a binding that doesn't exist. One supporting demonstration:
99.4%. What looked like a reasoning limit was demonstration coverage.
That's BDH-CQ's Table 3 in miniature — matched support takes their depth-5 nesting from 19/24 to 24/24."

## C. 5-minute technical version

Add: the interface equations and the S/H distinction; BDH's σ and the `n ≪ m ≪ n²` capacity argument;
the diff table (what we took, what we changed); the terminology table; the honest negatives
(R does nothing; structural ops didn't train; our write is order-invariant and BDH-CQ's isn't);
the evidence-grade discussion of the co-author audit; then open the repo.

---

## D. Anticipated questions

**"Are the weights actually frozen?"**
Yes. `experiments.py::e8_param_identity` hashes the parameter buffer before and after ingesting
demonstrations: SHA-256 identical, `||Δθ|| = 0.0` exactly. No optimizer is constructed at inference and
no `.backward()` is reachable from the inference path. The artifact displays this.

**"Isn't this just in-context learning?"**
It is a *kind* of in-context learning — that is the point, and BDH-CQ's own title says so. The
distinction that matters is the carrier. A Transformer's carrier is a KV cache that grows linearly with
context and compresses nothing. Ours is a fixed 128×128 matrix. That is why coverage and interference
are live constraints here and why the transplant is even possible: you cannot meaningfully "move the
KV cache of task A onto task B's query," but you can move a fixed-size state.

**"Why isn't this test-time training?"**
Because there is no optimizer, loss or backward pass. TTT (Akyürek et al.) updates parameters — a
per-instance LoRA fitted from the task's own demonstrations. HRM/TRM go further: ARC Prize's 2025
analysis found each puzzle gets a learned identity embedding, so the inference data has to be in the
training set. Ours needs a forward pass. TTA is the goal; TTT is one method.

**"So nothing was learned — an activation changed."**
That is the other half of the misconception and it is also wrong. Linearised attention *is* fast-weight
programming (Schlag et al. 2021) and one linear self-attention layer implements one gradient-descent
step on a least-squares objective (von Oswald et al. 2023). Our write is literally
`S = Σ_t k_t ⊗ v_t` — an outer-product fast-weight update. Something with the algebraic form of
learning happened, on the weights of an *implicit inner model*. What is not in dispute is that no
gradient step was taken on θ.

**"How do you know it isn't memorising the task family?"**
Each task draws a fresh permutation from 9! = 362,880. Success requires ~18.5 bits of task-specific
content that cannot be in the weights. This is BDH-CQ's own binding probe (§6.3), and it is the design
constraint the whole project was built around. Supporting controls: another task's state scores
0.0%; a random scale-matched state 0.0%;
zero state 0.0%.

**"Couldn't the answer come from the query alone?"**
No — the permutation is not recoverable from the query. `e2_ablations` supplies another task's state
with the same query: 0.0% strict grid.

**"Is the transplant just two similar tasks?"**
Palettes are matched deliberately so every colour in B's query is bound by A's state, isolating the
rule from coverage. Rule collisions are measured and were 0.0% over 256 pairs.
Without palette matching the transplant drops to ~5.9% strict grid — and that is itself the coverage
law, not a failure of the mechanism. We found this, fixed the design, and report both.

**"What is live and what is precomputed?"**
Acts 1–3 and the sandbox are live in your browser: `web/engine.js`, a hand-written float32 forward pass,
~100 ms. Aggregate tables are precomputed by `experiments.py` at seed 20260908 and labelled. BDH-CQ
numbers are published and labelled. There are no scripted animations anywhere. The engine self-verifies
against PyTorch at load using an exported fixture (max logit Δ 5.5e-6) and the header says so if it fails.

**"Did you reproduce BDH-CQ?"**
No, and nobody can. Dimensions, update rules and implementation details are proprietary, no checkpoint
is public, and the evaluated system includes input transformations, candidate construction and ranking.
Ours is BDH-family-motivated: sparse positive activations, ReLU-lowrank blocks, linear attention as a
fixed-size associative state, and the additive write the paper names in Eq. (1). The diff table lists
every deviation.

**"Is `lucidrains/bdh-cq` your basis?"**
No. We read it and did not use it. It is marked work-in-progress, is a language model over token ids
with no ARC pipeline, and imports components from two 2026 papers (Kimi's *Attention Residuals*,
*Depth-Recurrent Attention Mixtures*) that are not part of BDH-CQ. It cannot be faithful for the reason
above. Citing it as "the BDH-CQ implementation" would be an incorrect claim.

**"Your R slider does nothing."**
Correct, and we say so in the artifact. Our task needs one associative lookup. We report it as a
negative result rather than hiding the control. It is also why we refused to build the central claim on
inference-time compute: BDH-CQ's own effort evidence is mixed — LOW/MED/HIGH gives 21/27/29.5% but the
model was *trained* across effort levels, and their MIN-vs-STANDARD comparison is, in their words,
statistically unresolved (McNemar p = 0.167).

**"Why is your consistency gap 0.26 points when theirs is 18.5?"**
Because our task is far simpler and our model saturates it. We report the contrast rather than the
flattering number: under a rule-induction account a correctly induced rule should transfer to *every*
test input, so their 18.5-point gap is the sharper limitation and we say so.

**"Why didn't you reproduce the composition asymmetry?"**
Because it is confounded in the source paper by the authors' own admission: colour swap fails composed
with relocation (0/72) but reaches only 26/72 *alone*, and 1/24 on the shuffled families. Reproducing a
confounded result on a different architecture teaches nothing. We reproduced Table 3 instead — one
variable, byte-identical inputs, clean causal read — and state openly that we declined the other.

**"What broke while you built this?"**
Three things, all in the README. Additive colour+position embeddings made colour-specific keys
impossible, so every associative read returned the mean of all values and accuracy pinned at the
background-only baseline; factored coordinate blocks fixed it (0% → 100%). That is BDH's own state-vs-
distinction capacity problem, hit empirically at 10⁵ parameters. And the joint structural-op family
never trained at this scale, which is why the rule family is colour binding.

**"Could another team build this in a weekend?"**
The UI, yes. The substrate, less easily: it needs a task generator designed to defeat memorisation, a
state that genuinely carries the rule (which took three architectural fixes to achieve), and a
browser forward pass that verifies against the reference implementation. The transplant only works
because the demonstrations have no other channel to the query.

**"Your state visualisation — is it real, or decoration?"**
The heatmap is the actual 128x128 float32 state, drawn pixel per entry. The *rule readout* is stronger:
it runs the model's own read-and-decode path on a probe grid containing every colour once, so what you
see is what the state causes the model to emit. With 3 demonstrations it matches the true permutation
on 99.7% of demonstrated colours and 0.0%
of never-shown ones. We deliberately do **not** claim to know how the binding is encoded inside the
16,384 numbers — that would be the fake-interpretability trap.

**"Show me the fixed-size state actually costing you something."**
Act 4. Because our write is additive, K tasks superpose into one state exactly:
`S = S_1 + ... + S_K`. Strict grid falls 99.5% -> 35.9%
-> 19.3% -> 3.1% for K = 1, 2, 3, 6, and the decoded
rule degrades gradually rather than vanishing. A growing KV cache would simply store more. This is the
trade-off the concept actually makes, demonstrated rather than asserted.

**"The weight hash — is that a stored string?"**
No. It is recomputed in your browser with SubtleCrypto over the live 298,720-float parameter buffer,
before and after ingestion, on every click, with an FNV-1a fallback for non-secure contexts. An earlier
build of this artifact displayed a hardcoded zero; we caught it in review and replaced it, because a
static string claiming a live measurement is exactly the failure mode this track penalises.
