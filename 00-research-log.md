# DataForge 2026 — Pathway Track
## Phase 0 research log: requirements, frontier map, architecture selection, central claim

**Topic (locked):** Skill Acquisition from Demonstrations + Test-Time Adaptation
**Status:** research complete, architecture selected, no code written
**Date:** 8 September 2026

---

# PART 1 — Official PS requirement map

## 1.1 Is our locked topic legal?

Yes, and the combination is explicitly sanctioned. The PS permits combining **two closely related topics** provided the artifact keeps **one central claim and one coherent learning journey**. We are combining:

- **Skill Acquisition from Demonstrations** (Reasoning and generalisation): inferring and applying a new rule from sparse demonstrations, as tested in ARC-style tasks, rather than reusing a training-time capability. The PS instructs: *use the BDH-CQ example.*
- **Test-Time Adaptation** (Memory and learning): framed by the PS itself as **Optimization versus Context**. HRM and TRM take the optimization route on ARC. BDH-CQ is named as the counterexample — no evaluation-task demonstrations in training, no parameter updates at inference, adaptation in recurrent state rather than weights.

These are not two topics stapled together. One is the *behaviour* (a rule is acquired), the other is the *locus* (where the acquisition is written). The second is the mechanism of the first. That is exactly the "closely related" test.

**Constraint this imposes on us:** one central claim, not two. §9 below resolves this.

## 1.2 Judging rubric and what it actually rewards

| Criterion | Points |
|---|---|
| Technical correctness and depth | 25 |
| Technical ownership and live defense | 15 |
| Learning effectiveness | 15 |
| Interactive substrate and honesty | 15 |
| BDH / BDH-CQ integration and evidence discipline | 10 |
| Craft, robustness, accessibility, provenance | 10 |
| One-page concept summary (PDF) | 10 |

**Read the distribution, not the list.** 55 of 100 points go to *technical correctness + ownership + substrate honesty*. Craft and visual polish are worth 10. The rubric is telling us, plainly, that a small real thing you can defend beats a large beautiful thing you cannot. Incorrect claims are singled out for "a major penalty."

Second signal: **10 points for one PDF page of 500–950 words.** That is the highest points-per-hour item in the entire competition and it must be drafted early, not last. The PS even gives its own quality test: hand the summary to a capable AI with no other context and check whether it can restate the claim, mechanism, BDH/BDH-CQ roles, evidence and limitations without introducing errors. We can literally run that test.

## 1.3 Requirement → implementation → evidence matrix

| # | PS requirement | Where it lives in our artifact | Evidence a judge can check |
|---|---|---|---|
| R1 | One precise technical claim | Persistent header line, present on every screen | The claim sentence; every panel traces to it |
| R2 | Learner manipulates a **real concept variable** | Three controls only: Teach/Forget, Transplant state, Demonstration coverage | Each control maps 1:1 to a named variable in the model (`S`, `S_source`, `D`) |
| R3 | Immediate, meaningful consequence | All controls re-run live in-browser inference | Sub-second recompute; no server |
| R4 | Output compared to ground truth | Oracle grid rendered beside model grid, per cell diff | Deterministic oracle ships with the task generator |
| R5 | Learner can explain the concept back | Guided narrative ends with a one-sentence restatement prompt | The 60-second test in §10 |
| R6 | Where the concept appears in BDH / BDH-CQ | Dedicated woven module with eqs (1)–(4), BDH σ lineage, our-toy-vs-paper diff table | Primary-source citations beside each claim |
| R7 | At least one limitation / failure / misconception | The coverage cliff act + the "no weights changed ≠ nothing was learned" misconception panel | Reproducible in-artifact; failure is a scripted act, not an accident |
| R8 | Real substrate, not scripted animation | Hand-written forward pass over exported weights, running in the browser | Weight file + forward-pass source in repo; live parameter-delta readout |
| R9 | Visible state | Fixed-size `S` rendered directly; norm, sparsity, diff-vs-zero | State tensor dumped to console / downloadable |
| R10 | Preset already running on load | Page opens mid-demonstration with a completed task | No Run button anywhere on the first screen |
| R11 | Fast feedback (<1s) | Model sized so a full forward pass is milliseconds | Timing readout displayed |
| R12 | Few controls | Three. Everything else is evidence, not interaction | Control count is auditable |
| R13 | Guide then sandbox | Acts 1–4 guided; sandbox unlocked after | Two explicit modes |
| R14 | No hidden limits | Stated caps: grid size, palette size, rule families, demo count | A "Limits" panel, always reachable |
| R15 | ≥3 primary papers 2022–2026, cited beside claims | Inline citations; full list in README | §2 below |
| R16 | Public artifact URL, no sign-in | Static hosted page | Open in incognito |
| R17 | Public repo + README + setup | Training, export, generation, eval scripts | Reproduction instructions with seeds |
| R18 | Source and license record | Table covering code, data, weights, fonts, graphics | §3.4 flags every reuse |
| R19 | AI assistance disclosure | Explicit README section | — |
| R20 | Evidence labelling (live / precomputed / synthetic / illustrative) | Badge on every panel | §23 of the directive, implemented as a visible legend |

## 1.4 Failure conditions we must not trip

The PS's "weak" list names our exact temptations: generic overviews, paper-summarising chatbots, static decks, **animations passed off as real computation**, unchanged forks, **a bolted-on BDH mention**, unsourced claims, **code the team can't explain**. The "exceptional" bar names three things, and the first is **a reusable substrate**. That is a direct hint: ship a small task generator + tiny model + eval harness that someone else can pick up.

---

# PART 2 — Frontier research map

Organised by the role each source plays for us, not by fame.

## 2.1 The testbed and why it is the right one

- **Chollet 2019, *On the Measure of Intelligence*** (arXiv:1911.01547). ARC is designed to measure *skill-acquisition efficiency* — not whether a system has a skill, but how much prior and experience it needs to get one. This is the definitional backbone of our topic. Pre-2022, so it supports rather than satisfies the recency requirement.
- **Moskvichev, Odouard & Mitchell 2023, ConceptARC** (TMLR). Organises ARC-like tasks by a designed concept ontology, enabling per-concept capability profiles instead of one aggregate score. BDH-CQ uses this as its capability map.
- **Chollet et al. 2024, ARC Prize Technical Report** (arXiv:2412.04604); **Chollet et al. 2025, ARC-AGI-2** (arXiv:2505.11831).

## 2.2 The optimization route (what BDH-CQ is a counterexample *to*)

- **Akyürek, Damani, Qiu, Guo, Kim & Andreas 2024/2025, *The Surprising Effectiveness of Test-Time Training***(arXiv:2411.07279; ICML 2025). Updates parameters temporarily at inference from a loss built out of the task's own demonstrations. Reports up to 6× over the fine-tuned base and 53% on the ARC public validation set with an 8B model. Names three necessary ingredients: initial fine-tuning on similar tasks, auxiliary task format and augmentations, per-instance training. **This is the cleanest published definition of TTT and the correct contrast object.** Second place, ARC Prize 2024.
- **Wang et al. 2025, HRM** (arXiv:2506.21734); **Jolicoeur-Martineau 2025, TRM** (arXiv:2510.04871). Recursive latent solvers with strong ARC/Sudoku/maze numbers.
- **ARC Prize Foundation 2025, *The Hidden Drivers of HRM's Performance on ARC-AGI***. Independent ablation. Findings that matter to us: the *outer refinement loop* and augmentation drive most of the gain rather than the hierarchical architecture; each puzzle gets a learned identity embedding, so **the inference data has to be part of the training dataset**; the system is purely transductive and the induced program stays implicit. This is the load-bearing evidence behind the PS's "optimization route" framing, and it is a *third-party* analysis — evidence-grade above a developer report.
- **Liao & Gu, CompressARC / ARC-AGI Without Pretraining.** The extreme of the optimization route: per-puzzle compression objective, no pretraining at all. Useful as the far end of our adaptation spectrum.

## 2.3 What the state is actually doing (mechanism of ICL)

This cluster is the reason our chosen architecture is scientifically grounded rather than invented.

- **Hendel, Geva & Globerson 2023, *In-Context Learning Creates Task Vectors*** (Findings of EMNLP 2023). Demonstrations compress into a single task vector θ(S) that can be **extracted from hidden states and patched into a different, zero-shot prompt to induce the same task.** This is a published methodology for exactly the intervention our artifact is built around — we are porting it from a Transformer residual stream to a recurrent state.
- **Todd et al. 2024, *Function Vectors in Large Language Models***. Same phenomenon localised via causal mediation over attention heads; robust across prompt format and distribution shift.
- **von Oswald et al. 2023, *Transformers Learn In-Context by Gradient Descent*** (ICML 2023, arXiv:2212.07677). A single linear self-attention layer implements one step of gradient descent on a least-squares regression objective. **Zhang, Frei & Bartlett 2023** (arXiv:2307.03576) prove that construction is the global minimiser of the pretraining loss. This is the sharpest thing in our whole terminology section — see §4.
- **Schlag, Irie & Schmidhuber 2021, *Linear Transformers Are Secretly Fast Weight Programmers*** (ICML 2021). Formal equivalence between linearised attention and fast-weight controllers; additive outer-product updates, and a delta-rule variant that corrects rather than accumulates.
- **Katharopoulos et al. 2020, *Transformers are RNNs*.** **Ba et al. 2016, *Using Fast Weights to Attend to the Recent Past*.**

## 2.4 Latent / recurrent reasoning (the R knob)

- **Hao et al. 2024, Coconut** (arXiv:2412.06769). Feeds the last hidden state back as the next input embedding. Note the honest caveat BDH-CQ itself repeats: the staged internalisation curriculum is essential, and curriculum controls recover much of the logical-reasoning result.
- **Geiping et al. 2025, recurrent depth** (arXiv:2502.05171). Scaling test-time compute by iterating a shared block over a latent state.
- **Saunshi et al. 2025, looped transformers** (arXiv:2502.17416); **Zhu et al. 2026, looped language models** (arXiv:2510.25741). Effective depth without distinct parameters per depth.
- **Zhu, Hao, Hu, Jiao, Russell & Tian 2025, *Reasoning by Superposition*** (NeurIPS 2025). For directed graph reachability, continuous thoughts encode **multiple search frontiers simultaneously and expand them in parallel.** This is the best available theoretical answer to "why not just emit tokens?", and it is a real theorem rather than an appeal to higher dimensionality.
- **Xu & Sato 2026** (ICML 2026). Formal comparison of chain of thought and latent thought: latent iteration exploits parallel structure in computation graphs; token decoding supplies approximate counting and sampling.

## 2.5 BDH family (primary sources)

- **Kosowski, Uznański, Chorowski, Stamirowska & Bartoszkiewicz 2025, *The Dragon Hatchling*** (arXiv:2509.26507). Code: `pathwaycom/bdh` (MIT). Blog: `pathway.com/research/bdh`.
- **Engdahl, Kosowski, Chorowski, Stamirowska, Uznański, Jiang, Phadke, Kinas & Zhong 2026, *BDH-CQ*** (arXiv:2608.09888). Blog: `pathway.com/research/introducing-bdh-cq`, 11 Aug 2026.
- **Pathway Research 2025**, Sudoku Extreme via BDH layers. Note the `pathwaycom/bdh` README's own disclaimer: the 97.4% Sudoku result is from Pathway's **internal** implementation, and the public repo does not reproduce it out of the box. Evidence-grade: developer-reported, not independently reproduced.
- **`pathwaycom/arc-task-gen`** (MIT, official Pathway). Generates fresh ARC-AGI-1-style tasks distribution-matched to the public eval set. Exists precisely because ARC-AGI-1 is public and cannot isolate few-shot rule induction from prior familiarity. **This is an official, MIT-licensed, directly citable asset for us** — see §5.5 for the caveat.

## 2.6 Required primary-source shortlist (2022–2026)

More than the required three, chosen for relevance rather than citation count:
Engdahl et al. 2026 · Kosowski et al. 2025 · Akyürek et al. 2024/2025 · Hendel et al. 2023 · Zhu et al. 2025 · Geiping et al. 2025 · Moskvichev et al. 2023 · ARC Prize Foundation 2025 (technical report, labelled as such).

---

# PART 3 — BDH-CQ technical deconstruction

## 3.1 The published interface

BDH-CQ publishes a *system-level interface*, not an architecture. Demonstrations `D = {(x_t, y_t)}` for t = 1..K, query `x*`.

**Contextual memory (in-context learning):**

```
S_t = U_θ(S_{t-1}, D_t)                     (1)
```

with θ fixed throughout. The paper relates this to attention, fast-weight memory and linear-attention views of contextual association, and names the special case where state accumulates additively per demonstration:

```
S_t = S_{t-1} + U_θ(D_t)
```

Crucially, demonstrations are processed **element-wise and sequentially** rather than compressed into a single task vector. Information available to later inputs depends on associations accumulated from earlier ones — so order can matter, and the state is not order-invariant by construction.

**Recurrent latent reasoning (after all K demonstrations are ingested):**

```
H_0     = E_θ(x*, S_K)                      (2)
H_{r+1} = F_θ(H_r, S_K),   r = 0..R-1       (3)
ŷ       = G_θ(H_R)                          (4)
```

Note that `S_K` conditions **both** the encoder and **every** reasoning iteration — the task memory is not injected once and forgotten. Intermediate `H_r` are never decoded into language.

**The two roles are distinct and this is the crux of the whole topic:** `S` changes as evidence arrives and supports in-context learning. `H` carries the ongoing computation for the current query. Conflating them is the most common misreading and we should teach the distinction explicitly.

## 3.2 What is published vs. withheld

| Published | Withheld |
|---|---|
| Interface eqs (1)–(4), roles of S and H | Dimensions, exact update rules, implementation details |
| Task formulation, data mixture provenance | Complete internal training recipe |
| Benchmark protocol, costs, effort tiers | Candidate construction and ranking internals |
| Behavioural experiments and ablations | Weights / checkpoint (none released) |

The paper states plainly that dimensions, exact update rules and implementation details remain proprietary, and that the complete evaluated system includes input transformations, candidate construction, ranking, and the inference pipeline. **Consequence: the headline 29.5% is a system number, not an architecture number, and no faithful reimplementation of BDH-CQ is possible by anyone.** The PS anticipates this and explicitly does not require us to reproduce it.

## 3.3 The results, with their own stated caveats

| Result | Number | Caveat the paper itself gives |
|---|---|---|
| ARC-AGI-1 public, pass@2 | 118/400 = 29.5% [Wilson 25.24–34.15] | 400 tasks; interval is descriptive |
| Computed cost | $0.00070/task, ~0.85 H200-s @ $3/H200-hr | Computed from measured hardware time; other systems' costs are leaderboard-reported, sometimes API prices |
| ConceptARC semantic, strict task | 95/160 = 59.38% | — |
| ConceptARC pair vs strict gap | 77.92% vs 59.38%, an 18.5-point gap | 52/160 tasks have 1–2 correct test inputs but fail as tasks |
| Contextual binding | 96/96 held-out at rank one; 24/24 at every binding count 2→8 | Fresh colour permutation per task, so this cannot be memorisation |
| Propagation / copying extrapolation | 48/48 each, no ceiling in tested range | 12 outputs per point |
| Ordering | saturated to length 5, then 29/36 → 8/24 → 1/24 at length 8 | Only 3/24 at length 8 even had correct dimensions |
| Nesting | saturated to depth 4, 29/36 at depth 5 | All 36 had correct dimensions; mean cell accuracy >99.9% |
| **Coverage (Table 3)** | Nesting d5: 19/24 → **24/24** with matched support. Ordering L8: 0/24 → **13/24** | Byte-identical test pairs; only context complexity varies |
| Composition (Table 4) | Rotation∘relocation 72/72; reflection∘relocation 47/72; colour swap∘relocation 0/72 | Colour swap only reaches 26/72 *alone*; shuffled families 1/24 alone |
| Effort tiers | LOW 21% / MEDIUM 27% / HIGH 29.5% | Model was *trained* across effort levels |
| MIN vs STANDARD | 111/400 vs 118/400, −1.75 pp, McNemar p = 0.167 | **"Statistically unresolved" — the paper's own words** |
| Identifier/batch replication | 374/480 pairs in both conditions; 96 vs 95 tasks | Rules out the *combined* confound only; does not make ConceptARC fresh, does not rule out training exposure, does not isolate the two interventions factorially |

Two things stand out, and both should shape our project.

**(a) Table 3 is the best experiment in the paper.** Byte-identical held-out inputs; the only variable is whether the demonstrations cover the target complexity. The nesting cliff is therefore largely a *coverage* failure — a failure to extrapolate a demonstrated relation depth — not an execution failure. Ordering benefits from support too but retains an execution bottleneck. This is a clean, causal, one-variable result, and it is the one we should reproduce on our toy.

**(b) The latent-compute story is the paper's softest ground.** Table 5 shows a monotone LOW→MEDIUM→HIGH curve, but the model was trained across effort levels, so the curve partly reflects training-time exposure. Separately, MIN vs STANDARD on the full eval is statistically unresolved by the authors' own test. Anyone who builds a central claim on "more latent compute improves accuracy" is building on the least secure result in the paper. **This is my main disagreement with the directive** — see §8.

## 3.4 BDH ↔ BDH-CQ: the actual mechanistic link

BDH-CQ's `S` is BDH's `σ`. In the BDH paper:

- A fixed ruleset `G` = trained weights; an evolving ruleset `σ` = inference-time state, updated by a Hebbian rule of the form "co-activation of i then j increases σ(i,j)".
- The design choice that makes this work: **state size and parameter count are comparable**, `n ≪ m ≪ n²`. A classical RNN keeps O(n) state; a fast-weight system keeps O(n²). BDH deliberately sits in between, on a sparse graph of m synapses.
- BDH-GPU realises this as **linear attention in a high neuron dimension n** plus a **ReLU-lowrank** feed-forward block `z ↦ (D E z)⁺`, giving positive activations that are empirically ~5% sparse. Reported properties: high Newman modularity, heavy-tailed degree distribution, monosemantic synapses at sub-100M scale.

So the sentence "adaptation lives in recurrent state rather than in weights" is not a slogan in BDH's case — it is a consequence of giving the state parameter-scale capacity and a Hebbian write rule. **That is the BDH module's core teaching point, and it is technically correct.**

Note also the PS's own instruction: do **not** classify BDH as an SSM in the Mamba sense. BDH-GPU is a separate GPU-friendly formulation built from ReLU-lowrank transformations with linear attention. The "isn't this just Mamba?" objection is live in public discussion and we should answer it head-on rather than dodge it.

## 3.5 Faithfulness audit: paper ↔ official description ↔ implementation

**`pathwaycom/bdh` (MIT).** Official, but it is the *BDH* baseline language model, not BDH-CQ. The README itself states the Sudoku result comes from an internal implementation and is not reproduced out of the box. Trustworthy as a reference for BDH-GPU's block structure.

**`lucidrains/bdh-cq` (MIT).** Audited. Findings:

| Aspect | Status |
|---|---|
| Self-described state | Marked **"(wip)"**, ~20 commits |
| Modality | A **language model over token ids** (`BDH(dim, num_tokens)`), plus a `BDHReasoningWrapper` interleaving token chunks with integer latent-reasoning steps |
| ARC pipeline | **Absent.** No grid encoding, no input transformations, no candidate construction, no ranking |
| Extra components | Cites and draws on **Kimi Team 2026, *Attention Residuals*** (arXiv:2603.15031) and **Knupp et al. 2026, *Depth-Recurrent Attention Mixtures*** (arXiv:2601.21582) — neither is part of BDH-CQ |
| Faithfulness ceiling | Bounded by the paper: dimensions and update rules are proprietary, so **no implementation can be faithful**, this one included |

**Verdict:** useful as a second reading of eqs (1)–(4) and as a sanity check on our own interleaving design. **Not** citable as "the BDH-CQ implementation," and we must never imply it reproduces the paper. If we borrow any code we disclose it, name the file, and explain it. MIT permits reuse; it does not confer scientific faithfulness.

**One evidence-grade point most teams will miss.** The paper describes an independent black-box audit that reproduced 29.5% pass@2 — conducted **by co-authors** from Bielik AI and NYU, without weights access, under a documented protocol. The `arc-task-gen` README additionally names Łukasz Kaiser as having evaluated and reproduced the result. The PS asks us to label evidence correctly and warns that a developer-reported result is not an external reproduction. The correct, fair label: *co-author-conducted black-box audit under a documented protocol, weights not accessible to auditors* — stronger than a bare developer claim, weaker than a fully external replication. Stating this precisely, without sneering, is exactly the "evidence discipline" the 10-point BDH criterion is asking for.

---

# PART 4 — Terminology audit

The single highest-leverage table in the project. Judges will probe this.

| Term | What changes | By what process | Optimizer / backward pass? | Persists after the task? |
|---|---|---|---|---|
| Pretraining / fine-tuning | θ, the slow weights | gradient descent over a training set | yes | yes |
| **Test-time training (TTT)** | θ, or a per-instance LoRA ΔW | gradient descent at inference on a loss built from the test input | **yes** | usually discarded per instance |
| **Test-time adaptation (TTA)** | the system's behaviour on the test task | *either* optimization *or* context | either | either |
| **In-context learning (ICL)** | activations / KV cache | a forward pass over demonstrations | no | no — dies with the context |
| **State adaptation** | a **fixed-size** recurrent state S | an architecturally fixed forward-pass update rule | no | no — dies with the session |
| **Recurrent latent reasoning** | a workspace H, over R iterations | repeated application of F_θ | no | no |
| **Skill acquisition from demonstrations** | *(a behavioural claim, not a mechanism)* | — | — | — |

## 4.1 The six rules we hold ourselves to

**1. TTA is a goal; TTT is a method.** BDH-CQ performs TTA via context and state. HRM/TRM/Akyürek perform TTA via optimization. Calling BDH-CQ "test-time training" is simply wrong.

**2. "No weights change" is true, and it needs a qualifier.** von Oswald et al. 2023 show a single linear self-attention layer implements one gradient-descent step on a least-squares objective; Zhang et al. 2023 prove that construction is the global minimiser of the pretraining loss; Schlag et al. 2021 establish that linearised attention *is* fast-weight programming, with additive outer-product writes. An additive state update in a linear-attention system is therefore formally isomorphic to a gradient step on the weights of an **implicit inner model**.

So both of these are wrong:
- ✗ "The weights changed." They did not — bit for bit.
- ✗ "Nothing was learned, an activation just changed." Something with the algebraic form of learning happened.

The defensible sentence: **the trained parameters θ are bit-identical before and after the demonstrations; what changes is a fixed-size state whose capacity is parameter-scale and whose update rule is baked into the architecture. Whether you call that "learning" depends on whether you count the implicit inner model. What is not in dispute is that no optimizer, no loss and no backward pass are involved.**

That paragraph is worth more than any animation, and it is precisely the misconception our artifact should target under R7.

**3. ICL ⊃ state adaptation.** A Transformer does ICL with a KV cache that grows linearly in context — nothing is compressed and nothing is forgotten. A recurrent model does ICL by writing into a fixed-size S — bounded capacity, and interference is possible. The fixed-size constraint is the scientific content, not a footnote.

**4. Define "skill" and "acquire" operationally.**
- *Skill* = a grid→grid mapping specified **only** by the demonstrations and not nameable from the query.
- *Acquire* = after ingesting D, the system applies the mapping to held-out inputs **consistently across every held-out input of the task**.

The strict-task criterion matters. BDH-CQ's own 18.5-point pair-vs-task gap shows a system can produce correct outputs for many inputs while not applying the inferred transformation consistently. One correct output is not evidence of rule acquisition, and our artifact must score the strict way.

**5. Never call it learning because an activation changed. Never call it TTT without an optimizer. Never imply weights moved.**

**6. Within-session memory ≠ continual learning.** The PS warns against this explicitly. Our S dies when the tab closes. Say so.

---

# PART 5 — Research gaps and open questions

**G1. Is latent compute a real dial?** Table 5 is monotone; §6.6's MIN-vs-STANDARD comparison is statistically unresolved (p = 0.167). They measure different knobs, and a careless read collapses them. Open.

**G2. Is the composition asymmetry about composition?** Colour swap fails composed with relocation (0/72) — but it also only reaches 26/72 *alone*, and 1/24 on the shuffled families. The paper says so itself: the composed failures cannot be attributed to composition alone. The real variable may be how inferable an operation is from a given colour layout. Open — and a reason **not** to attempt to "reproduce the asymmetry."

**G3. Is "extrapolation" mostly coverage?** Table 3 shows matched support eliminates the nesting cliff entirely. If one demonstration at the target complexity turns 19/24 into 24/24, then much of what gets reported as a generalisation limit may be a demonstration-design artifact. Large teaching opportunity; largely unexplored.

**G4. Nothing is published about S's internal structure.** No dimension, no update rule, no state visualisation from Pathway. **Any artifact claiming to display BDH-CQ's recurrent state is fabricating.** Ours must be our own toy's state, labelled as such.

**G5. How much of ARC-AGI-1 performance reflects distribution overlap?** The ARC-AGI-1 training set is in the mixture. The identifier/batch replication rules out one confound and Pathway says explicitly it does not make ConceptARC a fresh benchmark. `arc-task-gen` exists because of this.

**G6. What is the *minimum* substrate for state-based skill acquisition?** Nobody has published "the smallest model that demonstrably binds a novel grid transformation into a fixed-size state from three demonstrations." This is a small, real, hackathon-sized gap. **It is our opening.**

---

# PART 6 — Ten candidate architectures

| ID | Concept | Core interaction |
|---|---|---|
| **A1** | **State Surgery** — tiny frozen recurrent model; judge reads, zeroes, freezes and **transplants** S between tasks | Causal intervention on the state |
| A2 | Two Routes — context-adaptation vs a live in-browser TTT loop, side by side | Compare accuracy, latency, ‖Δθ‖ |
| A3 | The Effort Dial — judge controls R latent steps | Slider → accuracy curve |
| A4 | Coverage Cliff — demonstration coverage vs query complexity, query held byte-identical | Reproduces BDH-CQ Table 3 |
| A5 | Interference Lab — bind K mappings into a fixed-size S; watch capacity break | Push past state capacity |
| A6 | Consistency Auditor — all held-out inputs at once; green only if all correct | Reproduces the pair-vs-task gap |
| A7 | Leakage Red Team — judge injects shortcuts (task ids, position, palette) and tries to fool the model | Attack the evaluation |
| A8 | Synapse Scope — render σ as a neuron-neuron graph; watch synapses potentiate | Observation only |
| A9 | Paper Replica Lite — reimplement `lucidrains/bdh-cq`, train on ARC | Technical flex |
| A10 | Human vs Model — both see the demonstrations, both guess | Skill-acquisition efficiency, felt |

---

# PART 7 — Adversarial comparison

Scored 1–5. **Risk is inverted (5 = low risk).**

| | Depth | Educ. | Judge impact | BDH integ. | Sci. credibility | Risk⁻¹ | Repro. | Diff. | **Σ** |
|---|---|---|---|---|---|---|---|---|---|
| **A1 State Surgery** | 5 | 5 | 5 | 5 | 5 | 3 | 4 | 5 | **37** |
| A4 Coverage Cliff | 5 | 5 | 4 | 5 | 5 | 3 | 5 | 4 | **36** |
| A6 Consistency Auditor | 4 | 5 | 4 | 5 | 5 | 5 | 5 | 5 | **38** |
| A2 Two Routes | 4 | 5 | 4 | 4 | 4 | 2 | 3 | 4 | 30 |
| A5 Interference Lab | 4 | 4 | 3 | 4 | 4 | 4 | 5 | 3 | 31 |
| A7 Leakage Red Team | 4 | 4 | 4 | 2 | 5 | 4 | 5 | 5 | 33 |
| A3 Effort Dial | 3 | 4 | 3 | 5 | 3 | 4 | 4 | 2 | 28 |
| A8 Synapse Scope | 3 | 3 | 4 | 5 | 2 | 3 | 3 | 3 | 26 |
| A10 Human vs Model | 2 | 4 | 4 | 2 | 3 | 4 | 4 | 4 | 27 |
| A9 Paper Replica | 4 | 2 | 2 | 3 | 2 | 1 | 3 | 2 | 19 |

**Reject outright:**

- **A9.** Highest apparent impressiveness, lowest defensibility. The paper says dimensions and update rules are proprietary; a "replica" is unfaithful by construction and claiming otherwise is the kind of incorrect claim the rubric penalises heavily. It also depends on a repo its own author marks work-in-progress, which imports components from two papers unrelated to BDH-CQ.
- **A8 as a primary.** "This synapse encodes the rule" without causal evidence is exactly the fake interpretability the directive forbids. Demote to a secondary, explicitly labelled view, and only if we have intervention evidence.
- **A3 as a primary.** It is the slider every other team will build, and the science under it is the paper's weakest (§3.3b). Keep R as a *control variable* in our ablations, not as the headline.
- **A2 as a co-equal.** The fair-comparison burden — matching parameters, compute, demonstration count, training data, evaluation protocol — is real and expensive. Do it as one honest table of published numbers instead of an unfair live race.

---

# PART 8 — The selected architecture

## 8.1 Selection

**Spine: A1 (State Surgery). Acts 2 and 3: A4 (Coverage Cliff) and A6 (Consistency Auditor). Sandbox: A5. Cold open: a 15-second A10 beat. A7 becomes the leakage-audit section of the README plus one in-artifact shortcut toggle. A3 survives as one ablation control.**

## 8.2 Why A1 wins

Every other candidate has the judge **observe**. A1 has the judge **intervene**.

In A3 the judge moves a slider and a number changes — that is a dashboard, and a dashboard is what the PS calls a weak submission. In A1 the judge extracts the state produced by task A's demonstrations, drops it onto task B's query, and watches the model apply **A's rule to B's input** — confidently, and wrongly for B. A correct answer proves the system works. A *predictably wrong* answer, produced by a surgical intervention on one named variable, proves **where the rule lives**. That is causal evidence, and it is the difference between "we visualised the state" and "we showed the state is doing the work."

It also inherits a published methodology rather than inventing one. Extract-and-patch is exactly what Hendel et al. 2023 and Todd et al. 2024 do to Transformer hidden states. We are porting an established interpretability protocol to a recurrent state, in the setting the PS asked for. That is defensible novelty framing: *we investigate*, *we port*, *our toy explores* — never *nobody has done this*.

And it maps onto BDH-CQ's own strongest result. The paper's contextual-binding experiment defines a **fresh colour permutation per task** and solves all 96 held-out outputs at rank one — the model recovers a dense, task-specific mapping introduced entirely through context. That is the same phenomenon our transplant makes tangible.

## 8.3 The non-negotiable design constraint

If our training set contains, say, twelve rule families, then S could be nothing more than a twelve-way task classifier, and "acquisition" reduces to retrieval. A judge will find this in thirty seconds.

**Fix, taken directly from BDH-CQ's binding experiment: every task defines a fresh random colour permutation through its demonstrations.** With ten colours that is log₂(10!) ≈ 21.8 bits of task-specific content that cannot have been memorised, because the specific permutation has never been seen. This converts the claim from *it recognises the rule* to *it binds a novel mapping*, which is the claim we actually want. Rule families supply structure; the permutation supplies novelty. **This constraint drives the task generator design and must not be relaxed for convenience.**

## 8.4 Substrate decision

- Train offline in PyTorch. Target ~1–3M parameters: neuron dimension n ≈ 2048, low-rank d ≈ 64, ReLU-lowrank block, linear attention over a fixed-size associative state, R iterations of a shared latent block. **BDH-family-motivated**, using published BDH properties (sparse positive activations, linear attention in a high neuron dimension, parameter-scale state) — never described as "faithful BDH-CQ," which is impossible.
- Export weights to a compact binary and **hand-write the forward pass in JS/WASM.**
  - No backend, so the central demo cannot die on a timeout — a hard requirement in the directive and a real risk in the PS's "loading behavior" criterion.
  - Sub-second feedback trivially.
  - Static hosting, public URL, no sign-in.
  - Best of all for the 15 ownership points: we can point at the forty lines that implement `S_t = S_{t-1} + φ(k_t)v_tᵀ` and trace the whole system. ONNX Runtime Web would be faster to ship and would hide exactly the thing we need to defend.
- Stated caps, displayed in the artifact: grid size, palette size, rule families, demonstration count, R range, model size. "No hidden limits."

---

# PART 9 — The falsifiable central claim

## 9.1 The claim

> **A recurrent model with frozen weights can acquire an unseen grid-transformation rule purely by updating a fixed-size state as it reads three demonstrations — and that state is the sole carrier of the rule: transplant it and the rule transplants with it; erase it and the rule is gone.**

One sentence. One mechanism. Directly testable by the judge in under a minute. Its boundary — the coverage cliff — is the required limitation and gets its own act rather than a second claim.

## 9.2 What would support it

- Zero demonstrations → at chance on the permutation; three demonstrations → high strict-task accuracy on held-out inputs.
- `‖Δθ‖ = 0` exactly, across the whole interaction.
- Transplanting `S_A` onto task B's query yields **rule A applied to B's input** — not noise, not rule B.
- The effect holds on permutations never present in training.

## 9.3 What would falsify it

| # | Test | Falsifies if |
|---|---|---|
| F1 | Transplant `S_A` into task B's query | Model still applies rule B, or emits noise → S is not the carrier |
| F2 | Zero the state, keep the query | Accuracy stays high → demonstrations were never necessary; the family was memorised |
| F3 | Build S from the *wrong* task's demonstrations | Accuracy stays high → the query alone determines the answer |
| F4 | Freeze S, vary the query | Output does not track the query → the model is ignoring its input |
| F5 | Hash parameter tensors before and after ingestion | Hashes differ → we are doing TTT, not state adaptation |

**F5 is the cheapest high-value thing in this project.** Display the SHA-256 of the weight buffer before and after the demonstrations, side by side, identical, with `‖Δθ‖ = 0.00000000` next to `‖ΔS‖ = 3.42`. A technical judge asks "are the weights really frozen?" and gets an unfakeable answer at a glance. It is truth-beside-estimate applied to the mechanism itself.

## 9.4 Alternative explanations and how each is ruled out

| Alternative | Countermeasure |
|---|---|
| Memorised the task family | Fresh random permutation per task (§8.3); held-out permutations; procedural generation with published seeds |
| Query-only shortcut | F3 |
| Positional or palette shortcut | Randomise positions and palettes per instance; opaque task identifiers, mirroring BDH-CQ §6.5 |
| Transplant "works" because the two tasks are near-identical | Transplant across *different rule families*, and show the output is **confidently wrong for B and correct as an application of A** |
| S is a task *index*, not a bound operator | The permutation carries ~22 bits that no index could encode |
| One correct output counts as acquisition | Strict-task scoring only: every held-out input of a task must be right |

## 9.5 Where I disagree with the directive

1. **Drop latent compute from the central claim.** The directive's draft claim ends "...depends on the amount and structure of subsequent latent computation." That half rests on the paper's least secure evidence (§3.3b). Keep R as an ablation and a sandbox control; do not stake the headline on it.
2. **Lead with sufficiency, not necessity.** §12 proposes "remove the recurrent state update" — a degradation test. Transplanting the state is a *sufficiency* test and is strictly stronger as a demonstration. Run both; lead with the transplant.
3. **Do not reproduce the composition asymmetry (§28).** It is confounded in the source paper by the authors' own admission (§5 G2). Reproducing a confounded result on a different architecture teaches nothing. **Reproduce Table 3 instead** — one variable, byte-identical inputs, clean causal read. Then state openly that we deliberately did not claim the composition result, and why. A scientifically honest refusal is stronger than a muddy reproduction.
4. **"Faithful toy" is not available (§9 of the directive).** Dimensions and update rules are proprietary. We build a BDH-family-motivated toy and say exactly that.
5. **Resolve the §19-vs-§10/§12 tension explicitly.** Demo-count sweeps, R sweeps, complexity sweeps, generalisation and failure characterisation all belong in the repo and one labelled precomputed panel. **Only three controls reach the UI.** Experiments are evidence, not interaction.
6. **Draft the one-page PDF in week one.** 10 points for ~900 words, with a stated test we can run ourselves.

---

# PART 10 — The winning judge experience

## t = 0 — the page is already running

No blank canvas, no Run button, no configuration. On load the judge sees three demonstrations of a fresh-permutation recolour task streaming in one at a time, a state-norm readout climbing as each is ingested, and a held-out query with the **model's grid beside the oracle's grid** — both green. Beneath them, two counters:

```
‖Δθ‖ = 0.00000000        ‖ΔS‖ = 3.42
```

And one line at the top, always present: the claim.

## 0–30 s — Act 1: it works

The judge reads the claim, watches demonstrations arrive, sees truth beside estimate. Nothing has been asked of them yet.

## 30–60 s — Act 2: undo the teaching *(the 60-second test)*

One button: **Forget.** The state resets to zero. The *same* query re-runs. The model fails, visibly, in red. Everything else is unchanged, and the parameter hash is unchanged.

The judge has now personally established that the demonstrations — not the weights — produced the answer. This is the sixty-second test the rubric names, and it is passed by a single click.

## 1–2 min — Act 3: the transplant *(the moment)*

Two tasks side by side. Task A: mirror-and-recolour. Task B: a different rule entirely. Both already solved correctly.

One button: **Move A's state into B.**

The model now applies **A's rule to B's input.** Confidently. Wrongly for B. Correctly as an application of A. The oracle grid beside it is unchanged and red.

Nothing else moved: same weights, same query, same code path, hash identical. Only S changed. **The rule is in the state, and the judge just carried it from one task to another with their own hand.**

## 2–3 min — Act 4: the cliff *(the limitation)*

One slider: the complexity range covered by the demonstrations, with the held-out query held **byte-identical** throughout. The judge watches the model succeed inside the covered range and fail beyond it — and then watches **one added demonstration at the target complexity restore it.**

This is our reproduction of BDH-CQ Table 3 on our own toy, labelled as ours, with the paper's numbers shown beside it under a *Published* badge. The lesson the judge leaves with: what looks like a generalisation limit is often a demonstration-coverage artifact.

## 3 min+ — the BDH module, woven in at the point of maximum curiosity

The judge has just watched a fixed-size state carry a rule. The natural next question is *why would anyone build a model this way?* — which is exactly where the BDH module belongs, not in an appendix.

Contents: eqs (1)–(4) as published, with the distinct roles of S and H made explicit; BDH's σ / fast-weight lineage and the `n ≪ m ≪ n²` state-capacity argument; the ReLU-lowrank and high-dimension linear-attention blocks; an architecture diagram; **our toy's equations beside the paper's with every difference marked**; the published results in a clearly-badged evidence panel; the honest note on why no faithful reimplementation exists; and a direct answer to "isn't this just Mamba?"

## Sandbox

Write your own rule. Choose demonstration count, R, palette, grid size. Inject a shortcut and see whether the model takes it. Break it.

## Always present

An evidence legend on every panel: **Published · Official · Our experiment · Precomputed · Live · Synthetic · Illustrative.**

## Why a judge remembers this after twenty other submissions

Because they did not watch a model learn. **They performed surgery on one.** They removed a rule and the model forgot; they moved a rule between two tasks and the model followed. That is a memory of an action, not of a user interface — and it is not something another team reproduces in a weekend, because it requires a real trained substrate, a task generator built specifically to defeat memorisation, and a state that genuinely carries the task.

---

## Open item

**The submission deadline is not in the materials provided.** It gates scope decisions — chiefly whether Act 4 runs live or ships as a labelled precomputed panel. Everything above is written so that Acts 1–3 are the irreducible core and Act 4 can degrade to precomputed without weakening the central claim.

---

## Next phase (do not start until the above is reviewed)

Phase 1 is not the UI. It is: task generator with the permutation constraint → tiny model → **verify the transplant works at all**. If S does not carry the rule, we discover it in week one for the cost of a training run, not in week three under a half-built interface.
