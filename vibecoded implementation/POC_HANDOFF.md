# Graph-injected facts in TMT: proof of concept

**The idea:** the model works out *in its own latent space* what it's being asked about, a graph
walk "reads up" on that topic in an external fact graph, and the retrieved facts are injected
back into the model to shape its answer. No text search and no outside LLM at run time, and
editing the graph changes the answer without retraining.

**Status (2026-09-20):** each step works on its own, and the full chain measurably steers the
model's answer toward whatever the graph says. It does **not** yet make the model *say* the right
answer: this 10M-parameter checkpoint can't hold a question in memory long enough to answer
anything (details below).

## Results

All numbers come from `inject_test.py` / `test_graph_edit.py` on `real_graph.json`, using the frozen
checkpoint `checkpoints/base-10m-tracefix_snap2.safetensors`. "Fresh" means question sets that
weren't used to choose any setting.

| Step | Result |
|---|---|
| 1. Latent → find the topic node | 82/82 concepts land on their own node |
| 2. Graph walk → pick facts | 82/82 get their own facts in the top 3; 93% of the retrieved top-3 facts belong to the concept |
| 3. Edit the graph → retrieval follows | 8/8 edits (e.g. "ocean is salty" → "ocean is muddy") retrieved, old fact gone, no retraining |
| 4a. Inject a fact → answer leans toward it | 80%, 93%, 82%, 82%, 84% on five fresh question sets of 56 (50% = no effect); model barely disturbed |
| 4b. **Full chain + graph edit** → answer follows the edit | **91% (84/92)**: 32/34, 24/29, 28/29 on three fresh question sets with the default settings |
| Graph off → identical to plain TMT | `test_invisible.py` PASS (bit-identical) |

"Answer leans toward the fact" is measured per question. The model scores every candidate answer
(e.g. all 20 locations for "fish is found in ___"). We then compare how much it prefers the true
object ("ocean") over a false one ("desert") in two runs that differ **only** in which fact sits in
memory: the graph's version or an edited one. If injection did nothing, the answer would follow the
injected fact 50% of the time.

## How injection works (the part to look at)

TMT's per-layer memory is `state = decay*state + embed(byte) + dummy`. That's purely additive,
so the memory left after reading `"fish is found in ocean."` is a fixed vector per layer that we
precompute offline for every fact. At run time:

1. **Lock on.** The model reads the question; at the end of every word of 3+ letters its latent is
   cosine-matched against node vectors (`tau=0.01`, **no learned query matrix**), and the sharpest
   match wins. Two details matter for free-form questions: node vectors are averaged over a few
   lead-ins (`K_ctx`: "", "the ", "a ", …) so a word matches mid-sentence too, and only nodes that
   have facts of their own can be locked onto. Without these, "a knife is used for " locked onto
   "fire" and "tell me about the ocean" onto "snout". With them, 8/8 hand-checked questions lock
   on correctly.
2. A personalized-PageRank walk over the graph scores facts (degree-normalized so hub nodes don't
   dominate). The top 3 are taken.
3. Their precomputed memories are summed and **held** in the model's state through the existing
   `dummy` slot (`mem·(1−decay)` per step keeps a constant offset) while the model answers.

Two things we learned the hard way:

- **Memory lasts about 3–4 bytes.** Every one of the 16×768 memory dimensions has decay ≤ 0.82. A
  fact read as text right before the question does *nothing* (the `text` condition sits at chance),
  because it's gone before the answer starts. That's why the injection is *held* instead of
  added once.
- **The sign flips.** Holding the memory of "…ocean." makes the model *less* likely to write "ocean"
  (it has just finished that word). A positive scale gives ~27–31% (steers away), a negative one
  steers toward the fact. A scale of −0.1 is the one-number stand-in for the learned output
  projection `W_o` the original design has at exactly this point. The sign was picked on two
  question sets and confirmed on five fresh ones. Stronger injection (−0.3, or summing 7 facts)
  wrecks the model's output, so keep an eye on the "log-prob change" line.

## Reading `ask.py`'s output: injection costs fluency

`ask.py` prints the model's free text with the graph off and on. The ON text usually looks *worse*.
That is real, and worth understanding before judging the idea by it:

```
GRAPH OFF:   what is a boat made of? 'My sent at more at more at more at more at more '
GRAPH ON:    what is a boat made of? 'benenenligeeeeGde. I say shat benenenligeyingeee'
```

- **OFF is not actually better.** It is stuck in a loop ("at more at more") — the failure mode of a
  model too small to continue a sentence, wearing real words. ON is broken differently: novel
  letter salad instead of a loop. Neither is an answer; injection did not ruin a working answer.
- **Why ON degrades.** The injection holds a constant extra vector in the model's memory on *every*
  step while it writes, and the model was never trained with anything held there. Measured: holding
  one fact costs ~0.05 log-prob per byte, holding the three retrieved facts ~0.45–0.74.
- **The knob.** More injection = more steering, less fluency. `--alpha -0.1` (default) steers
  reliably (80–93%) with visible damage to free text; `-0.3` measures higher but the output
  collapses; `-0.03` keeps the text closer to OFF with a weaker effect; `0` is identical to OFF.
- This is why the evidence is `inject_test.py` (which never asks the model to write, only which of
  two candidate words it prefers) rather than anything you can read out of the generated text.

**The fix is to train the model *with* injection active**, so held memory is in-distribution and
the model learns to read the answer out of it instead of being disturbed by it. See next steps.

## What it does NOT show yet

- **The effect is relative, not absolute.** Holding the graph's facts in memory vs holding
  nothing doesn't on its own raise the true answer (the `graph vs nothing` line sits at 39–54%).
  What's reliable is the *difference* between graph versions: same question, same retrieval,
  only the fact's content changed, and the answer moves with it. Also, the asked fact was among
  the 3 retrieved in only 29–34 of 56 questions (each concept has 6–7 facts). The 91% is over those.
- **The model doesn't answer correctly.** Top-1 multiple-choice accuracy stays at chance
  (~4–9%, it picks "house" for almost everything) with or without injection. The injection shifts
  its preferences reliably, but not far enough to flip the top answer. The blocker is the base
  model: 10M parameters, ~0.9MB of training text, ~3-byte memory. It can't remember "fish" by the
  time it reaches "fish is found in ___".
- **The steering direction is calibrated, not learned.** A proper version would train `W_o` (not
  `W_q`, see below) to map retrieved memory onto the answer, then test on edited facts it never
  saw.

## Next steps, if someone picks this up

1. **Fine-tune with injection switched on** (the highest-value next step, and affordable). Train on
   the fact-sentence family — the 526 facts plus the question templates — with the retrieved
   memory held in state *during training*, so the model learns to read the answer out of injected
   memory instead of being disturbed by it. A few hundred KB of targeted text is hours, not months,
   on one GPU. The obvious objection is that it would just memorize the facts, and the existing
   edit test answers it: fine-tune, then **edit the graph** and ask again. If the answer follows
   the edited fact — which was never trained on — the model is using the injected memory, not
   recall. That is the strongest version of the original claim.
2. A base model with longer memory (larger, trained longer, or with decay pushed toward 1 for some
   dimensions). Everything above is bottlenecked by the 3-byte horizon, and full fluency is out of
   reach here: this training loop runs at ~32 bytes/sec (~1MB per 8 hours) and the current
   checkpoint has seen ~0.9MB. Real fluency needs orders of magnitude more, so that call belongs
   with whoever can batch the RTRL loop or widen the decay range.
3. Train only `W_o` on top of this no-`W_q` retrieval, replacing the hand-picked −0.1 scale with a
   learned projection. Evaluate on graph edits to show it follows the graph rather than memorizing.
4. Earlier dead end, so nobody repeats it: a *learned* query matrix `W_q` collapsed to always
   retrieving the same node in every training run. Using the model's own latent as the query
   (this PoC) avoids that entirely.

## Files

| File | What it is |
|---|---|
| `prove.py` | **runs the whole proof end to end (~5 min) and narrates it, including what fails** |
| `real_graph_table.json` | 82 real concepts × standardized slots, filled by a Haiku model |
| `make_real_graph.py` | table → `real_graph.json` (8 relation types, visible hand-corrections) |
| `build_graph_tensors.py` | graph → node/fact vectors + adjacency (`real_graph_tensors.safetensors`) |
| `demo_no_query.py` | shows steps 1–2: what the graph walk retrieves for a word |
| `ask.py` | ask a full question: shows the lock-on, the retrieved facts, and the model's text with the graph off vs on |
| `test_graph_edit.py` | step 3: edit facts, rebuild, check retrieval follows |
| `inject_test.py` | step 4: the injection experiment (all conditions + full chain + edit) |
| `test_invisible.py` | graph-off is bit-identical to plain TMT |
| `graph_hooks.py`, `tmt_frozen.py` | walk + frozen-forward helpers |
| `demo_no_query_generate.py` | older experiment: injects at the decoder input instead of memory. Superseded, kept for reference |
| `train_base.py` | train the base TMT model on real text |
| `clean_corpus.py` | prep training corpus from public-domain books |
| `checkpoint_archiver.py` | hourly snapshots of base model weights during training |
| `archive_madeup/` | everything from the earlier made-up-names experiments (not needed) |

```bash
.venv/bin/python prove.py                        # everything, narrated, ~5 min -- start here

.venv/bin/python make_real_graph.py && .venv/bin/python build_graph_tensors.py
.venv/bin/python demo_no_query.py --queries ocean,fish,knife
.venv/bin/python ask.py "what is a boat made of? "
.venv/bin/python test_graph_edit.py --n-edits 8
.venv/bin/python inject_test.py --seed 5          # ~10 min on CPU
.venv/bin/python test_invisible.py
```

Notes: `inject_test.py` runs on CPU by default because MLX's CUDA backend hits `illegal memory
access` on long per-byte loops. Checkpoints are too large to share. Rebuild with `train_base.py`
(the upstream trace-fix commit `94a6064` changed the recurrence, so older checkpoints don't load).
