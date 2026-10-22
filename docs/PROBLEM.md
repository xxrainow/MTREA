# Problem statement

MT-REA studies whether narrow fine-tuning of a pretrained vision-language-action
(VLA) model onto a new robot embodiment preserves the model's capability on
tasks it was *not* fine-tuned on.

## Setup: T, T_sub, held-out

- **T** — the broad task distribution the base VLA was pretrained on, on its
  original (source) embodiment. The pretrained model can already do these tasks
  there.
- **T_sub ⊂ T** — the small subset of tasks we actually fine-tune on, using
  teleoperated SO-101 demonstrations. In this project T_sub is a single task
  family, `pick_place`.
- **held-out** — `T \ T_sub`: tasks the base model could do but that we never
  fine-tune on (`stack`, `push`, `open_drawer`, `pour`). These are the
  measurement target. They are **not** a test split of the training data — they
  are a different set of task families, chosen to span decreasing motion overlap
  with `pick_place`.

The question is what fine-tuning on T_sub does to held-out performance on the
*new* embodiment (SO-101), relative to a baseline that also had to cross the
same embodiment gap.

## Research questions

- **RQ1 (phenomenon).** After H0, how much held-out capability does the model
  show on SO-101, and does it track motion overlap with `pick_place` (expected
  order `stack` > `push` > `pour` > `open_drawer`)?
- **RQ2 (method).** Can a fine-tuning method beyond H0 raise held-out success
  without lowering T_sub (`pick_place`) success?

## Why zero-shot is not a valid baseline

The natural-seeming baseline — run the pretrained model on SO-101 held-out tasks
with no fine-tuning — is invalid here. The base model was trained on a different
embodiment, so its vision inputs (camera placement, image statistics) and its
action space (joint layout, control frequency, calibration) do not match SO-101.
This embodiment mismatch floors zero-shot success near 0% for reasons that have
nothing to do with whether task-general skill was retained. A number near zero
tells us nothing about retention, and using it as a denominator is worse than
useless (see below).

Instead the baseline is **H0**: naive fine-tuning on T_sub. H0 has paid the same
embodiment-adaptation cost as every method under comparison, so differences in
held-out performance between H0 and a later method are attributable to *how* the
model was adapted, not to whether it was adapted at all.

## Reference points: floor and ceiling

Zero-shot is only a floor: embodiment mismatch alone puts it near 0%. Without
a ceiling as well, a low held-out number is uninterpretable — it cannot
distinguish "capability was lost" from "SO-101 never reaches more than this on
the task".

The ceiling is an **oracle**: a model fine-tuned with the same recipe as H0 on
demos of that held-out task itself, evaluated with the same protocol. It is run
for only 1–2 held-out tasks (which ones is TBD), because each needs its own
demo collection.

- The oracle is a reference point, not a baseline. H0 remains the baseline
  every method is compared against. The oracle is reported alongside as an
  absolute success rate, never used as a ratio denominator (see "Why absolute
  success rates, not ratios").
- Oracle demos are a separate dataset. They never enter H0/H1+ training, so the
  task stays held-out for every compared method.

## Why absolute success rates, not ratios

All comparisons are expressed as **absolute success rates** and their
**differences**, never as ratios. A ratio would divide by a baseline that can be
near zero — H0's held-out success, or zero-shot success — manufacturing
enormous, meaningless "improvements" (an increase from 1% to 3% is not a "3×
gain" in any sense we care about). Method A is judged better than method B on a
held-out task when its absolute success rate is higher by a margin that survives
the noise of a finite number of rollouts. Cross-task spread is reported as a
standard deviation across held-out tasks, again in absolute units.

## Core hypothesis

Fine-tuning a pretrained VLA onto a new embodiment conflates two distinct things:

- **(a) Embodiment adaptation** — learning to map SO-101's specific vision and
  action space: what its cameras see, how its joints move, how language grounds
  into *this* robot's observations and controls.
- **(b) Task-general skill** — the language-conditioned planning and manipulation
  competence retained from pretraining on T, which is what makes held-out tasks
  possible at all.

**Hypothesis:** naive fine-tuning (H0) updates every parameter to fit T_sub:
the vision-language model (VLM) that carries the task-general skill (b), and
the action expert that both (a) and (b) act through. Gradients from a single
task therefore reach both, so (a) and (b) *entangle*: adapting to SO-101 on
`pick_place` also moves the weights responsible for held-out tasks, corrupting
held-out performance. A method that separates the two should preserve held-out
performance better than H0 at equal T_sub performance. Because H0 restricts
nothing, choosing which modules adapt is a valid axis again (see "Methods
under comparison").

The project's job is to characterize this trade-off under realistic small-lab
constraints, not to win a leaderboard.

## Methods under comparison: H0, H1, H2, H3

H0, H1, H2, … denote fine-tuning methods being compared, all trained on the same
T_sub demos and evaluated on the same held-out families.

- **H0 — naive baseline.** Naive single-task full fine-tuning of π0.5
  (`lerobot/pi05_base`) on T_sub: all parameters train, with LeRobot's defaults
  `train_expert_only=false` and `freeze_vision_encoder=false`, at batch size 16.
  This is the entangled case the hypothesis predicts will degrade held-out
  performance most, and every other method is compared against H0.
  - Full fine-tuning is the baseline because it is the default in both openpi
    and LeRobot, and it is the standard baseline in the forgetting literature
    (RETAIN, Garg et al.).
  - It fits on the project's GPU, one A100 MIG 3g.40gb slice: bf16 with
    gradient checkpointing peaks at about 37 GiB at batch 16 (docs/COMPUTE.md).
  - The one deviation from the standard recipe is scale: batch 16 instead of
    openpi's 32–256, and a smaller sample budget, both forced by a single MIG
    slice (bs 32 runs out of memory, LeRobot has no gradient accumulation, and
    throughput saturates around 4 samples/s).
  - H0 is *not* LoRA-based. LoRA support for π0.5 is unreliable or unimplemented
    in both the openpi and LeRobot training paths and has been dropped from this
    project entirely.
- **H1, H2, H3 — beyond H0.** Methods informed by the hypothesis above that
  limit how far fine-tuning on T_sub can move the weights held-out tasks rely
  on. Candidate axes:
  - restricting which modules adapt. The first comparison is **expert-only
    fine-tuning** (`train_expert_only=true`, method `expert_only` in
    `research/train.py`): the VLM is frozen and only the action expert and
    projections train, following the knowledge-insulation argument that
    keeping action-learning gradients out of the VLM preserves its pretrained
    knowledge;
  - restricting which updates are allowed within the action expert;
  - regularizing toward the pretrained weights;
  - changing the fine-tuning data.

  Which methods become H1–H3 is still open. The concrete parameterization of
  each is specified when that method is run; what they share is testing whether
  they retain held-out capability that H0 loses.
