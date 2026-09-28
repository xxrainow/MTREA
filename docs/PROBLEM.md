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

**Hypothesis:** naive fine-tuning (H0) updates only the action expert's
parameters to fit T_sub, leaving the frozen vision-language backbone unchanged,
yet this still *entangles* (a) and (b) — because the action expert is the shared
machinery both rely on, tuning it toward the single fine-tuned task drags the
skill responsible for held-out tasks along with it, corrupting held-out
performance. Because H0 already freezes the entire vision-language backbone,
choosing which top-level module to freeze cannot make a method more selective
than H0: unfreezing more modules (e.g. the vision encoder alongside the action
expert) tunes *more* than H0, not less. Methods beyond H0 must therefore
intervene on a different axis — for example restricting which updates are
allowed *within* the action expert, regularizing toward the pretrained weights,
or changing the fine-tuning data. The concrete choice is left open. Such a
method should preserve held-out performance better than H0 at equal T_sub
performance.

The project's job is to characterize this trade-off under realistic small-lab
constraints, not to win a leaderboard.

## Methods under comparison: H0, H1, H2, H3

H0, H1, H2, … denote fine-tuning methods being compared, all trained on the same
T_sub demos and evaluated on the same held-out families.

- **H0 — naive baseline.** Naive single-task fine-tuning of π0.5 on T_sub with
  `train_expert_only=true`: the pretrained vision-language backbone is frozen and
  only the action expert is trained. This is *not* a LoRA-based method — LoRA
  support for π0.5 is unreliable/unimplemented in both the openpi and LeRobot
  training paths and has been dropped from this project entirely. This is the
  entangled case the hypothesis predicts will degrade held-out performance most,
  and every other method is compared against H0.
  - The standard, unconstrained approach would be full fine-tuning of *all*
    parameters, but that needs ~80 GB of GPU memory, exceeding the 40 GB A100
    available to this project.
  - `train_expert_only=true` is LeRobot's documented memory-saving
    configuration for π0.5. Published SO-101 π0.5 recipes use full fine-tuning
    on 80 GB+ GPUs, which this project cannot run.
  - Framing implication: because the vision-language backbone is left untouched,
    any held-out degradation observed under H0 cannot be attributed to
    catastrophic forgetting in the semantic/language pathway alone. It would
    instead indicate that even action-expert-only adaptation is sufficient to
    disrupt transfer — a stronger and more specific finding than naive
    full-parameter fine-tuning would demonstrate.
- **H1, H2, H3 — beyond H0.** Methods informed by the hypothesis above that
  intervene on an axis other than which top-level module is frozen (H0 already
  freezes the whole vision-language backbone) — e.g. restricting updates within
  the action expert, regularizing toward the pretrained weights, or changing the
  fine-tuning data. The concrete parameterization of each is specified when that
  method is run; what they share is testing whether they retain held-out
  capability that H0 loses.
