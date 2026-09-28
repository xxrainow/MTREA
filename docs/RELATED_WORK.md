# Related work

Three lines of work bear directly on MT-REA: where in a VLA a distribution shift
concentrates, what fine-tuning onto a new embodiment costs, and how much
pretrained VLAs forget under continual learning. Each informs our method; none
covers the setting we target. See `PROBLEM.md` for our setup.

## Not All Layers Need Tuning (arXiv:2609.18084)

*"Diagnosing and Directing Adaptation in Vision-Language-Action Models."*

Finds an **adaptation spectrum**: different kinds of distribution shift
concentrate in different layer regions of a VLA. Appearance shifts localize to
the **vision encoder**; instruction/language shifts localize to the **language
backbone**; novel object or geometry shifts require the **vision encoder and
action head together**. The paper diagnoses where a given shift lives using
CKA-based (centered kernel alignment) layer comparisons and then directs
adaptation there via **adaptive LoRA rank** — spending capacity on the regions
that actually need to move.

*Position.* This is the mechanistic backbone of our hypothesis. It tells us
*which* parameters plausibly carry embodiment adaptation (a) versus task-general
skill (b): a cross-embodiment shift is object/geometry-like, so we expect it to
live in the vision encoder + action head, and we design H1–H3 to confine tuning
there. We differ in what we measure and how: they study directing adaptation to
improve the *target* task; we hold the target-task recipe fixed and measure the
*collateral* effect on held-out tasks under a strict few-shot, single-GPU budget.

## UAM: A Dual-Stream Perspective on Forgetting in VLA Training (arXiv:2605.15735)

Names the **embodiment tax**: fine-tuning a VLA onto a new embodiment costs
previously held multimodal capability. Proposes a **dual-stream architecture** —
a **ventral** stream that preserves pretrained semantics and a **dorsal** stream
that handles new embodiment-specific visual processing — reporting retention of
**>95%** of the original multimodal capability.

*Position.* UAM names exactly the phenomenon we quantify (the embodiment tax) and
shows it is avoidable *if you can change the architecture*. We deliberately take
the opposite constraint: **no architecture changes**. We ask how far selective
fine-tuning of the existing network alone can go, because adding and training a
second stream is beyond what a small lab with one consumer-scale GPU and tens of
demos can do. UAM is our aspirational ceiling and our motivation, not our method.

## Pretrained VLAs are Surprisingly Resistant to Forgetting (arXiv:2603.03818)

*"...in Continual Learning."* Shows that large pretrained VLAs **resist
forgetting** when learning a sequence of tasks with replay — the pretrained
initialization is more robust than continual-learning intuition from smaller
models would predict.

*Position.* Encouraging, but a **different setting**. Its result is
**same-embodiment, multi-task-over-time with replay**: many tasks arrive
sequentially on one robot, and past-task data is replayed. Ours is
**single-step, cross-embodiment, few-shot with no replay**: one fine-tuning
episode onto a new robot from tens of demos, measuring capability that was never
in the fine-tuning stream. Their forgetting-resistance result does not transfer
to our regime, and whether it holds here is an open question our measurements
speak to.

## The gap

None of the three tests the setting a small lab actually faces:

- **few-shot** — tens of demonstrations, not thousands;
- **single-step** — one fine-tuning pass onto a new embodiment, not a continual
  stream, and no replay buffer;
- **cross-embodiment** — a genuine embodiment change (SO-101), not appearance or
  instruction shift on the same robot;
- **no architecture changes** — adapt the existing network in place, no added
  streams or modules;
- **single consumer-scale GPU** — the compute a small lab has, not a cluster.

Characterizing held-out capability retention under exactly these constraints —
and testing whether selective, adaptation-spectrum-informed tuning beats naive
fine-tuning within them — is what this project contributes.
