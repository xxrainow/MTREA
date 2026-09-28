# Data collection protocol

Rules for recording teleoperation demos (T_sub) and for running autonomous
evaluation rollouts (held-out). Task families and their roles are defined in
`tasks/*.yaml`; this file covers how the episodes for a family must be collected
and how held-out tasks are measured.

## Object diversity

Within a single task family, do not record every episode with the same object.
The `{object}` placeholder in each task's `instruction_template` exists so the
model learns the task's motion pattern rather than memorizing one object's
appearance.

Concrete rule, per task family:

- Use **at least 3 distinct object colors** across the family's episodes.
- Use **at least 2 distinct object types** (e.g. block and cup, not two blocks).
- No single (color, type) pairing may account for more than half of a family's
  `target_episodes`.
- Each episode's concrete instruction — the actual `{object}`/`{target}` filled
  in — is recorded in the LeRobotDataset with that episode, not in the yaml.

## Evaluation protocol

Held-out tasks are never teleop-recorded for training. Their `role: held_out`
yaml carries `target_eval_trials`, the number of autonomous rollout attempts the
fine-tuned policy makes, used only to estimate a success rate.

Concrete rule, per held-out task family:

- Run **30 rollout trials** (`target_eval_trials`). Score each trial against the
  task's `success_condition`; the success rate is successes / 30.
- **Reset between trials by reference-image overlay.** Before each trial, overlay
  a fixed reference image of the intended starting scene on the live camera feed
  and manually move the objects until their placement matches the overlay. This
  gives a reproducible starting state without motion-capture or other precision
  tracking hardware. This is the reset method VLA-REPLICA uses for the same
  purpose.
- Vary object identity/color across the 30 trials following the same **Object
  diversity** rule above, so the success rate reflects the task's motion pattern
  rather than one object instance.
