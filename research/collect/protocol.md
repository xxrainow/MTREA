# Data collection protocol

Rules for recording teleoperation demos. Task families and their roles
(T_sub vs held-out) are defined in `tasks/*.yaml`; this file covers how the
episodes for a family must be collected.

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
