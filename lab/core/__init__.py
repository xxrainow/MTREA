"""lab.core — robot control, recording, job running, state.

Knows nothing about HTTP. Never imports FastAPI; runner never imports
research (it runs shell wrappers as subprocesses). See AGENTS.md.
"""
