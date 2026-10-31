Machine-specific hardware and remote settings, read by `lab/core`.
Values that must never differ between machines (joint order, FPS, camera
names, tunnel port) are in `robot/config.py`, not here.
`remote.yaml` is gitignored; copy from `remote.yaml.example`.

## remote.yaml: the GPU host

`lab/core/remote.py` reads it for dataset sync and, later, SSH training jobs.
Missing, the lab server reports the GPU server as "not configured".

```yaml
host: gpu-box.tail1234.ts.net   # hostname or IP; a Tailscale MagicDNS name works
user: your_id
port: 22                        # optional, default 22
ssh_key: ~/.ssh/id_ed25519      # optional; omit to use ssh's defaults and agent
repo_dir: /home/your_id/MTREA   # the repo checkout on the GPU host
data_dir: /data/mtrea           # datasets land in <data_dir>/datasets/<name>
python: /home/your_id/.venv/bin/python   # the venv with lerobot; its bin/ goes on PATH
policy_port: 8000
gpu_count: 1
```

Unknown fields are refused, so a typo fails loudly instead of being ignored.

Before the first use, check that a non-interactive login works from the lab
machine. lab uses `BatchMode=yes`, so a password or host-key prompt fails
instead of waiting:

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 your_id@gpu-box.tail1234.ts.net -- true   # accept the host key once
ssh ... -- /home/your_id/.venv/bin/python --version                      # the configured python runs
ssh ... -- command -v rsync                                              # optional
```

Without rsync on either side, sync falls back to tar over ssh. That works but
is not resumable: a stopped transfer starts over. With rsync it resumes from
`<data_dir>/datasets/.incoming/<name>/`. On a Windows lab machine sync always
uses tar.
