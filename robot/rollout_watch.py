"""Read rollout timing snapshots without connecting to hardware or a server."""

import argparse
import json
import time
from pathlib import Path

from robot.config import data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_root())
    parser.add_argument("--trial-dir", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--joints", action="store_true", help="Compare measured joints and policy targets")
    args = parser.parse_args()
    root = args.data_dir / "rollouts" / "robot"
    print(f"Reading: {args.trial_dir or root} (Ctrl+C stops monitoring only)", flush=True)
    previous_path = None
    try:
        while True:
            files = ([args.trial_dir / "status.json"] if args.trial_dir else
                     list(root.glob("*/status.json")))
            if not files:
                print("No rollout status files yet; waiting for a new rollout.", flush=True)
            else:
                try:
                    path = max(files, key=lambda p: p.stat().st_mtime_ns)
                    status = json.loads(path.read_text())
                    if path != previous_path:
                        print(f"Trial: {path.parent}", flush=True)
                        previous_path = path
                    print(f"state={status.get('state')} queue={status.get('queue_size')} "
                          f"stalls={status.get('stalls')} "
                          f"latency_ms={status.get('last_latency_ms')} "
                          f"missed_ticks={status.get('control_deadlines_skipped', '—')}", flush=True)
                    sender = status.get("observation_sender")
                    if sender:
                        print(f"  sender: pending={sender['pending']} "
                              f"in_flight={sender['in_flight']} sent={sender['sent']} "
                              f"replaced={sender['replaced']} stale={sender['stale']} "
                              f"superseded={sender.get('superseded', '—')}", flush=True)
                    if args.joints:
                        tracking = status.get("joint_tracking") or {}
                        observed = tracking.get("measured") or tracking.get("observed") or {}
                        action = tracking.get("action") or {}
                        measured = observed.get("positions", {})
                        requested = action.get("requested", {})
                        sent = action.get("sent") or {}
                        if not measured or not requested:
                            print("  Joint tracking unavailable: start a new rollout with current code.", flush=True)
                        else:
                            print(f"  joints: measured_at={observed['timestamp']:.3f} "
                                  f"action_at={action['timestamp']:.3f} "
                                  f"clamped_actions={tracking.get('clamped_actions', 0)}", flush=True)
                            for name, target in requested.items():
                                actual = f"{measured[name]:.2f}" if name in measured else "—"
                                commanded = f"{sent[name]:.2f}" if name in sent else "—"
                                print(f"    {name}: measured={actual} requested={target:.2f} sent={commanded}",
                                      flush=True)
                    timing = status.get("timing", {})
                    if not timing:
                        print("No timing data: start a NEW rollout with the updated worker.", flush=True)
                    for name, values in timing.items():
                        if args.joints and name != "action_interval":
                            continue
                        print(f"  {name}: last={values['last_ms']:.1f} "
                              f"mean={values['mean_ms']:.1f} max={values['max_ms']:.1f} ms "
                              f"(recent n={values['recent_count']})", flush=True)
                except (OSError, ValueError) as exc:
                    print(f"Cannot read snapshot: {exc}", flush=True)
            if args.once:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("Monitoring stopped.", flush=True)


if __name__ == "__main__":
    main()
