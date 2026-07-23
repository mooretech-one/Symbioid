#!/usr/bin/env python3
"""Smoke demo: Sensor/Observation formation; six-sets print on completion only.

Runs until interrupted with Ctrl-C (or SIGTERM).

Feedback test wiring: ear = sin(hand.output), eye = cos(hand.output).
"""

import math
import signal
import threading

from symbioid import Symbioid, format_six_set_line


def main() -> None:
    s = Symbioid(label="demo")
    eye = s.add_sensor(label="eye")
    ear = s.add_sensor(label="ear")
    # Cease continuous Inputs after N samples (sets) per sensor.
    eye.max_samples = 60
    ear.max_samples = 40
    hand = s.add_actuator(label="hand")
    hand.output = 0.0
    hand.output_step = 0.2  # radians advanced each gated fire

    # Closed-loop Feedback: sensors read functions of actuator output
    ear.transfer = lambda w: math.sin(w.get("hand", 0.0))
    eye.transfer = lambda w: math.cos(w.get("hand", 0.0))

    twin = s.twin_seed_thoughts()
    if twin:
        print(format_six_set_line("twin", twin, index=0), flush=True)

    stop = threading.Event()

    def _request_stop(signum: int, frame: object) -> None:
        stop.set()

    # Handle Ctrl-C and kill; restore defaults after so nested Ctrl-C can force-exit.
    prev_int = signal.signal(signal.SIGINT, _request_stop)
    prev_term = signal.signal(signal.SIGTERM, _request_stop)

    print("Running (Ctrl-C to stop)…", flush=True)
    s.start_processes()
    try:
        # Timed wait so the signal handler can run between waits (bare wait can miss SIGINT).
        while not stop.wait(timeout=0.25):
            pass
    except KeyboardInterrupt:
        # Fallback if a default SIGINT still surfaces
        pass
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        print("\nStopping processes…", flush=True)
        s.stop_processes()
        active = s.innerface.active_set_summary()
        acts = [
            (a.label, a.fire_count, a.deny_count) for a in s.actuators
        ]
        print(
            f"stopped: samples={[ (sen.label, sen.sample_count) for sen in s.sensors ]} "
            f"formations={len(s.innerface.completed_formations)} "
            f"syncs={len(s.innerface.completed_syncs)} "
            f"integrates={len(s.innerface.completed_integrates)} "
            f"depth_folds={s.innerface.depth_fold_count} "
            f"active_sets={s.innerface.active_set_count} {active} "
            f"beliefs={len(s.outerface.active_belief_ids)} "
            f"(created={s.outerface.beliefs_created} "
            f"updated={s.outerface.beliefs_updated} "
            f"confirm={s.outerface.belief_confirms} "
            f"challenge={s.outerface.belief_challenges} "
            f"stale={s.outerface.belief_stale_skips}) "
            f"actuator_fires={s.outerface.actuator_fires} "
            f"actuators={acts}",
            flush=True,
        )


if __name__ == "__main__":
    main()
