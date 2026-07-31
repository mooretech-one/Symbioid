#!/usr/bin/env python3
"""
Phase 0 GPU feasibility baselines for Symbioid (CPU measurements).

No torch/cupy required. Prints JSON-ish tables for Work-Log.

  PYTHONPATH=. .venv/bin/python scripts/bench_gpu_phase0.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from symbioid import Symbioid, Thought, Link
from symbioid.Core.spectral import SpectralBank, HolonomicStore, key_to_vector


def _median_ms(fn, n_warm: int = 3, n_run: int = 30) -> float:
    for _ in range(n_warm):
        fn()
    samples = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def bench_pulse() -> list[dict]:
    rows = []
    for n_poles in (100, 500, 1000, 2000, 4000):
        s = Symbioid(install_constitution=False)
        s.mind.dynamics_enabled = True
        s.mind.spectral_mix_enabled = False
        poles = []
        for i in range(n_poles):
            t = Thought(label=f"p{i}", activation=0.5 + (i % 7) * 0.1)
            s.add_thought(t)
            poles.append(t)
        # 2 outgoing links per pole (ring + skip)
        lt = poles[0]
        for i, t in enumerate(poles):
            for j in (1, 3):
                tgt = poles[(i + j) % n_poles]
                link = Link(source=t, target=tgt, link_type=lt, weight=1.0)
                s.add_thought(link)
        # heat a fraction so hot-set path is exercised
        for t in poles[::2]:
            t.receive(1.5)
            s._hot_ids.add(t.id)

        def once():
            s.pulse_tick()

        ms = _median_ms(once, n_warm=5, n_run=25)
        n_thoughts = len(s.thoughts)
        rows.append(
            {
                "poles": n_poles,
                "thoughts": n_thoughts,
                "pulse_ms": round(ms, 4),
                "us_per_thought": round(ms * 1000.0 / max(1, n_thoughts), 3),
            }
        )
    return rows


def bench_spectral() -> list[dict]:
    rows = []
    for size in (64, 256, 1024, 4096, 16384):
        bank = SpectralBank(size=size)
        # bind synthetic slots
        for i in range(min(size, 64)):
            bank.bind(f"t{i}", prefer_slot=i)
        host_thoughts = {}
        for i in range(min(size, 64)):
            host_thoughts[f"t{i}"] = type("T", (), {"activation": float(i % 5) * 0.2})()

        class H:
            thoughts = host_thoughts

        h = H()
        bank.pack_from_activations(h)
        # pack path

        def mix():
            bank.pack_from_activations(h)
            bank.apply_mix_filter(soft_threshold=0.01, lowpass=0.9)
            bank.unpack_to_activations(h, gain=0.15, mode="add")

        ms = _median_ms(mix, n_warm=5, n_run=40)
        # pure FFT only
        bank.pack_from_activations(h)

        def fft_only():
            bank.fft()
            bank.ifft()

        ms_fft = _median_ms(fft_only, n_warm=5, n_run=40)
        rows.append(
            {
                "bank_size": bank.size,
                "mix_ms": round(ms, 4),
                "fft_ifft_ms": round(ms_fft, 4),
            }
        )
    return rows


def bench_holonomic() -> list[dict]:
    rows = []
    for n in (64, 256, 1024, 4096):
        store = HolonomicStore(capacity=n)
        keys = [f"key-{i}" for i in range(32)]
        for k in keys:
            store.write_key(k, strength=0.5)

        def probe():
            for k in keys:
                store.score_key(k)

        ms = _median_ms(probe, n_warm=3, n_run=30)
        rows.append({"store_size": store.capacity, "score_32_keys_ms": round(ms, 4)})
    return rows


def bench_placement() -> dict:
    from random import Random
    from symbioid.world.tetris import TetrisWorld, ActionCipher
    from symbioid.world.tetris_learn import TetrisCoach

    # Import demo scoring without pygame path issues
    sys.path.insert(0, str(ROOT))
    import tetris_demo as mod

    rng = Random(1)
    cipher = ActionCipher.fixed({1: "left", 2: "right", 3: "rotate", 4: "hard"})
    w = TetrisWorld(rng=rng, cipher=cipher, gravity_interval=9999)
    s = mod.build_symbioid(w)
    coach = TetrisCoach(rng=Random(2), network_primary=True, map_threshold=1)
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach.effect_counts[b][e] = 3
        coach.bytes_tried.add(b)
    coach.graph_placement_weight = 0.60
    coach.graph_placement_bonus = (
        lambda world, rot, col, _s=s: mod.cell_thought_placement_score(_s, world, rot, col)
    )
    # warm sample
    for i in range(20):
        mod.sample_into_symbioid(s, w, tick=i)
        s.pulse_tick()

    opts = w.legal_placements()
    n_opts = len(opts)

    def score_all():
        for rot, col in opts:
            mod.cell_thought_placement_score(s, w, rot, col)

    ms_scores = _median_ms(score_all, n_warm=3, n_run=20)

    def choose():
        coach._target = None
        coach.choose_target(w)

    ms_choose = _median_ms(choose, n_warm=2, n_run=15)

    def pulse():
        s.pulse_tick()

    ms_pulse = _median_ms(pulse, n_warm=5, n_run=30)

    return {
        "legal_placements": n_opts,
        "score_all_poses_ms": round(ms_scores, 4),
        "ms_per_pose": round(ms_scores / max(1, n_opts), 5),
        "choose_target_ms": round(ms_choose, 4),
        "pulse_after_sample_ms": round(ms_pulse, 4),
        "thoughts": len(s.thoughts),
        "hot": len(getattr(s, "_hot_ids", ()) or ()),
    }


def bench_vector_vs_object() -> list[dict]:
    """Phase 2: object vs vector backend at Phase 0 pole sizes."""
    rows = []
    for n_poles in (2000, 4000):
        for backend in ("object", "vector"):
            s = Symbioid(install_constitution=False)
            s.mind.dynamics_enabled = True
            s.mind.spectral_mix_enabled = False
            s.mind.holonomic_store_enabled = False
            s.mind.set_dynamics_backend(backend)
            poles = []
            for i in range(n_poles):
                t = Thought(label=f"p{i}", activation=0.5 + (i % 7) * 0.1)
                s.add_thought(t)
                poles.append(t)
            lt = poles[0]
            for i, t in enumerate(poles):
                for j in (1, 3):
                    s.add_thought(
                        Link(
                            source=t,
                            target=poles[(i + j) % n_poles],
                            link_type=lt,
                            weight=1.0,
                        )
                    )
            for t in poles[::2]:
                t.receive(1.5)
                s.mark_hot(t)
            s.pulse_tick()  # stabilize hot
            ms = _median_ms(lambda: s.pulse_tick(), n_warm=5, n_run=20)
            rows.append(
                {
                    "backend": backend,
                    "poles": n_poles,
                    "thoughts": len(s.thoughts),
                    "hot": len(s._hot_ids),
                    "pulse_ms": round(ms, 4),
                }
            )
    return rows


def main() -> None:
    print("=== Phase 0 Symbioid GPU baselines (CPU) ===")
    print(f"numpy {np.__version__}")
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        print(f"gpu: {out}")
    except Exception as exc:
        print(f"gpu: unavailable ({exc})")
    print("torch: not installed (expected Phase 0)")

    print("\n## pulse_tick vs graph size")
    for r in bench_pulse():
        print(
            f"  poles={r['poles']:5d} thoughts={r['thoughts']:5d} "
            f"pulse={r['pulse_ms']:8.3f} ms  ({r['us_per_thought']:.2f} µs/thought)"
        )

    print("\n## spectral pack+mix+unpack vs bank size")
    for r in bench_spectral():
        print(
            f"  n={r['bank_size']:5d}  mix={r['mix_ms']:8.4f} ms  "
            f"fft_ifft={r['fft_ifft_ms']:8.4f} ms"
        )

    print("\n## holonomic score 32 keys")
    for r in bench_holonomic():
        print(
            f"  store_n={r['store_size']:5d}  score32={r['score_32_keys_ms']:8.4f} ms"
        )

    print("\n## tetris placement (network-primary build)")
    p = bench_placement()
    for k, v in p.items():
        print(f"  {k}: {v}")

    # Break-even heuristic: GPU launch overhead often ~5–50µs; need ms-scale work
    print("\n## object vs vector pulse (Phase 2)")
    for r in bench_vector_vs_object():
        print(
            f"  {r['backend']:6s} poles={r['poles']:5d} thoughts={r['thoughts']:5d} "
            f"hot={r['hot']:5d} pulse={r['pulse_ms']:8.3f} ms"
        )

    print("\n## Phase 0 read")
    print(
        "  Spectral default n=64 mix is sub-0.1ms class → CUDA unlikely to help until n>>1k."
    )
    print(
        "  Pulse scales ~linear in thoughts → vectorize on CPU first; GPU when N large + CSR resident."
    )
    print(
        "  Placement score_all / choose_target multi-ms → batch features (Phase 3) high leverage."
    )


if __name__ == "__main__":
    main()
