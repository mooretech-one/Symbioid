#!/usr/bin/env python3
"""
Phase 4 TFT A/B: headless multi-game comparison of strategy knobs.

Configs (all keep credit hygiene 0.0.52):
  A  hygiene_only   — tft.enabled=False, no warm-start
  B  ship_default   — tft on (forgive N=4, gamma=0.5), warm-start 0.12
  C  aggressive_forgive — forgive_after_n_c=2, gamma=0.25, warm-start
  D  generous_noise — forgive_random_d_prob=0.15 + ship_default

Usage:
  PYTHONPATH=. .venv/bin/python scripts/bench_tft_phase4.py
  PYTHONPATH=. .venv/bin/python scripts/bench_tft_phase4.py --games 5 --max-frames 1500 --seeds 1,2,3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris_demo import run_multi_game_metric  # noqa: E402


def _cfg_hygiene_only(s) -> None:
    s.mind.tft.config.enabled = False
    s.mind.warm_start_actions = False


def _cfg_ship_default(s) -> None:
    s.mind.tft.config.enabled = True
    s.mind.tft.config.forgive_after_n_c = 4
    s.mind.tft.config.forgive_gamma = 0.5
    s.mind.tft.config.forgive_random_d_prob = 0.0
    s.mind.warm_start_actions = True
    s.mind.warm_start_prior = 0.12


def _cfg_aggressive_forgive(s) -> None:
    _cfg_ship_default(s)
    s.mind.tft.config.forgive_after_n_c = 2
    s.mind.tft.config.forgive_gamma = 0.25


def _cfg_generous_noise(s) -> None:
    _cfg_ship_default(s)
    s.mind.tft.config.forgive_random_d_prob = 0.15


CONFIGS = {
    "A_hygiene_only": _cfg_hygiene_only,
    "B_ship_default": _cfg_ship_default,
    "C_aggressive_forgive": _cfg_aggressive_forgive,
    "D_generous_noise": _cfg_generous_noise,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TFT Phase 4 A/B headless bench")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument(
        "--configs",
        type=str,
        default=",".join(CONFIGS.keys()),
        help="Comma-separated config names",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional JSON output path",
    )
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    names = [x.strip() for x in args.configs.split(",") if x.strip()]

    results: list[dict] = []
    t0 = time.time()
    for name in names:
        setup = CONFIGS.get(name)
        if setup is None:
            print(f"unknown config {name}", file=sys.stderr)
            return 2
        for seed in seeds:
            print(
                f"=== {name} seed={seed} games={args.games} "
                f"max_frames={args.max_frames} ===",
                flush=True,
            )
            t1 = time.time()
            rows, summary = run_multi_game_metric(
                games=int(args.games),
                max_frames=int(args.max_frames),
                seed=int(seed),
                verbose=True,
                mind_setup=setup,
            )
            elapsed = time.time() - t1
            rec = {
                "config": name,
                "seed": seed,
                "elapsed_s": round(elapsed, 2),
                "summary": summary,
                "games": [r.as_dict() for r in rows],
            }
            results.append(rec)
            print(
                f"  -> mean_score={summary.get('mean_score', 0):.1f} "
                f"c_rate={summary.get('c_rate', 0):.3f} "
                f"top_out_rate={summary.get('top_out_rate', 0):.2f} "
                f"mean_frames={summary.get('mean_frames', 0):.0f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    # Aggregate by config across seeds
    print("\n======== AGGREGATE (mean of seed means) ========", flush=True)
    by_cfg: dict[str, list[dict]] = {}
    for r in results:
        by_cfg.setdefault(r["config"], []).append(r["summary"])
    agg = {}
    for name, sums in by_cfg.items():
        n = float(len(sums)) or 1.0
        keys = (
            "mean_score",
            "mean_lines",
            "mean_holes",
            "mean_pieces",
            "mean_frames",
            "c_rate",
            "top_out_rate",
            "mean_C",
            "mean_D",
        )
        row = {k: sum(float(s.get(k, 0) or 0) for s in sums) / n for k in keys}
        agg[name] = row
        print(
            f"{name:24s}  score={row['mean_score']:7.1f}  "
            f"c_rate={row['c_rate']:.3f}  top_out={row['top_out_rate']:.2f}  "
            f"frames={row['mean_frames']:.0f}  holes={row['mean_holes']:.2f}",
            flush=True,
        )

    # Rank by mean_score then c_rate
    ranked = sorted(
        agg.items(),
        key=lambda kv: (kv[1]["mean_score"], kv[1]["c_rate"]),
        reverse=True,
    )
    print("\n======== RANK (score, then c_rate) ========", flush=True)
    for i, (name, row) in enumerate(ranked, 1):
        print(f"  {i}. {name}  score={row['mean_score']:.1f}", flush=True)

    total = time.time() - t0
    print(f"\nTotal wall {total:.1f}s", flush=True)

    out = {
        "games": int(args.games),
        "max_frames": int(args.max_frames),
        "seeds": seeds,
        "aggregate": agg,
        "rank": [n for n, _ in ranked],
        "runs": results,
        "wall_s": round(total, 2),
        "version": "0.0.55+",
    }
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"Wrote {path}", flush=True)
    else:
        # default under scripts/
        path = ROOT / "scripts" / "bench_tft_phase4_results.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"Wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
