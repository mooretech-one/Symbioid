#!/usr/bin/env python3
"""
Symbioid audio demo — Phase 0–1: mic/synthetic → FFT20 → 20 Sensors.

Sense-only for now (no actuators / speakers). Host loop matches Pong/Tetris:
sample → formation handoffs → pulse. Optional pygame band HUD.

  PYTHONPATH=. .venv/bin/python audio_demo.py              # synthetic bursts
  PYTHONPATH=. .venv/bin/python audio_demo.py --mic        # ALSA C925e (plughw:1,0)
  PYTHONPATH=. .venv/bin/python audio_demo.py --headless --frames 30
  PYTHONPATH=. .venv/bin/python audio_demo.py --list-devices
  PYTHONPATH=. .venv/bin/python audio_demo.py --tone 1000 --frames 10 --headless

Env: SYMBIOID_AUDIO_ALSA_DEVICE or GROK_AUDIO_ALSA_DEVICE (default hw:1,0).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from symbioid import (
    Sensor,
    Symbioid,
    default_memory_path,
    save_memory,
    set_console_emit,
    try_load_into,
)
from symbioid.world.audio import (
    CHUNK_SIZE,
    DEFAULT_ALSA_DEVICE,
    NUM_BANDS,
    SAMPLE_RATE,
    AudioWorld,
    open_capture,
    probe_arecord_devices,
)

HOST_ID = "sym-audio-fft20"
DEFAULT_MEMORY = default_memory_path("audio_memory.json")

# UI
W, H = 720, 420
FPS = 24  # ~ block rate (48k/2048 ≈ 23.4 Hz)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Symbioid audio demo Phase 0–1 (FFT20 sense)"
    )
    p.add_argument(
        "--mic",
        action="store_true",
        help=f"Live ALSA capture (default device {DEFAULT_ALSA_DEVICE})",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="ALSA device for arecord (e.g. hw:1,0 or plughw:1,0)",
    )
    p.add_argument(
        "--backend",
        choices=("synthetic", "arecord", "auto"),
        default=None,
        help="Capture backend (default: synthetic, or arecord with --mic)",
    )
    p.add_argument(
        "--tone",
        type=float,
        default=None,
        metavar="HZ",
        help="Synthetic pure tone at HZ (implies synthetic backend)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="No pygame window; print band summary lines",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after N audio blocks (0 = run until quit)",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="Print arecord -l and exit",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable six-set console dumps",
    )
    p.add_argument(
        "--memory",
        type=Path,
        default=DEFAULT_MEMORY,
        help=f"Agent memory JSON (default {DEFAULT_MEMORY})",
    )
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--reset-memory", action="store_true")
    return p.parse_args(argv)


def build_symbioid(world: AudioWorld) -> Symbioid:
    s = Symbioid(id=HOST_ID, label="audio-fft20")
    s.interface.continuous_inputs = False
    s.outerface.wait_for_feedback = False

    for i in range(world.num_bands):
        lab = f"band_{i:02d}"
        sen = s.add_sensor(
            Sensor(id=f"{HOST_ID}:sen:{lab}", label=lab),
            awareness=False,  # 20 terminators; skip full awareness six-sets
        )
        # Capture i in default arg so closure is correct
        sen.transfer = lambda w, key=lab: float(w.get(key, 0.0))

    return s


def sample_into_symbioid(s: Symbioid, world: AudioWorld, tick: int) -> int:
    w = world.sensor_world()
    handoffs = []
    for sen in s.sensors:
        sense = sen.sample(tick=tick, world=w)
        if sense is None:
            continue
        h = s.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h is not None:
            handoffs.append(h)
    if not handoffs:
        return 0
    if len(handoffs) > 1:
        s.innerface.post(
            {"kind": "formation_batch", "handoffs": handoffs, "tick": tick}
        )
    else:
        s.innerface.post(handoffs[0])
    return len(handoffs)


def draw_hud(
    screen,
    world: AudioWorld,
    s: Symbioid,
    backend: str,
    font,
    font_sm,
) -> None:
    import pygame

    screen.fill((12, 14, 22))
    title = font.render(
        f"Symbioid audio FFT{world.num_bands}  [{backend}]",
        True,
        (200, 210, 230),
    )
    screen.blit(title, (16, 12))

    n = world.num_bands
    margin_x = 24
    top = 56
    bar_w = (W - 2 * margin_x) // n
    bar_max_h = H - 140
    for i, v in enumerate(world.bands):
        h = max(1, int(v * bar_max_h))
        x = margin_x + i * bar_w
        y = top + bar_max_h - h
        # cool → hot by band index
        col = (
            min(255, 40 + i * 8),
            min(255, 120 + int(v * 100)),
            min(255, 200 - i * 4),
        )
        pygame.draw.rect(screen, col, (x + 2, y, bar_w - 4, h))
        if i % 2 == 0:
            lab = font_sm.render(f"{i:02d}", True, (90, 100, 120))
            screen.blit(lab, (x + 2, top + bar_max_h + 4))

    stats = font_sm.render(
        f"blocks={world.blocks}  rms={world.last_rms:.4f}  peak={world.last_peak:.4f}  "
        f"cap_ms={world.last_block_ms:.1f}  thoughts={len(s.thoughts)}  "
        f"formations={s.innerface.formation_ticks}",
        True,
        (160, 170, 190),
    )
    screen.blit(stats, (16, H - 48))
    hint = font_sm.render(
        "Phase 0–1 sense only · Esc quit · mic: --mic",
        True,
        (100, 110, 130),
    )
    screen.blit(hint, (16, H - 28))


def run_headless(
    s: Symbioid,
    world: AudioWorld,
    capture,
    backend: str,
    frames: int,
) -> int:
    target = frames if frames > 0 else 60
    print(
        f"[audio] headless backend={backend} device={getattr(capture, 'device', '-')} "
        f"sr={world.sample_rate} chunk={world.chunk_size} blocks={target}",
        flush=True,
    )
    for tick in range(1, target + 1):
        world.step(capture)
        n_h = sample_into_symbioid(s, world, tick)
        s.pulse_tick()
        # Let face workers drain formation inbox
        time.sleep(0.03)
        if tick == 1 or tick % 5 == 0 or tick == target:
            print(
                f"[{tick:04d}] {world.summary_line()} handoffs={n_h} "
                f"thoughts={len(s.thoughts)} form={s.innerface.formation_ticks}",
                flush=True,
            )
    return 0


def run_gui(
    s: Symbioid,
    world: AudioWorld,
    capture,
    backend: str,
    frames: int,
) -> int:
    try:
        import pygame
    except ImportError:
        print("pygame required for GUI; use --headless", file=sys.stderr)
        return 2

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Symbioid Audio FFT{NUM_BANDS}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("DejaVu Sans", 20)
    font_sm = pygame.font.SysFont("DejaVu Sans", 14)

    tick = 0
    running = True
    err: str | None = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        try:
            world.step(capture)
            tick += 1
            sample_into_symbioid(s, world, tick)
            s.pulse_tick()
        except Exception as e:  # noqa: BLE001 — show on HUD then exit
            err = str(e)
            running = False

        draw_hud(screen, world, s, backend, font, font_sm)
        if err:
            msg = font_sm.render(f"error: {err}", True, (255, 80, 80))
            screen.blit(msg, (16, H // 2))
        pygame.display.flip()
        clock.tick(FPS)

        if frames > 0 and tick >= frames:
            running = False

    pygame.quit()
    return 1 if err else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        print(f"default device: {DEFAULT_ALSA_DEVICE}")
        print(probe_arecord_devices())
        return 0

    set_console_emit(bool(args.verbose))

    if args.backend:
        backend_req = args.backend
    elif args.mic:
        backend_req = "arecord"
    elif args.tone is not None:
        backend_req = "synthetic"
    else:
        backend_req = "synthetic"

    world = AudioWorld(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, num_bands=NUM_BANDS)
    try:
        capture, backend = open_capture(
            backend=backend_req,
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
            device=args.device,
            synthetic_freq=args.tone,
            synthetic_mode="tone" if args.tone is not None else "burst",
        )
    except Exception as e:
        print(f"capture open failed: {e}", file=sys.stderr)
        if backend_req == "arecord":
            print("Hint: try --list-devices or --device plughw:1,0", file=sys.stderr)
        return 2

    world.backend = backend
    s = build_symbioid(world)

    if not args.no_memory:
        if args.reset_memory and args.memory.exists():
            args.memory.unlink()
        try_load_into(s, args.memory)

    # Face workers drain formation inbox (same as Pong/Tetris)
    s.interface.tick_interval = 0.02
    s.innerface.tick_interval = 0.02
    s.outerface.tick_interval = 0.02
    s.start_processes()

    t0 = time.perf_counter()
    code = 1
    try:
        if args.headless:
            code = run_headless(s, world, capture, backend, args.frames)
        else:
            code = run_gui(s, world, capture, backend, args.frames)
    finally:
        s.stop_processes()
        capture.close()
        if not args.no_memory:
            try:
                save_memory(s, args.memory)
            except Exception as e:  # noqa: BLE001
                print(f"memory save failed: {e}", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(
        f"[audio] done blocks={world.blocks} thoughts={len(s.thoughts)} "
        f"formations={s.innerface.formation_ticks} elapsed={elapsed:.2f}s code={code}",
        flush=True,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
