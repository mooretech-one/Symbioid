#!/usr/bin/env python3
"""
Symbioid audio demo — Phases 0–5: mic/synth → FFT20 → network → babble → speakers.

  PYTHONPATH=. .venv/bin/python audio_demo.py                      # GUI sense+babble closed
  PYTHONPATH=. .venv/bin/python audio_demo.py --sense-only         # Phase 1
  PYTHONPATH=. .venv/bin/python audio_demo.py --babble --no-play   # Phase 2 silent motor
  PYTHONPATH=. .venv/bin/python audio_demo.py --closed --no-play   # Phase 3 digital self-mix
  PYTHONPATH=. .venv/bin/python audio_demo.py --contingent --headless --frames 40 --no-play
  PYTHONPATH=. .venv/bin/python audio_demo.py --mic --play         # live hear + speakers
  PYTHONPATH=. .venv/bin/python audio_demo.py --list-devices

Env: SYMBIOID_AUDIO_ALSA_DEVICE (capture, default plughw:1,0),
     SYMBIOID_AUDIO_PLAY_DEVICE (playback, default default).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from symbioid import (
    Actuator,
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
    DEFAULT_PLAY_DEVICE,
    NUM_BANDS,
    SAMPLE_RATE,
    AudioWorld,
    BabbleCoach,
    open_capture,
    open_playback,
    probe_arecord_devices,
)

HOST_ID = "sym-audio-fft20"
DEFAULT_MEMORY = default_memory_path("audio_memory.json")

W, H = 720, 480
FPS = 24


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Symbioid audio demo Phases 0–5 (FFT20 + babble)"
    )
    p.add_argument("--mic", action="store_true", help="Live ALSA capture")
    p.add_argument("--device", type=str, default=None, help="arecord device")
    p.add_argument(
        "--backend",
        choices=("synthetic", "arecord", "auto"),
        default=None,
    )
    p.add_argument("--tone", type=float, default=None, metavar="HZ")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--reset-memory", action="store_true")

    # Motor / loop phases
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--sense-only",
        action="store_true",
        help="Phase 1: no actuators / no synth (default if no motor flags)",
    )
    mode.add_argument(
        "--babble",
        action="store_true",
        help="Phase 2: open-loop explore coach → synth (no self-mix)",
    )
    mode.add_argument(
        "--closed",
        action="store_true",
        help="Phase 3: digital self-mix + explore coach",
    )
    mode.add_argument(
        "--contingent",
        action="store_true",
        help="Phase 4: self-mix + contingent IM valence",
    )
    mode.add_argument(
        "--noncontingent",
        action="store_true",
        help="Phase 4 control: self-mix + non-contingent reward",
    )

    p.add_argument(
        "--play",
        action="store_true",
        help="Stream synth to speakers (aplay)",
    )
    p.add_argument(
        "--no-play",
        action="store_true",
        help="Force silent playback (null)",
    )
    p.add_argument(
        "--play-device",
        type=str,
        default=None,
        help=f"aplay device (default {DEFAULT_PLAY_DEVICE})",
    )
    p.add_argument(
        "--self-mix",
        type=float,
        default=None,
        help="Override digital self-mix gain [0,1]",
    )
    p.add_argument("--seed", type=int, default=None, help="Coach RNG seed")
    p.add_argument(
        "--mic-gain",
        type=float,
        default=None,
        help="Mic gain before mix (default 1; closed modes often keep 1 or 0 for pure self)",
    )
    p.add_argument(
        "--duck",
        action="store_true",
        help="Enable acoustic howl ducking (auto-on with --mic --play)",
    )
    p.add_argument(
        "--no-duck",
        action="store_true",
        help="Disable ducking even for live mic+play",
    )
    p.add_argument(
        "--leakage",
        type=float,
        default=None,
        metavar="G",
        help="Leakage cancel gain for duck path (default 0.35 when ducking)",
    )
    return p.parse_args(argv)


def resolve_loop_mode(args: argparse.Namespace) -> str:
    if args.sense_only:
        return "sense"
    if args.babble:
        return "babble"
    if args.closed:
        return "closed"
    if args.contingent:
        return "contingent"
    if args.noncontingent:
        return "noncontingent"
    # Default interactive: closed babble (full demo path)
    return "closed"


def build_symbioid(world: AudioWorld, *, with_actuators: bool) -> Symbioid:
    s = Symbioid(id=HOST_ID, label="audio-fft20")
    s.interface.continuous_inputs = False
    s.outerface.wait_for_feedback = False

    for i in range(world.num_bands):
        lab = f"band_{i:02d}"
        sen = s.add_sensor(
            Sensor(id=f"{HOST_ID}:sen:{lab}", label=lab),
            awareness=False,
        )
        sen.transfer = lambda w, key=lab: float(w.get(key, 0.0))

    if with_actuators:
        for i in range(world.num_bands):
            lab = f"act_{i:02d}"
            act = s.add_actuator(
                Actuator(id=f"{HOST_ID}:act:{lab}", label=lab),
                awareness=False,
            )
            act.output = 0.0
            act.output_step = 0.05

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
    loop_mode: str,
    coach: BabbleCoach | None,
    play_name: str,
    font,
    font_sm,
) -> None:
    import pygame

    screen.fill((12, 14, 22))
    title = font.render(
        f"Symbioid audio FFT{world.num_bands}  [{backend}/{play_name}]  {loop_mode}",
        True,
        (200, 210, 230),
    )
    screen.blit(title, (16, 10))

    n = world.num_bands
    margin_x = 24
    top = 48
    bar_w = (W - 2 * margin_x) // n
    bar_max_h = 160

    # hear bars (top)
    lab_h = font_sm.render("hear", True, (120, 160, 200))
    screen.blit(lab_h, (margin_x, top - 16))
    for i, v in enumerate(world.bands):
        h = max(1, int(v * bar_max_h))
        x = margin_x + i * bar_w
        y = top + bar_max_h - h
        col = (min(255, 40 + i * 8), min(255, 120 + int(v * 100)), min(255, 200 - i * 4))
        pygame.draw.rect(screen, col, (x + 2, y, bar_w - 4, h))

    # act bars (bottom)
    top2 = top + bar_max_h + 36
    lab_a = font_sm.render("act", True, (200, 160, 100))
    screen.blit(lab_a, (margin_x, top2 - 16))
    for i, v in enumerate(world.act_levels):
        h = max(1, int(v * bar_max_h))
        x = margin_x + i * bar_w
        y = top2 + bar_max_h - h
        col = (min(255, 180 + int(v * 40)), min(255, 90 + i * 4), 60)
        pygame.draw.rect(screen, col, (x + 2, y, bar_w - 4, h))
        if i % 2 == 0:
            lab = font_sm.render(f"{i:02d}", True, (90, 100, 120))
            screen.blit(lab, (x + 2, top2 + bar_max_h + 2))

    coach_s = coach.summary() if coach else "coach=off"
    stats = font_sm.render(
        f"blocks={world.blocks} rms={world.last_rms:.3f} synth={world.last_synth_rms:.3f} "
        f"err={world.last_pred_err:.3f} thoughts={len(s.thoughts)} form={s.innerface.formation_ticks}",
        True,
        (160, 170, 190),
    )
    screen.blit(stats, (16, H - 52))
    screen.blit(font_sm.render(coach_s, True, (140, 180, 140)), (16, H - 34))
    screen.blit(
        font_sm.render("Esc quit · --sense-only · --babble · --closed · --contingent", True, (90, 100, 120)),
        (16, H - 18),
    )


def one_block(
    s: Symbioid,
    world: AudioWorld,
    capture,
    playback,
    coach: BabbleCoach | None,
    *,
    motor: bool,
    tick: int,
) -> int:
    if coach is not None and motor:
        coach.decide(world)
        coach.apply_to_host(s)
    world.step(capture, playback=playback if motor else None, render_motor=motor)
    n_h = sample_into_symbioid(s, world, tick)
    if coach is not None and motor:
        coach.reinforce(s, world)
    s.pulse_tick()
    return n_h


def run_headless(
    s: Symbioid,
    world: AudioWorld,
    capture,
    playback,
    backend: str,
    play_name: str,
    loop_mode: str,
    coach: BabbleCoach | None,
    frames: int,
    motor: bool,
) -> int:
    target = frames if frames > 0 else 60
    print(
        f"[audio] headless mode={loop_mode} cap={backend} play={play_name} "
        f"self_mix={world.self_mix} duck={world.duck.enabled} blocks={target}",
        flush=True,
    )
    for tick in range(1, target + 1):
        n_h = one_block(s, world, capture, playback, coach, motor=motor, tick=tick)
        time.sleep(0.02)
        if tick == 1 or tick % 5 == 0 or tick == target:
            extra = f" {coach.summary()}" if coach else ""
            print(
                f"[{tick:04d}] {world.summary_line()} handoffs={n_h} "
                f"thoughts={len(s.thoughts)}{extra}",
                flush=True,
            )
    return 0


def run_gui(
    s: Symbioid,
    world: AudioWorld,
    capture,
    playback,
    backend: str,
    play_name: str,
    loop_mode: str,
    coach: BabbleCoach | None,
    frames: int,
    motor: bool,
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
    font = pygame.font.SysFont("DejaVu Sans", 18)
    font_sm = pygame.font.SysFont("DejaVu Sans", 13)

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
            tick += 1
            one_block(s, world, capture, playback, coach, motor=motor, tick=tick)
        except Exception as e:  # noqa: BLE001
            err = str(e)
            running = False

        draw_hud(
            screen, world, s, backend, loop_mode, coach, play_name, font, font_sm
        )
        if err:
            screen.blit(
                font_sm.render(f"error: {err}", True, (255, 80, 80)),
                (16, H // 2),
            )
        pygame.display.flip()
        clock.tick(FPS)
        if frames > 0 and tick >= frames:
            running = False

    pygame.quit()
    return 1 if err else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        print(f"capture default: {DEFAULT_ALSA_DEVICE}")
        print(f"play default:    {DEFAULT_PLAY_DEVICE}")
        print(probe_arecord_devices())
        return 0

    set_console_emit(bool(args.verbose))
    loop_mode = resolve_loop_mode(args)
    motor = loop_mode != "sense"

    if args.backend:
        backend_req = args.backend
    elif args.mic:
        backend_req = "arecord"
    elif args.tone is not None:
        backend_req = "synthetic"
    else:
        # closed digital self-mix works with silence capture offline
        if motor and not args.mic:
            backend_req = "synthetic"
        else:
            backend_req = "synthetic"

    world = AudioWorld(
        sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, num_bands=NUM_BANDS
    )

    # self-mix defaults by mode
    if args.self_mix is not None:
        world.self_mix = float(args.self_mix)
    elif loop_mode in ("closed", "contingent", "noncontingent"):
        world.self_mix = 0.75
    else:
        world.self_mix = 0.0

    if args.mic_gain is not None:
        world.mic_gain = float(args.mic_gain)
    elif loop_mode in ("closed", "contingent", "noncontingent") and not args.mic:
        # pure digital self-hearing for offline babble
        world.mic_gain = 0.0

    # Acoustic howl guard: default on for live mic + playback
    want_play = bool(args.play) or (motor and not args.headless and not args.no_play)
    if args.no_duck:
        world.duck.enabled = False
    elif args.duck or (args.mic and want_play):
        world.duck.enabled = True
    if args.leakage is not None:
        world.duck.leakage_cancel = float(args.leakage)

    synth_mode = "tone" if args.tone is not None else "burst"
    if motor and not args.mic and args.tone is None:
        synth_mode = "silence"

    try:
        capture, backend = open_capture(
            backend=backend_req,
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
            device=args.device,
            synthetic_freq=args.tone,
            synthetic_mode=synth_mode,
        )
    except Exception as e:
        print(f"capture open failed: {e}", file=sys.stderr)
        return 2

    world.backend = backend

    # playback
    if args.no_play or args.headless and not args.play:
        play_backend = "null"
    elif args.play:
        play_backend = "aplay"
    elif motor and not args.headless:
        play_backend = "aplay"  # GUI motor → speakers by default
    else:
        play_backend = "null"

    try:
        playback, play_name = open_playback(
            backend=play_backend,
            sample_rate=SAMPLE_RATE,
            device=args.play_device,
        )
    except Exception as e:
        print(f"playback open failed ({e}); falling back to null", file=sys.stderr)
        playback, play_name = open_playback(backend="null")

    s = build_symbioid(world, with_actuators=motor)

    coach: BabbleCoach | None = None
    if motor:
        cmode = {
            "babble": "explore",
            "closed": "explore",
            "contingent": "contingent",
            "noncontingent": "noncontingent",
        }.get(loop_mode, "explore")
        coach = BabbleCoach(mode=cmode, seed=args.seed)

    if not args.no_memory:
        if args.reset_memory and args.memory.exists():
            args.memory.unlink()
        try_load_into(s, args.memory)

    s.interface.tick_interval = 0.02
    s.innerface.tick_interval = 0.02
    s.outerface.tick_interval = 0.02
    s.start_processes()

    t0 = time.perf_counter()
    code = 1
    try:
        if args.headless:
            code = run_headless(
                s,
                world,
                capture,
                playback,
                backend,
                play_name,
                loop_mode,
                coach,
                args.frames,
                motor,
            )
        else:
            code = run_gui(
                s,
                world,
                capture,
                playback,
                backend,
                play_name,
                loop_mode,
                coach,
                args.frames,
                motor,
            )
    finally:
        s.stop_processes()
        capture.close()
        playback.close()
        if not args.no_memory:
            try:
                save_memory(s, args.memory)
            except Exception as e:  # noqa: BLE001
                print(f"memory save failed: {e}", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    coach_line = f" {coach.summary()}" if coach else ""
    print(
        f"[audio] done mode={loop_mode} blocks={world.blocks} thoughts={len(s.thoughts)} "
        f"formations={s.innerface.formation_ticks} elapsed={elapsed:.2f}s "
        f"code={code}{coach_line}",
        flush=True,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
