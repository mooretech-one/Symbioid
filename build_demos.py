#!/usr/bin/env python3
"""
Build single-file executables for Pong and Tetris demos.

Produces (under dist/):
  SymbioidPong[.exe]
  SymbioidTetris[.exe]
  plus tagged copies: SymbioidPong-linux-x64, SymbioidTetris-macos-arm64, …

PyInstaller embeds the interpreter + deps for *this* OS/arch only.
Build on each target platform (or use GitHub Actions workflow).

Usage:
  .venv/bin/pip install -r requirements-build.txt
  python build_demos.py              # both demos (onefile + console)
  python build_demos.py pong
  python build_demos.py tetris --dir # onedir (faster start)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build" / "pyinstaller"

DEMOS = {
    "pong": {
        "script": ROOT / "pong_demo.py",
        "name": "SymbioidPong",
    },
    "tetris": {
        "script": ROOT / "tetris_demo.py",
        "name": "SymbioidTetris",
    },
}


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        system = "macos"
    if machine in ("amd64", "x86_64"):
        machine = "x64"
    elif machine in ("aarch64", "arm64"):
        machine = "arm64"
    return f"{system}-{machine}"


def _check_deps() -> None:
    missing = []
    try:
        import pygame  # noqa: F401
    except ImportError:
        missing.append("pygame")
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        missing.append("pyinstaller")
    if missing:
        print(
            "Missing: "
            + ", ".join(missing)
            + "\nInstall with:\n  pip install -r requirements-build.txt",
            file=sys.stderr,
        )
        sys.exit(1)


def build_one(demo_key: str, *, onefile: bool, console: bool, clean: bool) -> Path:
    demo = DEMOS[demo_key]
    script = demo["script"]
    name = demo["name"]
    if not script.is_file():
        raise FileNotFoundError(script)

    DIST.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(script),
        f"--name={name}",
        f"--distpath={DIST}",
        f"--workpath={BUILD / demo_key}",
        f"--specpath={BUILD / 'specs'}",
        "--noconfirm",
        f"--paths={ROOT}",
        # pygame + SDL binaries (avoid shipping tests/examples — huge + unused)
        "--collect-binaries=pygame",
        "--collect-data=pygame",
        "--exclude-module=pygame.tests",
        "--exclude-module=pygame.examples",
        "--exclude-module=tkinter",
        "--exclude-module=unittest",
        "--hidden-import=symbioid",
        "--hidden-import=symbioid.world",
        "--hidden-import=symbioid.world.pong",
        "--hidden-import=symbioid.world.paddle_learn",
        "--hidden-import=symbioid.world.tetris",
        "--hidden-import=symbioid.world.tetris_learn",
        "--hidden-import=symbioid.Core",
        "--hidden-import=symbioid.Core.Symbioid",
        "--hidden-import=symbioid.core",
    ]
    if onefile:
        args.append("--onefile")
    else:
        args.append("--onedir")
    if console:
        args.append("--console")
    else:
        args.append("--windowed")
    if clean:
        args.append("--clean")

    print(f"\n=== Building {name} ({demo_key}) ===", flush=True)
    print(" ", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=ROOT)

    suffix = ".exe" if platform.system() == "Windows" else ""
    if onefile:
        out = DIST / f"{name}{suffix}"
    else:
        out = DIST / name / f"{name}{suffix}"
    if not out.exists():
        app = DIST / f"{name}.app"
        if app.exists():
            out = app
        else:
            raise FileNotFoundError(f"Expected output missing: {out}")

    tag = _platform_tag()
    if out.is_file():
        tagged = DIST / f"{name}-{tag}{suffix}"
        shutil.copy2(out, tagged)
        print(f"OK  {out}")
        print(f"    (+ tagged copy {tagged.name})")
        return out
    print(f"OK  {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Symbioid Pong/Tetris single-file executables"
    )
    parser.add_argument(
        "demos",
        nargs="*",
        choices=sorted(DEMOS.keys()),
        help="Demos to build (default: both pong and tetris)",
    )
    parser.add_argument(
        "--dir",
        action="store_true",
        help="onedir build instead of onefile (faster startup)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="GUI only, no console (default keeps console for learning logs)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip PyInstaller --clean",
    )
    args = parser.parse_args()
    targets = list(args.demos) if args.demos else list(DEMOS.keys())

    _check_deps()
    print(f"Platform: {_platform_tag()}")
    print(f"Python:   {sys.version.split()[0]}  ({sys.executable})")

    outputs: list[Path] = []
    for key in targets:
        outputs.append(
            build_one(
                key,
                onefile=not args.dir,
                console=not args.windowed,
                clean=not args.no_clean,
            )
        )

    print("\n=== Done ===")
    for p in outputs:
        print(f"  {p}")
    print(f"\nRun example:\n  {outputs[0]}")


if __name__ == "__main__":
    main()
