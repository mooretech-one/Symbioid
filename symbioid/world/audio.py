"""Audio world for Symbioid demo — capture, FFT20 bands, sense world.

Phase 0–1: mic / synthetic PCM → 20 log-spaced magnitude bands [0, 1].
Phase 2+ (later): actuators → synthesis → speakers.

Defaults match the tower plan: 48 kHz, mono, 2048-sample blocks (~42.7 ms),
bands 80 Hz … 12 kHz. Mic default ALSA ``plughw:1,0`` (Logitech C925e —
``hw:1,0`` often rejects mono channel count); override with env
``SYMBIOID_AUDIO_ALSA_DEVICE`` or ``GROK_AUDIO_ALSA_DEVICE``.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48_000
CHUNK_SIZE = 2048  # ~42.67 ms @ 48 kHz
NUM_BANDS = 20
F_MIN = 80.0
F_MAX = 12_000.0

DEFAULT_ALSA_DEVICE = os.environ.get(
    "SYMBIOID_AUDIO_ALSA_DEVICE",
    os.environ.get("GROK_AUDIO_ALSA_DEVICE", "plughw:1,0"),
)


def block_duration_s(
    sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE
) -> float:
    return float(chunk_size) / float(sample_rate)


# ---------------------------------------------------------------------------
# FFT → 20 log bands (AtomIc-style; local copy, no atomic import)
# ---------------------------------------------------------------------------


class FFT20Bands:
    """Windowed rFFT → ``num_bands`` log-spaced mean-magnitude bands."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        fft_size: int = CHUNK_SIZE,
        num_bands: int = NUM_BANDS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
    ) -> None:
        if num_bands < 1:
            raise ValueError("num_bands must be >= 1")
        if f_min <= 0 or f_max <= f_min:
            raise ValueError("need 0 < f_min < f_max")
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.num_bands = int(num_bands)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.band_edges = self._log_edges(self.f_min, self.f_max, self.num_bands)
        self.band_centers = [
            (self.band_edges[i] + self.band_edges[i + 1]) / 2.0
            for i in range(self.num_bands)
        ]
        self._buffer = np.zeros(self.fft_size, dtype=np.float32)
        # Soft ceiling for unit-scale PCM: full-scale sine energy in rFFT
        self._norm_ref = math.log1p(float(self.fft_size) * 0.25)

    @staticmethod
    def _log_edges(f_min: float, f_max: float, n: int) -> list[float]:
        edges = np.geomspace(f_min, f_max, n + 1)
        return [float(x) for x in edges]

    def analyze_raw(self, chunk: np.ndarray) -> np.ndarray:
        """Return mean |rFFT| per band (not unit-normalized)."""
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = min(len(x), self.fft_size)
        self._buffer[:] = 0.0
        if n > 0:
            self._buffer[-n:] = x[-n:]
        windowed = self._buffer * self.window
        spectrum = np.fft.rfft(windowed)
        magnitudes = np.abs(spectrum)
        freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        band_mags = np.zeros(self.num_bands, dtype=np.float32)
        for i in range(self.num_bands):
            lo, hi = self.band_edges[i], self.band_edges[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                band_mags[i] = float(np.mean(magnitudes[mask]))
        return band_mags

    def normalize(self, band_mags: np.ndarray) -> np.ndarray:
        """Map raw band magnitudes → [0, 1] with log compression."""
        m = np.asarray(band_mags, dtype=np.float32)
        # log1p compress; divide by full-scale-ish ref
        out = np.log1p(m) / max(self._norm_ref, 1e-9)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def analyze(self, chunk: np.ndarray) -> np.ndarray:
        """Return unit-normalized band levels in [0, 1]."""
        return self.normalize(self.analyze_raw(chunk))


# ---------------------------------------------------------------------------
# Capture backends
# ---------------------------------------------------------------------------


class AudioCapture(Protocol):
    sample_rate: int
    chunk_size: int

    def read(self) -> np.ndarray: ...

    def close(self) -> None: ...


class SyntheticCapture:
    """Deterministic / burst tones for offline Phase 0–1 (no hardware)."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE,
        freqs: Optional[list[float]] = None,
        gain: float = 0.5,
        mode: str = "burst",
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.freqs = freqs or [
            120.0,
            250.0,
            500.0,
            1000.0,
            2000.0,
            4000.0,
            8000.0,
        ]
        self.gain = float(gain)
        self.mode = mode
        self._phase = 0
        self._chunk_idx = 0

    def read(self) -> np.ndarray:
        t = (np.arange(self.chunk_size, dtype=np.float64) + self._phase) / self.sample_rate
        self._phase += self.chunk_size
        self._chunk_idx += 1

        if self.mode == "silence":
            return np.zeros(self.chunk_size, dtype=np.float32)

        if self.mode == "tone":
            freq = self.freqs[0]
            env = 1.0
        else:
            # burst: hold each tone a few chunks, then near-silence
            freq = self.freqs[(self._chunk_idx // 3) % len(self.freqs)]
            env = 1.0 if (self._chunk_idx % 6) < 4 else 0.02

        signal = env * self.gain * np.sin(2.0 * math.pi * freq * t)
        noise = 0.002 * np.random.randn(self.chunk_size)
        return (signal + noise).astype(np.float32)

    def close(self) -> None:
        pass


class ArecordCapture:
    """Live mic via ALSA ``arecord`` (S16_LE mono)."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE,
        device: Optional[str] = None,
    ) -> None:
        if shutil.which("arecord") is None:
            raise RuntimeError("arecord not found; install alsa-utils")

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device if device is not None else DEFAULT_ALSA_DEVICE
        self._bytes_per_chunk = chunk_size * 2  # S16_LE mono

        cmd = [
            "arecord",
            "-q",
            "-f",
            "S16_LE",
            "-r",
            str(sample_rate),
            "-c",
            "1",
            "-t",
            "raw",
        ]
        if self.device:
            cmd.extend(["-D", self.device])

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self._proc.stdout is None:
            raise RuntimeError("arecord stdout unavailable")

    def read(self) -> np.ndarray:
        assert self._proc.stdout is not None
        raw = self._proc.stdout.read(self._bytes_per_chunk)
        if len(raw) < self._bytes_per_chunk:
            err = ""
            if self._proc.stderr:
                err = self._proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"arecord ended early ({len(raw)} bytes) device={self.device!r}: {err}"
            )
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return samples / 32768.0

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def open_capture(
    *,
    backend: str = "synthetic",
    sample_rate: int = SAMPLE_RATE,
    chunk_size: int = CHUNK_SIZE,
    device: Optional[str] = None,
    synthetic_mode: str = "burst",
    synthetic_freq: Optional[float] = None,
) -> tuple[AudioCapture, str]:
    """
    Open a capture source.

    ``backend``: ``synthetic`` | ``arecord`` | ``auto``
      - auto → arecord if available, else synthetic
    """
    b = (backend or "synthetic").lower().strip()
    if b == "auto":
        if shutil.which("arecord"):
            b = "arecord"
        else:
            b = "synthetic"

    if b == "synthetic":
        freqs = [float(synthetic_freq)] if synthetic_freq else None
        mode = synthetic_mode if synthetic_mode != "tone" or freqs else "burst"
        if synthetic_freq is not None:
            mode = "tone"
        return (
            SyntheticCapture(
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                freqs=freqs,
                mode=mode,
            ),
            "synthetic",
        )

    if b in ("arecord", "mic", "live"):
        return (
            ArecordCapture(
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                device=device,
            ),
            "arecord",
        )

    raise ValueError(f"unknown capture backend: {backend!r}")


def probe_arecord_devices() -> str:
    """Return ``arecord -l`` text (or error string)."""
    if shutil.which("arecord") is None:
        return "arecord not found"
    try:
        r = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001 — probe helper
        return f"arecord -l failed: {e}"


# ---------------------------------------------------------------------------
# AudioWorld — band levels for Sensor.transfer
# ---------------------------------------------------------------------------


@dataclass
class AudioWorld:
    """
    Sense-side audio world: latest 20 band levels + capture stats.

    ``sensor_world()`` returns ``band_00`` … ``band_19`` in [0, 1] for
    Sensor.transfer closures (Pong/Tetris pattern).
    """

    sample_rate: int = SAMPLE_RATE
    chunk_size: int = CHUNK_SIZE
    num_bands: int = NUM_BANDS
    fft: FFT20Bands = field(init=False)
    bands: list[float] = field(init=False)
    last_rms: float = 0.0
    last_peak: float = 0.0
    last_block_ms: float = 0.0
    blocks: int = 0
    backend: str = "synthetic"

    def __post_init__(self) -> None:
        self.fft = FFT20Bands(
            sample_rate=self.sample_rate,
            fft_size=self.chunk_size,
            num_bands=self.num_bands,
        )
        self.bands = [0.0] * self.num_bands

    def step_pcm(self, chunk: np.ndarray, *, block_ms: Optional[float] = None) -> list[float]:
        """Analyze one PCM block; update bands and stats. Returns unit bands."""
        t0 = time.perf_counter()
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            self.bands = [0.0] * self.num_bands
            return list(self.bands)

        self.last_rms = float(np.sqrt(np.mean(np.square(x))))
        self.last_peak = float(np.max(np.abs(x)))
        unit = self.fft.analyze(x)
        self.bands = [float(v) for v in unit]
        self.blocks += 1
        if block_ms is not None:
            self.last_block_ms = float(block_ms)
        else:
            self.last_block_ms = (time.perf_counter() - t0) * 1000.0
        return list(self.bands)

    def step(self, capture: AudioCapture) -> list[float]:
        """Read one block from capture, analyze, return unit bands."""
        t0 = time.perf_counter()
        chunk = capture.read()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        bands = self.step_pcm(chunk, block_ms=wall_ms)
        return bands

    def sensor_world(self) -> dict[str, float]:
        """Map for Sensor.transfer: band_00 … band_{n-1}."""
        w: dict[str, float] = {}
        for i, v in enumerate(self.bands):
            w[f"band_{i:02d}"] = float(v)
        w["rms"] = float(self.last_rms)
        w["peak"] = float(self.last_peak)
        return w

    def band_label(self, i: int) -> str:
        return f"band_{i:02d}"

    def summary_line(self) -> str:
        bars = "".join("█" if v > 0.35 else ("░" if v > 0.08 else "·") for v in self.bands)
        return (
            f"blk={self.blocks} rms={self.last_rms:.4f} peak={self.last_peak:.4f} "
            f"cap_ms={self.last_block_ms:.1f} |{bars}|"
        )


def band_index_for_freq(
    freq_hz: float,
    *,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
    num_bands: int = NUM_BANDS,
) -> int:
    """Which log band index a pure tone should land in (for tests)."""
    if freq_hz < f_min:
        return 0
    if freq_hz >= f_max:
        return num_bands - 1
    # invert geomspace: edges = geomspace(f_min, f_max, n+1)
    # index = floor( n * log(f/f_min) / log(f_max/f_min) )
    ratio = math.log(freq_hz / f_min) / math.log(f_max / f_min)
    idx = int(math.floor(ratio * num_bands))
    return max(0, min(num_bands - 1, idx))
