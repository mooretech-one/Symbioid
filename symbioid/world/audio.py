"""Audio world for Symbioid demo — capture, FFT20, synth, babble coach.

Phase 0–1: mic / synthetic PCM → 20 log-spaced magnitude bands [0, 1].
Phase 2:   20 actuators → band oscillators → speakers (open-loop babble).
Phase 3:   digital self-mix so production changes hearing.
Phase 4:   contingent vs non-contingent intrinsic-motivation valence.

Defaults: 48 kHz mono, 2048-sample blocks (~42.7 ms), bands 80 Hz … 12 kHz.
Mic default ALSA ``plughw:1,0`` (Logitech C925e). Playback default system
``default`` via aplay. Override with ``SYMBIOID_AUDIO_ALSA_DEVICE`` /
``GROK_AUDIO_ALSA_DEVICE`` (capture) and ``SYMBIOID_AUDIO_PLAY_DEVICE``.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

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
DEFAULT_PLAY_DEVICE = os.environ.get("SYMBIOID_AUDIO_PLAY_DEVICE", "default")


def block_duration_s(
    sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE
) -> float:
    return float(chunk_size) / float(sample_rate)


# ---------------------------------------------------------------------------
# FFT → 20 log bands
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
        self._bytes_per_chunk = chunk_size * 2

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
    """Open a capture source. ``backend``: synthetic | arecord | auto."""
    b = (backend or "synthetic").lower().strip()
    if b == "auto":
        b = "arecord" if shutil.which("arecord") else "synthetic"

    if b == "synthetic":
        freqs = [float(synthetic_freq)] if synthetic_freq else None
        mode = "tone" if synthetic_freq is not None else synthetic_mode
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
    if shutil.which("arecord") is None:
        return "arecord not found"
    try:
        r = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, timeout=5
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"arecord -l failed: {e}"


# ---------------------------------------------------------------------------
# Phase 2 — band synthesis + playback
# ---------------------------------------------------------------------------


class BandSynth:
    """Sum of band-center oscillators controlled by 20 gains in [0, 1]."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE,
        band_centers: Optional[list[float]] = None,
        master_gain: float = 0.22,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        if band_centers is None:
            edges = np.geomspace(F_MIN, F_MAX, NUM_BANDS + 1)
            band_centers = [
                float((edges[i] + edges[i + 1]) / 2.0) for i in range(NUM_BANDS)
            ]
        self.band_centers = list(band_centers)
        self.num_bands = len(self.band_centers)
        self.master_gain = float(master_gain)
        self._phase = np.zeros(self.num_bands, dtype=np.float64)
        self._omega = (
            2.0 * math.pi * np.asarray(self.band_centers, dtype=np.float64) / sample_rate
        )

    def render(
        self,
        gains: list[float] | np.ndarray,
        *,
        gain_scale: float = 1.0,
    ) -> np.ndarray:
        g = np.asarray(gains, dtype=np.float64).reshape(-1)
        if len(g) < self.num_bands:
            g = np.pad(g, (0, self.num_bands - len(g)))
        g = np.clip(g[: self.num_bands], 0.0, 1.0)
        n = self.chunk_size
        t = np.arange(n, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for i in range(self.num_bands):
            if g[i] < 1e-6:
                self._phase[i] = (self._phase[i] + self._omega[i] * n) % (2.0 * math.pi)
                continue
            ph = self._phase[i] + self._omega[i] * t
            out += g[i] * np.sin(ph)
            self._phase[i] = float(ph[-1] + self._omega[i]) % (2.0 * math.pi)
        # soft peak limit + external duck scale
        peak = float(np.max(np.abs(out))) if n else 0.0
        scale = self.master_gain * float(max(0.0, min(1.0, gain_scale)))
        if peak * scale > 0.95:
            scale = 0.95 / max(peak, 1e-9)
        return (out * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Acoustic closed-loop howl mitigation
# ---------------------------------------------------------------------------


@dataclass
class AcousticDucker:
    """
    Soft-gain ducking + mic high-pass for live speaker→mic feedback (howl).

    When mic energy is high relative to intended synth (or absolute ceiling),
    lower ``play_gain`` quickly; recover slowly when quiet. Optional spectral
    subtract of last synth from mic before FFT (leakage cancel).
    """

    enabled: bool = False
    play_gain: float = 1.0
    min_gain: float = 0.04
    max_gain: float = 1.0
    attack: float = 0.45  # fraction toward min on howl
    release: float = 0.06  # fraction toward max when clear
    howl_rms: float = 0.22
    howl_ratio: float = 1.6  # mic_rms / max(synth_rms, eps)
    peak_ceil: float = 0.85
    # one-pole high-pass (~80 Hz @ 48k): y[n] = a*(y[n-1] + x[n] - x[n-1])
    highpass: bool = True
    hp_coeff: float = 0.9895  # ~80 Hz @ 48 kHz
    # subtract attenuated last_synth from mic (acoustic echo cancel lite)
    leakage_cancel: float = 0.35
    last_howl_score: float = 0.0
    duck_events: int = 0
    _hp_x1: float = field(default=0.0, init=False, repr=False)
    _hp_y1: float = field(default=0.0, init=False, repr=False)

    def highpass_filter(self, mic: np.ndarray) -> np.ndarray:
        if not self.highpass:
            return np.asarray(mic, dtype=np.float32)
        x = np.asarray(mic, dtype=np.float64).reshape(-1)
        if len(x) == 0:
            return x.astype(np.float32)
        a = float(self.hp_coeff)
        y = np.empty_like(x)
        x1, y1 = self._hp_x1, self._hp_y1
        for i, xi in enumerate(x):
            yi = a * (y1 + xi - x1)
            y[i] = yi
            x1, y1 = xi, yi
        self._hp_x1, self._hp_y1 = float(x1), float(y1)
        return y.astype(np.float32)

    def cancel_leakage(self, mic: np.ndarray, last_synth: Optional[np.ndarray]) -> np.ndarray:
        """mic - leakage_cancel * play_gain * last_synth (when enabled)."""
        x = np.asarray(mic, dtype=np.float32).reshape(-1)
        if (
            not self.enabled
            or self.leakage_cancel <= 0.0
            or last_synth is None
        ):
            return x
        s = np.asarray(last_synth, dtype=np.float32).reshape(-1)
        n = min(len(x), len(s))
        if n == 0:
            return x
        out = x.copy()
        out[:n] = out[:n] - float(self.leakage_cancel) * float(self.play_gain) * s[:n]
        return out

    def update(self, *, mic_rms: float, synth_rms: float, mic_peak: float) -> float:
        """Update play_gain from levels; return current play_gain."""
        if not self.enabled:
            self.play_gain = self.max_gain
            return self.play_gain
        eps = 1e-3
        ratio = float(mic_rms) / max(float(synth_rms), eps)
        howl = 0.0
        if mic_rms >= self.howl_rms and ratio >= self.howl_ratio:
            howl = min(1.0, (ratio / self.howl_ratio - 1.0) + (mic_rms / self.howl_rms - 1.0))
        if mic_peak >= self.peak_ceil:
            howl = max(howl, 0.75)
        self.last_howl_score = float(howl)
        if howl > 0.05:
            # duck toward min_gain
            target = self.min_gain + (self.max_gain - self.min_gain) * max(0.0, 1.0 - howl)
            self.play_gain += self.attack * (target - self.play_gain)
            self.duck_events += 1
        else:
            self.play_gain += self.release * (self.max_gain - self.play_gain)
        self.play_gain = float(max(self.min_gain, min(self.max_gain, self.play_gain)))
        return self.play_gain


class AudioPlayback(Protocol):
    def write(self, pcm: np.ndarray) -> None: ...

    def close(self) -> None: ...


class NullPlayback:
    """Discard PCM (headless / CI)."""

    def __init__(self) -> None:
        self.blocks = 0
        self.last_rms = 0.0

    def write(self, pcm: np.ndarray) -> None:
        x = np.asarray(pcm, dtype=np.float32)
        self.blocks += 1
        if len(x):
            self.last_rms = float(np.sqrt(np.mean(np.square(x))))

    def close(self) -> None:
        pass


class AplayPlayback:
    """Stream S16_LE mono to ALSA via ``aplay``."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        device: Optional[str] = None,
    ) -> None:
        if shutil.which("aplay") is None:
            raise RuntimeError("aplay not found; install alsa-utils")
        self.sample_rate = sample_rate
        self.device = device if device is not None else DEFAULT_PLAY_DEVICE
        cmd = [
            "aplay",
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
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            raise RuntimeError("aplay stdin unavailable")

    def write(self, pcm: np.ndarray) -> None:
        if self._proc.poll() is not None:
            err = ""
            if self._proc.stderr:
                err = self._proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"aplay exited: {err}")
        x = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16)
        assert self._proc.stdin is not None
        self._proc.stdin.write(i16.tobytes())
        self._proc.stdin.flush()

    def close(self) -> None:
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def open_playback(
    *,
    backend: str = "null",
    sample_rate: int = SAMPLE_RATE,
    device: Optional[str] = None,
) -> tuple[AudioPlayback, str]:
    """``backend``: null | aplay | auto (aplay if present else null)."""
    b = (backend or "null").lower().strip()
    if b == "auto":
        b = "aplay" if shutil.which("aplay") else "null"
    if b in ("null", "none", "silent"):
        return NullPlayback(), "null"
    if b == "aplay":
        return AplayPlayback(sample_rate=sample_rate, device=device), "aplay"
    raise ValueError(f"unknown playback backend: {backend!r}")


# ---------------------------------------------------------------------------
# AudioWorld — sense + motor + closed-loop mix
# ---------------------------------------------------------------------------


@dataclass
class AudioWorld:
    """
    Audio world: hear (FFT20) + optional babble synth/playback.

    ``sensor_world()`` → band_00…band_19 for Sensor.transfer.
    ``act_levels`` written by coach / actuators; ``render_and_play`` for Phase 2.
    ``self_mix`` digitally mixes last synth into next hear (Phase 3).
    """

    sample_rate: int = SAMPLE_RATE
    chunk_size: int = CHUNK_SIZE
    num_bands: int = NUM_BANDS
    fft: FFT20Bands = field(init=False)
    synth: BandSynth = field(init=False)
    bands: list[float] = field(init=False)
    act_levels: list[float] = field(init=False)
    last_rms: float = 0.0
    last_peak: float = 0.0
    last_block_ms: float = 0.0
    last_synth_rms: float = 0.0
    last_pred_err: float = 0.0
    blocks: int = 0
    backend: str = "synthetic"
    # Phase 3: mix self-output into heard PCM (digital; offline-safe)
    self_mix: float = 0.0
    mic_gain: float = 1.0
    # Live acoustic howl guard (speaker → mic)
    duck: AcousticDucker = field(default_factory=AcousticDucker)
    last_synth: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    last_mic_rms: float = 0.0
    last_play_gain: float = 1.0

    def __post_init__(self) -> None:
        self.fft = FFT20Bands(
            sample_rate=self.sample_rate,
            fft_size=self.chunk_size,
            num_bands=self.num_bands,
        )
        self.synth = BandSynth(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            band_centers=list(self.fft.band_centers),
        )
        self.bands = [0.0] * self.num_bands
        self.act_levels = [0.0] * self.num_bands
        if not isinstance(self.duck, AcousticDucker):
            self.duck = AcousticDucker()

    def set_act_levels(self, levels: list[float] | np.ndarray) -> None:
        g = [float(max(0.0, min(1.0, x))) for x in levels]
        if len(g) < self.num_bands:
            g = g + [0.0] * (self.num_bands - len(g))
        self.act_levels = g[: self.num_bands]

    def pull_actuators(self, actuators: list[Any]) -> None:
        """Copy Actuator.output → act_levels (demo writes actuators, not request_fire)."""
        levels = []
        for i in range(self.num_bands):
            if i < len(actuators):
                levels.append(float(getattr(actuators[i], "output", 0.0)))
            else:
                levels.append(0.0)
        self.set_act_levels(levels)

    def push_actuators(self, actuators: list[Any]) -> None:
        for i, a in enumerate(actuators):
            if i >= self.num_bands:
                break
            a.output = float(self.act_levels[i])

    def render_synth(self) -> np.ndarray:
        scale = float(self.duck.play_gain) if self.duck.enabled else 1.0
        self.last_play_gain = scale
        pcm = self.synth.render(self.act_levels, gain_scale=scale)
        self.last_synth = pcm
        self.last_synth_rms = float(np.sqrt(np.mean(np.square(pcm)))) if len(pcm) else 0.0
        return pcm

    def play(self, playback: AudioPlayback, pcm: Optional[np.ndarray] = None) -> None:
        if pcm is None:
            pcm = self.last_synth if self.last_synth is not None else self.render_synth()
        playback.write(pcm)

    def step_pcm(self, chunk: np.ndarray, *, block_ms: Optional[float] = None) -> list[float]:
        """Analyze one PCM block (after optional self-mix applied by caller)."""
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
        # prediction error: hear vs current act (closed-loop match signal)
        pred = np.asarray(self.act_levels, dtype=np.float32)
        hear = np.asarray(self.bands, dtype=np.float32)
        self.last_pred_err = float(np.mean(np.square(hear - pred)))
        return list(self.bands)

    def mix_with_self(self, mic: np.ndarray) -> np.ndarray:
        """mic_gain * mic + self_mix * last_synth (digital closed loop)."""
        x = self.mic_gain * np.asarray(mic, dtype=np.float32).reshape(-1)
        if self.self_mix > 0.0 and self.last_synth is not None:
            s = np.asarray(self.last_synth, dtype=np.float32).reshape(-1)
            n = min(len(x), len(s))
            if n > 0:
                y = x.copy()
                y[:n] = y[:n] + float(self.self_mix) * s[:n]
                # soft clip
                peak = float(np.max(np.abs(y))) if n else 0.0
                if peak > 1.0:
                    y = y / peak
                return y
        return x

    def prepare_mic(self, mic: np.ndarray) -> np.ndarray:
        """High-pass + optional leakage cancel for acoustic closed loop."""
        x = self.duck.highpass_filter(mic) if self.duck.enabled else np.asarray(
            mic, dtype=np.float32
        )
        if self.duck.enabled and self.self_mix <= 0.0:
            # acoustic path: cancel predicted speaker leakage before FFT
            x = self.duck.cancel_leakage(x, self.last_synth)
        return x

    def step(
        self,
        capture: AudioCapture,
        *,
        playback: Optional[AudioPlayback] = None,
        render_motor: bool = False,
    ) -> list[float]:
        """
        One block: optional motor render/play → capture → duck update → mix → FFT.

        Order matches developmental loop: produce, then hear (including self).
        """
        if render_motor:
            pcm = self.render_synth()
            if playback is not None:
                self.play(playback, pcm)

        t0 = time.perf_counter()
        mic_raw = capture.read()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        mic = self.prepare_mic(mic_raw)
        mic_f = np.asarray(mic, dtype=np.float32)
        self.last_mic_rms = (
            float(np.sqrt(np.mean(np.square(mic_f)))) if len(mic_f) else 0.0
        )
        mic_peak = float(np.max(np.abs(mic_f))) if len(mic_f) else 0.0
        # Update duck from this capture (affects *next* render_synth play_gain)
        self.duck.update(
            mic_rms=self.last_mic_rms,
            synth_rms=self.last_synth_rms,
            mic_peak=mic_peak,
        )
        self.last_play_gain = float(self.duck.play_gain)

        heard = self.mix_with_self(mic)
        return self.step_pcm(heard, block_ms=wall_ms)

    def sensor_world(self) -> dict[str, float]:
        w: dict[str, float] = {}
        for i, v in enumerate(self.bands):
            w[f"band_{i:02d}"] = float(v)
        for i, v in enumerate(self.act_levels):
            w[f"act_{i:02d}"] = float(v)
        w["rms"] = float(self.last_rms)
        w["peak"] = float(self.last_peak)
        w["synth_rms"] = float(self.last_synth_rms)
        w["pred_err"] = float(self.last_pred_err)
        w["play_gain"] = float(self.last_play_gain)
        w["mic_rms"] = float(self.last_mic_rms)
        w["howl"] = float(self.duck.last_howl_score)
        return w

    def band_label(self, i: int) -> str:
        return f"band_{i:02d}"

    def summary_line(self) -> str:
        hear = "".join("█" if v > 0.35 else ("░" if v > 0.08 else "·") for v in self.bands)
        act = "".join("█" if v > 0.35 else ("░" if v > 0.08 else "·") for v in self.act_levels)
        duck = (
            f" pg={self.last_play_gain:.2f} howl={self.duck.last_howl_score:.2f}"
            if self.duck.enabled
            else ""
        )
        return (
            f"blk={self.blocks} rms={self.last_rms:.4f} synth={self.last_synth_rms:.4f} "
            f"err={self.last_pred_err:.3f} cap_ms={self.last_block_ms:.1f}{duck} "
            f"H|{hear}| A|{act}|"
        )


# ---------------------------------------------------------------------------
# Phase 2–4 — babble coach + contingent IM
# ---------------------------------------------------------------------------


@dataclass
class BabbleCoach:
    """
    Open-loop spectral play + optional contingent valence.

    Writes Actuator.output directly (plan: not primary request_fire).
    Modes:
      explore       — random-walk gains (Phase 2)
      contingent    — reward when hear≈play (self-mix / acoustic) (Phase 4)
      noncontingent — same motor, reward from unrelated noise (control)
    """

    num_bands: int = NUM_BANDS
    mode: str = "explore"  # explore | contingent | noncontingent | freeze
    step_sigma: float = 0.12
    mutate_p: float = 0.08
    seed: Optional[int] = None
    levels: list[float] = field(init=False)
    steps: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    last_err: float = 1.0
    learning_progress: float = 0.0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.levels = [self._rng.random() * 0.4 for _ in range(self.num_bands)]

    def _explore(self) -> None:
        for i in range(self.num_bands):
            if self._rng.random() < self.mutate_p:
                self.levels[i] = self._rng.random()
            else:
                self.levels[i] += self._rng.gauss(0.0, self.step_sigma)
            self.levels[i] = max(0.0, min(1.0, self.levels[i]))
        # keep a few bands quiet to reduce density / howl risk
        if self._rng.random() < 0.3:
            k = self._rng.randrange(self.num_bands)
            self.levels[k] = 0.0

    def decide(self, world: AudioWorld) -> list[float]:
        """Update motor levels for this block; return copy."""
        if self.mode != "freeze":
            self._explore()
        world.set_act_levels(self.levels)
        self.steps += 1
        return list(self.levels)

    def apply_to_host(self, host: Any) -> None:
        acts = getattr(host, "actuators", []) or []
        for i, a in enumerate(acts):
            if i >= self.num_bands:
                break
            a.output = float(self.levels[i])

    def action_token(self) -> str:
        """Coarse action key for Mind (top band + energy bucket)."""
        if not self.levels:
            return "act:silence"
        peak_i = int(max(range(len(self.levels)), key=lambda i: self.levels[i]))
        energy = sum(self.levels) / max(1, len(self.levels))
        bucket = int(min(9, max(0, energy * 10)))
        return f"act:b{peak_i:02d}_e{bucket}"

    def compute_reward(self, world: AudioWorld) -> float:
        """
        Intrinsic reward.

        contingent: learning progress on pred_err (hear vs act) + match bonus
        noncontingent: random reward (same scale) — control
        explore: mild novelty on motor energy only
        """
        err = float(world.last_pred_err)
        prev = self.last_err
        lp = prev - err  # positive when error decreases
        self.learning_progress = lp
        self.last_err = err

        if self.mode == "noncontingent":
            r = self._rng.uniform(-0.3, 0.5)
        elif self.mode == "contingent":
            # match quality: low err → positive; learning progress bonus
            match = max(0.0, 1.0 - err * 4.0)  # err~0.25 → 0
            r = 0.6 * match + 1.2 * max(0.0, lp) - 0.15 * err
        else:
            # explore: small energy novelty (prefer mid energy)
            e = sum(self.levels) / max(1, len(self.levels))
            r = 0.1 * (1.0 - abs(e - 0.35)) + 0.05 * self._rng.random()

        self.last_reward = float(r)
        self.total_reward += self.last_reward
        return self.last_reward

    def reinforce(self, host: Any, world: AudioWorld) -> float:
        """Apply reward via Mind.record_outcome + direct action valence."""
        r = self.compute_reward(world)
        mind = getattr(host, "mind", None)
        if mind is None or not getattr(mind, "enabled", True):
            return r

        host_id = str(getattr(host, "id", "host"))
        token = self.action_token()
        state_thoughts = collect_audio_state_poles(host, world, top_k=6)

        # Phase 5 spectral: imprint / probe band envelope in holonomic store
        spectral_bonus = 0.0
        if getattr(mind, "holonomic_store_enabled", False):
            try:
                import numpy as np

                cap = int(getattr(mind, "holonomic_capacity", 64) or 64)
                vec = np.zeros(cap, dtype=np.float32)
                bands = list(getattr(world, "bands", []) or [])
                n = min(len(bands), cap)
                if n > 0:
                    vec[:n] = np.asarray(bands[:n], dtype=np.float32)
                    store = mind.ensure_holonomic_store()
                    # write envelope every step (strength scales with reward polarity)
                    strength = max(0.15, abs(float(r))) if r != 0 else 0.1
                    store.write(vec, strength=strength)
                    mind.holonomic_writes += 1
                    if self.mode == "contingent":
                        score = float(store.score_probe(vec))
                        spectral_bonus = 0.12 * score
                        if spectral_bonus != 0.0:
                            mind.note_valence(
                                channel="board",
                                delta=spectral_bonus,
                                recent=6,
                            )
                            r = float(r) + spectral_bonus
                            self.last_reward = float(r)
            except Exception:  # noqa: BLE001
                pass

        # Scale reward to Mind's /50 path: pass r*50 so delta ≈ r
        action = mind.record_outcome(
            state_thoughts,
            token,
            domain="audio",
            host_id=host_id,
            reward=float(r) * 50.0,
            channel="audio",
            host=host,
        )
        # Always stamp valence on the Action pole (empty state still learns motor)
        if action is not None:
            mind.note_valence(thought_id=action.id, delta=float(r))
            try:
                ck = mind.action_content_key("audio", token)
                mind.note_valence(content_key=ck, delta=float(r) * 0.5)
            except Exception:  # noqa: BLE001
                pass
        return r

    def summary(self) -> str:
        return (
            f"coach[{self.mode}] steps={self.steps} R={self.total_reward:.2f} "
            f"lastR={self.last_reward:.3f} lp={self.learning_progress:.3f} "
            f"err={self.last_err:.3f}"
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
    ratio = math.log(freq_hz / f_min) / math.log(f_max / f_min)
    idx = int(math.floor(ratio * num_bands))
    return max(0, min(num_bands - 1, idx))


def _ensure_meta_observation(
    mind: Any,
    host: Any,
    *,
    channel: str,
    token: str,
    label: str,
    host_id: str,
) -> Any:
    """Stable meta Observation pole registered in Mind (for record_outcome state)."""
    from symbioid.Core.Thought import Thought
    from symbioid.Core.thought_layers import ThoughtLayer

    ck = f"meta:audio:{channel}:{token}"
    with mind._lock:
        existing = mind._observations.get(ck)
        if existing is not None:
            return existing
        oid = f"{host_id}:obs:meta:{mind._hash_key(ck)}"
        thought = Thought(
            id=oid,
            label=label,
            transient=False,
            layer=ThoughtLayer.PATTERN,
        )
        mind._register(ck, thought, sensor_id=f"meta:{channel}", valence=0.05)
    if host is not None and hasattr(host, "add_thought"):
        if thought.id not in getattr(host, "thoughts", {}):
            host.add_thought(thought)
    return thought


def collect_audio_state_poles(
    host: Any,
    world: AudioWorld,
    *,
    top_k: int = 6,
) -> list[Any]:
    """
    Rich state poles for Mind.record_outcome:

    1. Latest band Observations from top-energy sensors (Innerface last_obs)
    2. Meta poles: peak band bucket, energy bucket, pred_err bucket, howl/duck
    """
    poles: list[Any] = []
    seen: set[str] = set()

    def _add(t: Any) -> None:
        if t is None:
            return
        tid = getattr(t, "id", None)
        if tid is None or tid in seen:
            return
        seen.add(tid)
        poles.append(t)

    # --- (1) top-k band Observations by current energy ---
    sensors = list(getattr(host, "sensors", []) or [])
    by_label = {str(getattr(s, "label", "") or ""): s for s in sensors}
    ranked = sorted(
        range(world.num_bands),
        key=lambda i: float(world.bands[i]) if i < len(world.bands) else 0.0,
        reverse=True,
    )
    inner = getattr(host, "innerface", None)
    last_map = getattr(inner, "_last_obs_by_sensor", {}) if inner is not None else {}
    for i in ranked[: max(1, top_k)]:
        lab = f"band_{i:02d}"
        sen = by_label.get(lab)
        if sen is None:
            continue
        obs = last_map.get(sen.id)
        _add(obs)

    # Fallback: high-activation mind observations
    mind = getattr(host, "mind", None)
    if mind is not None and len(poles) < 2:
        try:
            with mind._lock:
                obs_list = list(mind._observations.values())
            obs_list.sort(
                key=lambda t: float(getattr(t, "activation", 0.0)), reverse=True
            )
            for t in obs_list[:top_k]:
                _add(t)
        except Exception:  # noqa: BLE001
            pass

    # --- (2) meta poles (always available, content-stable) ---
    if mind is not None and getattr(mind, "enabled", True):
        host_id = str(getattr(host, "id", "host"))
        peak_i = (
            int(max(range(len(world.bands)), key=lambda j: world.bands[j]))
            if world.bands
            else 0
        )
        energy = (
            sum(world.bands) / max(1, len(world.bands)) if world.bands else 0.0
        )
        e_bucket = int(min(9, max(0, energy * 10)))
        err_bucket = int(min(9, max(0, world.last_pred_err * 10)))
        howl_bucket = int(min(9, max(0, world.duck.last_howl_score * 10)))
        pg_bucket = int(min(9, max(0, world.last_play_gain * 10)))

        _add(
            _ensure_meta_observation(
                mind,
                host,
                channel="peak",
                token=f"b{peak_i:02d}",
                label=f"hear_peak:{peak_i:02d}",
                host_id=host_id,
            )
        )
        _add(
            _ensure_meta_observation(
                mind,
                host,
                channel="energy",
                token=f"e{e_bucket}",
                label=f"hear_energy:{e_bucket}",
                host_id=host_id,
            )
        )
        _add(
            _ensure_meta_observation(
                mind,
                host,
                channel="err",
                token=f"r{err_bucket}",
                label=f"pred_err:{err_bucket}",
                host_id=host_id,
            )
        )
        if world.duck.enabled:
            _add(
                _ensure_meta_observation(
                    mind,
                    host,
                    channel="howl",
                    token=f"h{howl_bucket}_g{pg_bucket}",
                    label=f"howl:{howl_bucket}/pg:{pg_bucket}",
                    host_id=host_id,
                )
            )
        # act peak meta (motor context)
        if world.act_levels:
            ap = int(max(range(len(world.act_levels)), key=lambda j: world.act_levels[j]))
            _add(
                _ensure_meta_observation(
                    mind,
                    host,
                    channel="act_peak",
                    token=f"a{ap:02d}",
                    label=f"act_peak:{ap:02d}",
                    host_id=host_id,
                )
            )

    return poles


def compare_contingent_vs_noncontingent(
    *,
    blocks: int = 40,
    seed: int = 0,
    spectral: bool = False,
    spectral_primary: bool = False,
) -> dict[str, float]:
    """
    Offline Phase 4 check: same motor seed; contingent should accumulate
    more total reward when self-mix makes hear≈play.

    spectral=True enables Mind spectral mix + holonomic + phase Hebb (Phase 5).
    spectral_primary=True uses Mode B (FFT mix only, no Link spread).
    """
    from symbioid import Actuator, Sensor, Symbioid

    def _run(mode: str) -> float:
        world = AudioWorld()
        world.self_mix = 0.85
        world.mic_gain = 0.0  # pure self-hearing
        cap = SyntheticCapture(mode="silence")
        coach = BabbleCoach(mode=mode, seed=seed)
        host = Symbioid(id=f"cmp-{mode}", label=mode)
        host.interface.continuous_inputs = False
        host.outerface.wait_for_feedback = False
        if spectral or spectral_primary:
            host.mind.enable_spectral_demo(
                phase_hebb=True, primary=bool(spectral_primary)
            )
        else:
            # keep compare stable for legacy tests
            host.mind.set_dynamics_mode("graph")
            host.mind.spectral_mix_enabled = False
            host.mind.holonomic_store_enabled = False
        for i in range(NUM_BANDS):
            host.add_sensor(
                Sensor(id=f"cmp-{mode}:sen:band_{i:02d}", label=f"band_{i:02d}"),
                awareness=False,
            )
            host.add_actuator(
                Actuator(id=f"cmp-{mode}:act:act_{i:02d}", label=f"act_{i:02d}"),
                awareness=False,
            )
        for _ in range(blocks):
            coach.decide(world)
            coach.apply_to_host(host)
            world.step(cap, render_motor=True)
            coach.reinforce(host, world)
            if spectral and getattr(host.mind, "dynamics_enabled", True):
                host.pulse_tick()
        cap.close()
        return float(coach.total_reward)

    c = _run("contingent")
    n = _run("noncontingent")
    return {"contingent": c, "noncontingent": n, "delta": c - n}
