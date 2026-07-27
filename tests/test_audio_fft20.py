"""Phase 0–1 audio: offline FFT20 + sense path (no hardware required)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from symbioid import Sensor, Symbioid
from symbioid.world.audio import (
    CHUNK_SIZE,
    F_MAX,
    F_MIN,
    NUM_BANDS,
    SAMPLE_RATE,
    AudioWorld,
    FFT20Bands,
    SyntheticCapture,
    band_index_for_freq,
    block_duration_s,
    open_capture,
)


def _sine(freq: float, *, n: int = CHUNK_SIZE, sr: int = SAMPLE_RATE, gain: float = 0.5):
    t = np.arange(n, dtype=np.float64) / sr
    return (gain * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def test_block_duration():
    assert abs(block_duration_s() - 2048 / 48000) < 1e-9


def test_log_edges_count_and_span():
    fft = FFT20Bands()
    assert len(fft.band_edges) == NUM_BANDS + 1
    assert abs(fft.band_edges[0] - F_MIN) < 1e-6
    assert abs(fft.band_edges[-1] - F_MAX) < 1e-3
    # strictly increasing
    for i in range(len(fft.band_edges) - 1):
        assert fft.band_edges[i] < fft.band_edges[i + 1]


def test_band_index_for_known_freqs():
    # mid of first band-ish
    assert band_index_for_freq(100.0) == 0 or band_index_for_freq(100.0) < 3
    # 1 kHz lands mid-low
    i1k = band_index_for_freq(1000.0)
    assert 4 <= i1k <= 12
    # 8 kHz near top
    i8k = band_index_for_freq(8000.0)
    assert i8k > i1k
    assert i8k < NUM_BANDS


def test_fft_tone_peaks_correct_band():
    fft = FFT20Bands()
    for freq in (250.0, 1000.0, 4000.0):
        mags = fft.analyze_raw(_sine(freq, gain=0.8))
        unit = fft.normalize(mags)
        peak_i = int(np.argmax(mags))
        expected = band_index_for_freq(freq)
        # allow ±1 band (window leakage / edge straddling)
        assert abs(peak_i - expected) <= 1, (
            f"freq={freq} peak_i={peak_i} expected~{expected} mags={mags}"
        )
        assert float(unit[peak_i]) > 0.05
        assert float(unit[peak_i]) <= 1.0 + 1e-6


def test_silence_near_zero():
    fft = FFT20Bands()
    unit = fft.analyze(np.zeros(CHUNK_SIZE, dtype=np.float32))
    assert float(np.max(unit)) < 0.02


def test_synthetic_capture_and_world():
    cap, name = open_capture(backend="synthetic", synthetic_mode="burst")
    assert name == "synthetic"
    world = AudioWorld()
    for _ in range(5):
        bands = world.step(cap)
        assert len(bands) == NUM_BANDS
        assert all(0.0 <= b <= 1.0 for b in bands)
    assert world.blocks == 5
    assert world.last_rms >= 0.0
    w = world.sensor_world()
    assert "band_00" in w and "band_19" in w and "rms" in w
    cap.close()


def test_tone_mode_emphasizes_one_band():
    cap, _ = open_capture(backend="synthetic", synthetic_freq=1000.0)
    world = AudioWorld()
    # warm up a couple of blocks
    for _ in range(3):
        world.step(cap)
    peak_i = int(np.argmax(world.bands))
    expected = band_index_for_freq(1000.0)
    assert abs(peak_i - expected) <= 1
    cap.close()


def test_symbioid_samples_twenty_band_sensors():
    """Phase 1 exit: stable band Observations via Interface/Innerface."""
    world = AudioWorld()
    # inject a known tone without capture
    world.step_pcm(_sine(500.0, gain=0.6))

    host = Symbioid(id="sym-test-audio", label="audio-test")
    host.interface.continuous_inputs = False
    host.outerface.wait_for_feedback = False
    for i in range(NUM_BANDS):
        lab = f"band_{i:02d}"
        sen = host.add_sensor(
            Sensor(id=f"sym-test-audio:sen:{lab}", label=lab),
            awareness=False,
        )
        sen.transfer = lambda w, key=lab: float(w.get(key, 0.0))

    assert len(host.sensors) == NUM_BANDS
    w = world.sensor_world()
    handoffs = []
    for sen in host.sensors:
        sense = sen.sample(tick=1, world=w)
        assert sense is not None
        assert 0.0 <= sense["reading"] <= 1.0
        h = host.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h is not None:
            handoffs.append(h)
    assert len(handoffs) == NUM_BANDS
    # Sync path (tests); demos use start_processes + post
    host.innerface.accept_formation_batch(
        {"kind": "formation_batch", "handoffs": handoffs, "tick": 1}
    )
    host.pulse_tick()
    # At least one completed formation and Observation-like content
    assert host.innerface.formation_ticks >= 1
    assert len(host.thoughts) > 20  # seeds + formations scaffolding

    # Second sample with same tone → mostly reuse/skip path should not explode
    n_before = len(host.thoughts)
    world.step_pcm(_sine(500.0, gain=0.6))
    w2 = world.sensor_world()
    handoffs2 = []
    for sen in host.sensors:
        sense = sen.sample(tick=2, world=w2)
        h = host.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h is not None:
            handoffs2.append(h)
    if handoffs2:
        host.innerface.accept_formation_batch(
            {"kind": "formation_batch", "handoffs": handoffs2, "tick": 2}
        )
    host.pulse_tick()
    # growth should be modest (reuse), not 20× first-batch scaffolding again
    growth = len(host.thoughts) - n_before
    assert growth < 200, f"unexpected thought growth on reuse: {growth}"


def test_sensor_world_keys_stable():
    world = AudioWorld()
    world.bands = [float(i) / NUM_BANDS for i in range(NUM_BANDS)]
    w = world.sensor_world()
    for i in range(NUM_BANDS):
        assert w[f"band_{i:02d}"] == pytest.approx(i / NUM_BANDS)


def test_synthetic_silence_mode():
    cap = SyntheticCapture(mode="silence")
    chunk = cap.read()
    assert chunk.shape == (CHUNK_SIZE,)
    assert float(np.max(np.abs(chunk))) < 1e-6
    cap.close()
