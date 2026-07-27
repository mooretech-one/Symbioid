"""Audio polish: acoustic ducking, rich state poles (no hardware)."""

from __future__ import annotations

import numpy as np

from symbioid import Actuator, Sensor, Symbioid
from symbioid.world.audio import (
    NUM_BANDS,
    AcousticDucker,
    AudioWorld,
    BabbleCoach,
    SyntheticCapture,
    collect_audio_state_poles,
)


def test_ducker_lowers_gain_on_howl():
    d = AcousticDucker(enabled=True, play_gain=1.0, attack=0.5, release=0.01)
    g0 = d.play_gain
    # loud mic vs quiet synth → howl
    for _ in range(5):
        d.update(mic_rms=0.5, synth_rms=0.05, mic_peak=0.9)
    assert d.play_gain < g0
    assert d.duck_events >= 1
    assert d.last_howl_score > 0.0


def test_ducker_recovers_when_quiet():
    d = AcousticDucker(enabled=True, play_gain=0.1, attack=0.5, release=0.2)
    for _ in range(30):
        d.update(mic_rms=0.01, synth_rms=0.2, mic_peak=0.02)
    assert d.play_gain > 0.5


def test_highpass_removes_dc():
    d = AcousticDucker(enabled=True, highpass=True)
    # DC step
    mic = np.ones(512, dtype=np.float32) * 0.5
    y = d.highpass_filter(mic)
    # after settling, near zero mean
    assert abs(float(np.mean(y[256:]))) < 0.05


def test_world_duck_scales_synth():
    world = AudioWorld()
    world.duck.enabled = True
    world.duck.play_gain = 0.2
    world.set_act_levels([1.0] + [0.0] * (NUM_BANDS - 1))
    pcm_soft = world.render_synth()
    world.duck.play_gain = 1.0
    pcm_loud = world.render_synth()
    assert float(np.max(np.abs(pcm_loud))) > float(np.max(np.abs(pcm_soft))) * 1.5


def test_howl_path_on_step_lowers_play_gain():
    """Simulate acoustic howl: silence capture replaced by loud PCM after play."""
    world = AudioWorld()
    world.duck.enabled = True
    world.duck.howl_rms = 0.1
    world.duck.howl_ratio = 1.2
    world.self_mix = 0.0
    world.mic_gain = 1.0

    class LoudCapture:
        sample_rate = 48000
        chunk_size = 2048

        def read(self):
            # loud noise = howl-ish feedback
            return (0.6 * np.random.randn(2048)).astype(np.float32)

        def close(self):
            pass

    world.set_act_levels([0.8] * NUM_BANDS)
    cap = LoudCapture()
    g0 = world.duck.play_gain
    for _ in range(6):
        world.step(cap, render_motor=True)
    assert world.duck.play_gain < g0
    assert world.duck.duck_events >= 1


def test_collect_meta_poles_and_reinforce():
    host = Symbioid(id="pole-test", label="pole")
    host.interface.continuous_inputs = False
    for i in range(NUM_BANDS):
        host.add_sensor(
            Sensor(id=f"pole:sen:band_{i:02d}", label=f"band_{i:02d}"),
            awareness=False,
        )
        host.add_actuator(
            Actuator(id=f"pole:act:act_{i:02d}", label=f"act_{i:02d}"),
            awareness=False,
        )
    world = AudioWorld()
    world.self_mix = 0.8
    world.mic_gain = 0.0
    # put energy in a band
    levels = [0.0] * NUM_BANDS
    levels[5] = 1.0
    world.set_act_levels(levels)
    cap = SyntheticCapture(mode="silence")
    world.step(cap, render_motor=True)

    poles = collect_audio_state_poles(host, world, top_k=4)
    assert len(poles) >= 3  # peak, energy, err, act_peak at least
    labels = {getattr(p, "label", None) for p in poles}
    assert any(lab and str(lab).startswith("hear_peak") for lab in labels)
    assert any(lab and str(lab).startswith("act_peak") for lab in labels)

    # Sync sense so last_obs exists, then collect again
    w = world.sensor_world()
    handoffs = []
    for sen in host.sensors:
        sense = sen.sample(tick=1, world=w)
        h = host.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h:
            handoffs.append(h)
    host.innerface.accept_formation_batch(
        {"kind": "formation_batch", "handoffs": handoffs, "tick": 1}
    )
    poles2 = collect_audio_state_poles(host, world, top_k=6)
    # should include band observations now
    assert len(poles2) >= len(poles)

    coach = BabbleCoach(mode="contingent", seed=1)
    coach.levels = list(levels)
    coach.decide(world)
    coach.apply_to_host(host)
    n_val_before = len(host.mind._valence)
    coach.reinforce(host, world)
    assert len(host.mind._valence) >= n_val_before
    # record_outcome should have strengthened pairs → more valence keys
    assert len(host.mind._actions) >= 1
    cap.close()


def test_leakage_cancel_reduces_self_in_mic():
    d = AcousticDucker(enabled=True, leakage_cancel=1.0, play_gain=1.0)
    synth = (0.5 * np.sin(2 * np.pi * np.arange(256) / 32)).astype(np.float32)
    mic = synth.copy()  # pure self-leakage
    out = d.cancel_leakage(mic, synth)
    assert float(np.sqrt(np.mean(np.square(out)))) < 0.05
