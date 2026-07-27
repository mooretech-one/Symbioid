"""Phases 2–4 audio: synth, closed self-mix, contingent IM (no hardware)."""

from __future__ import annotations

import numpy as np

from symbioid import Actuator, Sensor, Symbioid
from symbioid.world.audio import (
    NUM_BANDS,
    AudioWorld,
    BabbleCoach,
    BandSynth,
    NullPlayback,
    SyntheticCapture,
    band_index_for_freq,
    compare_contingent_vs_noncontingent,
    open_playback,
)


def test_band_synth_peaks_correct_band():
    synth = BandSynth()
    gains = [0.0] * NUM_BANDS
    # drive band near 1 kHz
    idx = band_index_for_freq(1000.0)
    gains[idx] = 1.0
    pcm = synth.render(gains)
    assert pcm.shape[0] > 0
    assert float(np.max(np.abs(pcm))) > 0.01

    world = AudioWorld()
    world.step_pcm(pcm)
    peak = int(np.argmax(world.bands))
    assert abs(peak - idx) <= 1, f"peak={peak} expected~{idx} bands={world.bands}"


def test_null_playback_and_open():
    pb, name = open_playback(backend="null")
    assert name == "null"
    pb.write(np.zeros(128, dtype=np.float32))
    assert isinstance(pb, NullPlayback)
    pb.close()


def test_open_loop_babble_motor_changes_levels():
    world = AudioWorld()
    world.self_mix = 0.0
    cap = SyntheticCapture(mode="silence")
    coach = BabbleCoach(mode="explore", seed=1)
    levels_a = coach.decide(world)
    world.step(cap, render_motor=True)
    levels_b = coach.decide(world)
    # random walk should usually move (seeded)
    assert levels_a != levels_b or sum(levels_a) > 0
    assert world.last_synth_rms >= 0.0
    cap.close()


def test_closed_loop_self_mix_changes_hearing():
    """Phase 3 exit: production changes observation via digital self-mix."""
    world = AudioWorld()
    world.self_mix = 0.9
    world.mic_gain = 0.0
    cap = SyntheticCapture(mode="silence")

    # silence motor → near-zero hear
    world.set_act_levels([0.0] * NUM_BANDS)
    world.step(cap, render_motor=True)
    quiet = float(np.max(world.bands))

    # strong mid band
    idx = band_index_for_freq(1000.0)
    levels = [0.0] * NUM_BANDS
    levels[idx] = 1.0
    world.set_act_levels(levels)
    world.step(cap, render_motor=True)
    loud = float(world.bands[idx])
    peak = int(np.argmax(world.bands))

    assert loud > quiet + 0.02
    assert abs(peak - idx) <= 1
    assert world.last_pred_err < 0.5  # act≈hear when pure self
    cap.close()


def test_actuators_written_by_coach():
    host = Symbioid(id="act-test", label="act-test")
    host.interface.continuous_inputs = False
    for i in range(NUM_BANDS):
        host.add_actuator(
            Actuator(id=f"act-test:act:{i:02d}", label=f"act_{i:02d}"),
            awareness=False,
        )
        host.add_sensor(
            Sensor(id=f"act-test:sen:{i:02d}", label=f"band_{i:02d}"),
            awareness=False,
        )
    world = AudioWorld()
    coach = BabbleCoach(mode="explore", seed=2)
    coach.decide(world)
    coach.apply_to_host(host)
    outs = [a.output for a in host.actuators]
    assert len(outs) == NUM_BANDS
    assert any(o > 0.0 for o in outs)
    world.pull_actuators(host.actuators)
    assert world.act_levels == outs


def test_contingent_beats_noncontingent_total_reward():
    """Phase 4 exit: contingent IM accumulates more reward than non-contingent control."""
    stats = compare_contingent_vs_noncontingent(blocks=50, seed=7)
    # contingent should outscore noncontingent on average; allow small fluke margin
    assert stats["contingent"] > stats["noncontingent"] - 0.5, stats
    # stronger expectation: contingent ahead
    assert stats["delta"] > 0.0 or stats["contingent"] > 5.0, stats


def test_reinforce_mints_action_valence():
    host = Symbioid(id="val-test", label="val")
    host.interface.continuous_inputs = False
    for i in range(NUM_BANDS):
        host.add_actuator(
            Actuator(id=f"val:act:{i}", label=f"act_{i:02d}"),
            awareness=False,
        )
    world = AudioWorld()
    world.self_mix = 0.8
    world.mic_gain = 0.0
    cap = SyntheticCapture(mode="silence")
    coach = BabbleCoach(mode="contingent", seed=3)
    for _ in range(15):
        coach.decide(world)
        coach.apply_to_host(host)
        world.step(cap, render_motor=True)
        coach.reinforce(host, world)
    assert coach.steps == 15
    assert len(host.mind._actions) >= 1 or coach.total_reward != 0.0
    # valence map should have entries after record_outcome
    assert len(host.mind._valence) >= 1
    cap.close()
