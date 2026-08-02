"""v0.0.53: iterated twin TFT — C/D labels + forgiveness."""

from __future__ import annotations

from symbioid.Core.Mind import Mind
from symbioid.Core.strategy import (
    RoundLabel,
    TitForTatConfig,
    TitForTatPolicy,
)


def test_round_label_parse():
    assert RoundLabel.parse("C") is RoundLabel.C
    assert RoundLabel.parse("D") is RoundLabel.D_env
    assert RoundLabel.parse("D_self") is RoundLabel.D_self


def test_note_round_and_counts():
    m = Mind()
    m.note_round("C", source="env", channel="t")
    m.note_round("C", source="env", channel="t")
    m.note_round("D_env", source="env", channel="t", keys=["a:place"])
    snap = m.tft_snapshot()
    assert snap["counts"]["C"] == 2
    assert snap["counts"]["D"] == 1
    assert snap["tft_state"] == "retaliate"
    assert snap["c_streak"] == 0
    assert "a:place" in m.tft.grudge_keys


def test_forgive_after_cooperative_streak():
    m = Mind()
    m.tft.config = TitForTatConfig(forgive_after_n_c=3, forgive_gamma=0.5)
    m.note_valence(content_key="cell_r10_c03:place", delta=-4.0)
    m.note_round("D_env", keys=["cell_r10_c03:place"])
    assert m._valence["cell_r10_c03:place"] == -4.0
    # Not enough C yet
    m.note_round("C")
    m.note_round("C")
    r = m.maybe_forgive()
    assert r["forgiven"] == 0
    assert m._valence["cell_r10_c03:place"] == -4.0
    m.note_round("C")
    r = m.maybe_forgive()
    assert r["forgiven"] == 1
    assert r["n_keys"] == 1
    # gamma 0.5 → -2.0
    assert abs(m._valence["cell_r10_c03:place"] - (-2.0)) < 1e-9
    assert m.tft_snapshot()["tft_state"] == "open"
    assert not m.tft.grudge_keys


def test_forgive_disabled():
    m = Mind()
    m.tft.config.enabled = False
    m.note_valence(content_key="k", delta=-3.0)
    m.note_round("D_env", keys=["k"])
    for _ in range(10):
        m.note_round("C")
    r = m.maybe_forgive()
    assert r["forgiven"] == 0
    assert m._valence["k"] == -3.0


def test_policy_standalone():
    p = TitForTatPolicy(config=TitForTatConfig(forgive_after_n_c=1, forgive_gamma=0.0))
    val = {"x": -5.0}
    p.note_round("D_env", keys=["x"])
    p.note_round("C")
    r = p.maybe_forgive(val)
    assert r["forgiven"] == 1
    assert val["x"] == 0.0


def test_warm_start_action_prior():
    m = Mind(warm_start_actions=True, warm_start_prior=0.12)
    th = m.ensure_action_thought("tetris", "hard", host_id="h")
    ck = m.action_content_key("tetris", "hard")
    assert m._valence.get(ck, 0.0) == 0.12
    # second ensure does not re-mint or clobber
    m._valence[ck] = 1.5
    th2 = m.ensure_action_thought("tetris", "hard", host_id="h")
    assert th2 is th
    assert m._valence[ck] == 1.5


def test_warm_start_disabled():
    m = Mind(warm_start_actions=False, warm_start_prior=0.12)
    m.ensure_action_thought("pong", "up", host_id="h")
    ck = m.action_content_key("pong", "up")
    assert ck not in m._valence or m._valence.get(ck, 0.0) == 0.0


def test_tft_to_from_dict_roundtrip():
    m = Mind()
    m.tft.config.forgive_after_n_c = 5
    m.tft.config.forgive_random_d_prob = 0.25
    m.tft.config.retaliate_gate = True
    m.tft.config.block_tokens_on_retaliate = ("hard", "explore")
    m.note_round("D_env", keys=["k1", "k2"])
    m.note_round("C")
    blob = m.tft_export()
    m2 = Mind()
    m2.tft_import(blob)
    assert m2.tft.state == "retaliate"
    assert m2.tft.c_streak == 1
    assert m2.tft.grudge_keys == {"k1", "k2"}
    assert m2.tft.config.forgive_after_n_c == 5
    assert abs(m2.tft.config.forgive_random_d_prob - 0.25) < 1e-9
    assert m2.tft.config.retaliate_gate is True
    assert set(m2.tft.config.block_tokens_on_retaliate) == {"hard", "explore"}


def test_generous_tft_noise_forgive():
    p = TitForTatPolicy(
        config=TitForTatConfig(forgive_random_d_prob=1.0)  # always ignore D_env
    )
    p.note_round("D_env", keys=["x"])
    assert p.state == "open"
    assert not p.grudge_keys
    assert p.counts.get("noise_forgive", 0) == 1
    assert p.counts.get("D", 0) == 0


def test_retaliate_gate_blocks_token():
    m = Mind()
    m.tft.config.retaliate_gate = True
    m.tft.config.block_tokens_on_retaliate = ("hard",)
    assert m.should_block_token("hard") is False  # open
    m.note_round("D_env", keys=["g"])
    assert m.tft.state == "retaliate"
    assert m.should_block_token("hard") is True
    assert m.should_block_token("left") is False
