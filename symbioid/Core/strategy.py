"""
Iterated twin strategy layer (Tit-for-Tat shaped).

System ⋈ Environment is an infinite game. This module labels rounds and
implements **bounded retaliation + forgiveness** over episode grudge keys —
complementing credit hygiene (asymmetric negatives / skip place-on-topout).

See vault: Work-Log/2026-08-02-research-loop-symbioid-game-theory-tit-for-tat.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence


class RoundLabel(str, Enum):
    """Cooperate / defect taxonomy for env–system rounds."""

    C = "C"
    D_env = "D_env"
    D_self = "D_self"
    U = "U"

    @classmethod
    def parse(cls, value: object) -> "RoundLabel":
        if isinstance(value, cls):
            return value
        s = str(value).strip()
        if s in ("D", "d", "defect"):
            return cls.D_env
        try:
            return cls(s)
        except ValueError:
            # Accept D_ENV style
            low = s.lower()
            for m in cls:
                if m.value.lower() == low:
                    return m
            return cls.U


class TftState(str, Enum):
    open = "open"
    retaliate = "retaliate"
    forgiving = "forgiving"


@dataclass
class RoundEvent:
    label: RoundLabel
    source: str = "unknown"  # env | self | unknown
    channel: str = ""
    keys: tuple[str, ...] = ()

    @classmethod
    def make(
        cls,
        label: object,
        *,
        source: str = "unknown",
        channel: str = "",
        keys: Optional[Iterable[str]] = None,
    ) -> "RoundEvent":
        ks = tuple(str(k) for k in (keys or ()) if k)
        return cls(
            label=RoundLabel.parse(label),
            source=str(source or "unknown"),
            channel=str(channel or ""),
            keys=ks,
        )


@dataclass
class TitForTatConfig:
    """Knobs for TFT-shaped valence recovery (runtime; not constitution)."""

    enabled: bool = True
    forgive_after_n_c: int = 4
    forgive_gamma: float = 0.5  # multiply grudge valence by gamma (pull toward 0 if |v| shrinks)
    # If True, move valence toward 0 by: v * gamma (gamma in (0,1) shrinks |v|)
    # gamma=0.5 halves residual grudge each forgiveness.


@dataclass
class TitForTatPolicy:
    """
    Nice / retaliatory / forgiving / clear episode controller.

    State machine:
      open  --D--> retaliate  --(N×C + forgive)--> open
      open  --C--> open (c_streak++)
    """

    config: TitForTatConfig = field(default_factory=TitForTatConfig)
    state: str = TftState.open.value
    c_streak: int = 0
    grudge_keys: set[str] = field(default_factory=set)
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "C": 0,
            "D_env": 0,
            "D_self": 0,
            "U": 0,
            "D": 0,  # any defect
        }
    )
    last_label: str = ""
    forgives: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "tft_state": self.state,
            "c_streak": int(self.c_streak),
            "grudge_n": len(self.grudge_keys),
            "counts": dict(self.counts),
            "last_label": self.last_label,
            "forgives": int(self.forgives),
            "enabled": bool(self.config.enabled),
            "forgive_after_n_c": int(self.config.forgive_after_n_c),
            "forgive_gamma": float(self.config.forgive_gamma),
        }

    def note_round(self, event: RoundEvent | object, **kwargs: Any) -> RoundEvent:
        if not isinstance(event, RoundEvent):
            event = RoundEvent.make(event, **kwargs)
        lab = event.label
        self.last_label = lab.value
        if lab == RoundLabel.C:
            self.counts["C"] = int(self.counts.get("C", 0)) + 1
            self.c_streak = int(self.c_streak) + 1
            if self.state == TftState.forgiving.value:
                pass
            # stay open or move toward forgive opportunity
        elif lab in (RoundLabel.D_env, RoundLabel.D_self):
            self.counts["D"] = int(self.counts.get("D", 0)) + 1
            self.counts[lab.value] = int(self.counts.get(lab.value, 0)) + 1
            self.c_streak = 0
            self.state = TftState.retaliate.value
            for k in event.keys:
                self.grudge_keys.add(str(k))
        else:
            self.counts["U"] = int(self.counts.get("U", 0)) + 1
            self.last_label = RoundLabel.U.value
        return event

    def maybe_forgive(
        self,
        valence: MutableMapping[str, float],
        *,
        floor: float = -5.0,
        ceil: float = 20.0,
    ) -> dict[str, Any]:
        """
        If c_streak >= N and grudge non-empty, shrink grudge key valence toward 0.

        Returns stats dict (forgiven, n_keys, state).
        """
        cfg = self.config
        out: dict[str, Any] = {
            "forgiven": 0,
            "n_keys": 0,
            "state": self.state,
            "c_streak": int(self.c_streak),
        }
        if not cfg.enabled:
            return out
        need = max(1, int(cfg.forgive_after_n_c))
        if int(self.c_streak) < need or not self.grudge_keys:
            return out
        gamma = float(cfg.forgive_gamma)
        gamma = min(1.0, max(0.0, gamma))
        self.state = TftState.forgiving.value
        touched = 0
        for ck in list(self.grudge_keys):
            if ck not in valence:
                continue
            v = float(valence.get(ck, 0.0))
            nv = v * gamma
            if abs(nv) < 1e-6:
                nv = 0.0
            valence[ck] = max(floor, min(ceil, nv))
            touched += 1
        self.grudge_keys.clear()
        self.state = TftState.open.value
        self.forgives += 1
        out["forgiven"] = 1
        out["n_keys"] = touched
        out["state"] = self.state
        return out

    def reset_episode(self, *, clear_counts: bool = True) -> None:
        """Clear streak/grudge/state. Optionally zero C/D counts (per-game metrics)."""
        self.state = TftState.open.value
        self.c_streak = 0
        self.grudge_keys.clear()
        self.last_label = ""
        if clear_counts:
            for k in list(self.counts.keys()):
                self.counts[k] = 0
