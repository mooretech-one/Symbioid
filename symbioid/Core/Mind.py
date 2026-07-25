"""Mind aspect — recognition, novelty/habituation, valence (Feeling/Reflection/Maker)."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from symbioid.Core.System import System
from symbioid.Core.Thought import Thought


@dataclass
class AdmitResult:
    """Mind decision for an Input or relation (Follows / Integrates)."""

    action: str  # "skip" | "reuse" | "mint"
    content_key: str
    observation: Optional[Thought] = None
    observation_id: Optional[str] = None
    formation_id: Optional[str] = None  # form id or sync id
    kind: str = "observation"  # "observation" | "follows" | "integrates"


@dataclass
class RecommendResult:
    """Behavior hint grounded in minted Action associations + valence."""

    token: str
    score: float
    domain: str
    content_key: str  # winning action content key
    support: int = 1  # number of state edges contributing


@dataclass
class Mind(System):
    """
    Processor substrate + recognition + policy bias from minted structure.

    Dual-triad roles (design reframe, not six seed poles):
      Thought    — content-addressed Observation + Follows + Integrates + Actions
      Reflection — compare sample / co-occurrence / pair; recommend from graph
      Feeling    — valence + habituation streak
      Maker      — gate mint of Observations / relations / Action poles
    """

    enabled: bool = True
    recognition_enabled: bool = True
    # Thought-as-neuron pulse model (Symbioid.pulse_tick)
    dynamics_enabled: bool = True
    sensor_stimulus: float = 0.8
    observation_stimulus: float = 1.2
    formation_pole_stimulus: float = 0.35
    integrate_pole_stimulus: float = 0.5
    outcome_action_stimulus: float = 1.0
    outcome_state_stimulus: float = 0.4
    propagate_gain: float = 0.6
    port_gain: float = 0.55  # Interface → Innerface export transfer
    recommend_activation_weight: float = 0.5
    # Phase 5: per-engine energy budgets (0 = unlimited)
    energy_budget_default: float = 0.0
    energy_budget_interface: float = 0.0
    energy_budget_innerface: float = 0.0
    energy_budget_outerface: float = 0.0
    energy_fire_cost: float = 1.0
    energy_spread_cost: float = 0.25
    # Phase 5: port queues + Port-Link Hebb (cross-engine only)
    port_queue_max: int = 256
    port_hebb_lr: float = 0.05  # strengthen Port Link on successful transfer
    # When membership pulse: Hebb non-port only if target in membership
    port_hebb_cross_only: bool = True
    # Hebbian Link.weight plasticity (Thought-as-neuron synapses)
    hebb_enabled: bool = True
    hebb_lr: float = 0.08  # base learning rate per co-fire / reinforce
    hebb_co_fire_scale: float = 1.0  # multiply lr when both poles fire same tick
    hebb_pre_post_scale: float = 0.35  # pre fires, post active but not firing
    weight_min: float = 0.05
    weight_max: float = 4.0
    # If weight * propagate_gain >= this × threshold, one pole can recruit mate
    # (documentation / tests; dynamics use weight directly)
    recruit_gain_factor: float = 1.0
    outcome_weight_lr: float = 0.05  # record_outcome nudges edges between poles
    quantize_decimals: int = 3
    # After this many *consecutive identical* keys, further samples are skipped.
    # 1 = mint once then skip; 2 = mint, then one more chance (reuse), then skip.
    habituate_after: int = 2
    max_registry_per_channel: int = 512
    max_follows_registry: int = 1024
    max_integrates_registry: int = 1024
    # Small valence bump on mint / decay on pure reuse
    surprise_valence: float = 0.15
    reuse_valence_decay: float = 0.02
    valence_floor: float = -5.0
    valence_ceil: float = 20.0
    # Behavior: minimum score to emit a recommendation (fail open if colder)
    recommend_min_valence: float = 0.2
    # Activation-based forgetting (structural GC of long-cold unprotected Thoughts)
    forget_cold_enabled: bool = True  # structural GC of long-cold unprotected Thoughts
    forget_cold_cycles: int = 64  # host pulse cycles since last_hot_cycle
    forget_activation_eps: float = 1e-6
    forget_transient_only: bool = False  # when cold-forget on: any unprotected Thought
    forget_max_per_pass: int = 64

    # stats (Observation + Follows + Integrates combined mint counters)
    admits_mint: int = field(default=0, init=False, repr=False)
    admits_reuse: int = field(default=0, init=False, repr=False)
    admits_skip: int = field(default=0, init=False, repr=False)
    follows_mint: int = field(default=0, init=False, repr=False)
    follows_reuse: int = field(default=0, init=False, repr=False)
    follows_skip: int = field(default=0, init=False, repr=False)
    integrates_mint: int = field(default=0, init=False, repr=False)
    integrates_reuse: int = field(default=0, init=False, repr=False)
    integrates_skip: int = field(default=0, init=False, repr=False)
    actions_mint: int = field(default=0, init=False, repr=False)
    outcomes_recorded: int = field(default=0, init=False, repr=False)
    recommends_hit: int = field(default=0, init=False, repr=False)
    recommends_miss: int = field(default=0, init=False, repr=False)
    hebb_updates: int = field(default=0, init=False, repr=False)
    forgets_cold: int = field(default=0, init=False, repr=False)

    # content_key → Observation Thought (canonical)
    _observations: dict[str, Thought] = field(default_factory=dict, init=False, repr=False)
    # content_key → valence
    _valence: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    # thought_id → content_key (for prune protection lookup)
    _thought_to_key: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # sensor_id → last content_key
    _last_key: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # sensor_id → consecutive streak for last key
    _streak: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    # sensor_id → ordered content_keys (LRU-ish)
    _channel_keys: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    # recent keys (any channel) for coach valence fan-out
    _recent_keys: list[str] = field(default_factory=list, init=False, repr=False)
    # Follows co-occurrence: pair content_key → stable sync_id
    _follows: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # Follows pair key → presentation streak
    _follows_streak: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    # LRU order of follows keys
    _follows_order: list[str] = field(default_factory=list, init=False, repr=False)
    # Integrates: pair+channel content_key → stable integrate_id
    _integrates: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _integrates_streak: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _integrates_order: list[str] = field(default_factory=list, init=False, repr=False)
    # Policy: act:{domain}:{token} → Action Thought
    _actions: dict[str, Thought] = field(default_factory=dict, init=False, repr=False)
    # action content_key → token string
    _action_tokens: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def content_key(self, sensor_id: str, sense: Optional[dict[str, Any]]) -> str:
        """Stable signature for one Input on one Sensor channel."""
        sense = sense or {}
        reading = sense.get("reading")
        if reading is not None:
            try:
                q = round(float(reading), int(self.quantize_decimals))
                return f"{sensor_id}:r:{q}"
            except (TypeError, ValueError):
                pass
        value = sense.get("value")
        if value is not None:
            return f"{sensor_id}:v:{str(value)}"
        sample = sense.get("sample")
        if sample is not None:
            return f"{sensor_id}:s:{sample}"
        return f"{sensor_id}:empty"

    def _hash_key(self, content_key: str) -> str:
        return hashlib.sha1(content_key.encode("utf-8")).hexdigest()[:12]

    def observation_id_for(self, host_id: str, content_key: str) -> str:
        return f"{host_id}:obs:{self._hash_key(content_key)}"

    def formation_id_for(self, host_id: str, sensor_id: str, content_key: str) -> str:
        return f"{host_id}:form:{sensor_id}:{self._hash_key(content_key)}"

    def sync_id_for(self, host_id: str, follows_key: str) -> str:
        """Stable Follows six-set id (content-addressed co-occurrence)."""
        return f"{host_id}:sync:{self._hash_key(follows_key)}"

    def pole_content_key(self, observation: Thought) -> str:
        """Best content key for an Observation pole (registry hit or id fallback)."""
        with self._lock:
            ck = self._thought_to_key.get(observation.id)
        if ck:
            return ck
        return observation.id

    def follows_content_key(
        self,
        observation_a: Thought,
        observation_b: Thought,
    ) -> str:
        """
        Undirected co-occurrence key for Follows / FollowedBy.

        Sorted pair of pole content keys so (A,B) and (B,A) mint once.
        """
        ka = self.pole_content_key(observation_a)
        kb = self.pole_content_key(observation_b)
        lo, hi = sorted((ka, kb))
        return f"follows:{lo}|{hi}"

    def integrates_content_key(
        self,
        observation_a: Thought,
        observation_b: Thought,
        *,
        channel: str = "_",
        depth_parents: Optional[tuple[str, str]] = None,
    ) -> str:
        """
        Undirected Integrate key: pair of pole content keys + awareness channel.

        Depth-fold includes sorted parent integrate ids so compressed layers
        do not collide with the first-level pair mint.
        """
        ka = self.pole_content_key(observation_a)
        kb = self.pole_content_key(observation_b)
        lo, hi = sorted((ka, kb))
        ch = channel or "_"
        if depth_parents:
            pa, pb = sorted(depth_parents)
            return f"int:{ch}:depth:{pa}|{pb}:{lo}|{hi}"
        return f"int:{ch}:{lo}|{hi}"

    def integrate_id_for(self, host_id: str, integrates_key: str) -> str:
        return f"{host_id}:int:{self._hash_key(integrates_key)}"

    def valence_of(self, content_key: Optional[str] = None, thought_id: Optional[str] = None) -> float:
        with self._lock:
            if content_key and content_key in self._valence:
                return self._valence[content_key]
            if thought_id:
                ck = self._thought_to_key.get(thought_id)
                if ck:
                    return self._valence.get(ck, 0.0)
            return 0.0

    def is_protected_observation(self, thought_id: str, *, min_valence: float = 0.5) -> bool:
        """True if Thought is a registered Observation with enough valence to keep."""
        with self._lock:
            ck = self._thought_to_key.get(thought_id)
            if ck is None:
                return False
            # Registered Observation / Action poles are always protected (O(1)).
            if ck in self._observations or ck in self._actions:
                return True
            return self._valence.get(ck, 0.0) >= min_valence

    def registered_observation_ids(self) -> set[str]:
        """Poles that must survive GC: Observations + Action policy poles."""
        with self._lock:
            ids = {t.id for t in self._observations.values()}
            ids.update(t.id for t in self._actions.values())
            return ids

    def action_content_key(self, domain: str, token: str) -> str:
        return f"act:{domain}:{token}"

    def ensure_action_thought(
        self,
        domain: str,
        token: str,
        *,
        host_id: str = "host",
        with_labels: bool = True,
    ) -> Thought:
        """Mint or return stable Action pole for behavior associations."""
        token = str(token).strip() or "act"
        domain = str(domain).strip() or "default"
        ck = self.action_content_key(domain, token)
        with self._lock:
            existing = self._actions.get(ck)
            if existing is not None:
                return existing
            oid = f"{host_id}:act:{self._hash_key(ck)}"
            lab = token if with_labels else None
            thought = Thought(id=oid, label=lab, transient=False)
            self._actions[ck] = thought
            self._action_tokens[ck] = token
            self._thought_to_key[thought.id] = ck
            self.actions_mint += 1
            self.admits_mint += 1
            return thought

    def _pair_other(self, lo: str, hi: str, state_key: str) -> Optional[str]:
        """Return the neighbor key if state_key is one side of (lo, hi)."""
        if lo == state_key:
            return hi
        if hi == state_key:
            return lo
        return None

    def _parse_follows_poles(self, fk: str) -> Optional[tuple[str, str]]:
        if not fk.startswith("follows:"):
            return None
        body = fk[len("follows:") :]
        if "|" not in body:
            return None
        lo, hi = body.split("|", 1)
        return lo, hi

    def _parse_integrates_poles(self, ik: str) -> Optional[tuple[str, str]]:
        """
        Parse undirected poles from integrate content keys.

        Formats:
          int:{channel}:{lo}|{hi}
          int:{channel}:depth:{pa}|{pb}:{lo}|{hi}
        """
        if not ik.startswith("int:"):
            return None
        rest = ik[len("int:") :]
        if ":depth:" in rest:
            after = rest.split(":depth:", 1)[1]
            if ":" not in after or "|" not in after:
                return None
            # pa|pb:lo|hi
            try:
                _parents, poles = after.split(":", 1)
                lo, hi = poles.split("|", 1)
                return lo, hi
            except ValueError:
                return None
        if "|" not in rest or ":" not in rest:
            return None
        left, right = rest.split("|", 1)
        # left = channel:lo (lo may contain colons)
        if ":" not in left:
            return None
        lo = left.split(":", 1)[1]
        return lo, right

    def _strengthen_pair(
        self,
        state: Thought,
        action: Thought,
        *,
        host_id: str,
        reward_delta: float,
        channel: str = "policy",
    ) -> None:
        """
        Ensure Follows + policy Integrates links exist and apply valence.
        Bypasses habituation skip so outcomes always reinforce.
        """
        fk = self.follows_content_key(state, action)
        ik = self.integrates_content_key(state, action, channel=channel)
        with self._lock:
            if fk not in self._follows:
                self._follows[fk] = self.sync_id_for(host_id, fk)
                self.follows_mint += 1
                self.admits_mint += 1
            self._touch_follows(fk, self._follows[fk])
            if ik not in self._integrates:
                self._integrates[ik] = self.integrate_id_for(host_id, ik)
                self.integrates_mint += 1
                self.admits_mint += 1
            self._touch_integrates(ik, self._integrates[ik])
        self.note_valence(content_key=fk, delta=reward_delta)
        self.note_valence(content_key=ik, delta=reward_delta)
        act_ck = self._thought_to_key.get(action.id)
        if act_ck:
            self.note_valence(content_key=act_ck, delta=reward_delta * 0.5)

    def record_outcome(
        self,
        state_thoughts: list[Thought],
        action_token: str,
        *,
        domain: str = "default",
        host_id: str = "host",
        reward: float = 0.0,
        channel: str = "policy",
        with_labels: bool = True,
        host: Any = None,
    ) -> Thought:
        """
        Write path: associate current state poles with an Action; apply reward valence.

        Returns the Action Thought. Safe with empty state (still mints Action).
        """
        action = self.ensure_action_thought(
            domain, action_token, host_id=host_id, with_labels=with_labels
        )
        # Scale reward similar to Tetris demo bridge
        delta = max(-2.0, min(2.0, float(reward) / 50.0))
        if abs(delta) < 1e-9:
            delta = 0.05 if reward >= 0 else -0.05
        for st in state_thoughts:
            if st is None:
                continue
            self._strengthen_pair(
                st, action, host_id=host_id, reward_delta=delta, channel=channel
            )
            # Short-term heat on good (or cold on bad) paths
            if self.dynamics_enabled:
                amt = (
                    self.outcome_state_stimulus
                    if delta >= 0
                    else -0.5 * self.outcome_state_stimulus
                )
                if host is not None and hasattr(host, "stimulate"):
                    host.stimulate(st, amt)
                else:
                    st.receive(amt)
        if self.dynamics_enabled:
            amt_a = (
                self.outcome_action_stimulus
                if delta >= 0
                else -0.5 * self.outcome_action_stimulus
            )
            if host is not None and hasattr(host, "stimulate"):
                host.stimulate(action, amt_a)
                if action.id not in getattr(host, "thoughts", {}):
                    host.add_thought(action)
            else:
                action.receive(amt_a)

        # Plastic synapses: reward strengthens edges both ways; penalty weakens
        if (
            host is not None
            and self.hebb_enabled
            and hasattr(host, "reinforce_edge")
        ):
            w_delta = float(self.outcome_weight_lr) * (
                1.0 if delta >= 0 else -1.0
            ) * max(0.25, abs(delta))
            for st in state_thoughts:
                if st is None:
                    continue
                if st.id not in getattr(host, "thoughts", {}):
                    host.add_thought(st)
                n = host.reinforce_edge(st, action, delta=w_delta)
                n += host.reinforce_edge(action, st, delta=w_delta)
                if n == 0 and hasattr(host, "ensure_reciprocal_links"):
                    host.ensure_reciprocal_links(st, action, initial_weight=1.0 + w_delta)

        with self._lock:
            self.outcomes_recorded += 1
        return action

    def recommend_action(
        self,
        state_thoughts: list[Thought],
        *,
        domain: str = "default",
        min_score: Optional[float] = None,
        require_activation: bool = False,
    ) -> Optional[RecommendResult]:
        """
        Read path: among Action poles linked to state via Follows/Integrates,
        return highest-valence (+ activation) action token in ``domain``.
        Fail open → None when cold.
        """
        if not self.enabled or not state_thoughts:
            with self._lock:
                self.recommends_miss += 1
            return None
        threshold = (
            float(self.recommend_min_valence)
            if min_score is None
            else float(min_score)
        )
        domain = str(domain).strip() or "default"
        prefix = f"act:{domain}:"
        alpha = float(self.recommend_activation_weight)

        scores: dict[str, float] = {}
        support: dict[str, int] = {}
        act_keys: dict[str, str] = {}

        with self._lock:
            state_keys = set()
            for t in state_thoughts:
                if t is None:
                    continue
                ck = self._thought_to_key.get(t.id) or t.id
                state_keys.add(ck)

            def consider(pair_key: str, other: str) -> None:
                if not other.startswith(prefix):
                    return
                token = self._action_tokens.get(other)
                if not token:
                    # parse act:domain:token
                    parts = other.split(":", 2)
                    if len(parts) < 3:
                        return
                    token = parts[2]
                act_th = self._actions.get(other)
                act_act = float(act_th.activation) if act_th is not None else 0.0
                fired = 1.0 if act_th is not None and act_th.just_fired else 0.0
                if require_activation and act_th is not None:
                    if act_act < 0.5 * float(act_th.threshold) and not act_th.just_fired:
                        return
                sc = (
                    self._valence.get(pair_key, 0.0)
                    + 0.5 * self._valence.get(other, 0.0)
                    + alpha * act_act
                    + 0.25 * fired
                )
                scores[token] = scores.get(token, 0.0) + sc
                support[token] = support.get(token, 0) + 1
                act_keys[token] = other

            for fk in self._follows:
                poles = self._parse_follows_poles(fk)
                if not poles:
                    continue
                lo, hi = poles
                for sk in state_keys:
                    other = self._pair_other(lo, hi, sk)
                    if other:
                        consider(fk, other)

            for ik in self._integrates:
                poles = self._parse_integrates_poles(ik)
                if not poles:
                    continue
                lo, hi = poles
                for sk in state_keys:
                    other = self._pair_other(lo, hi, sk)
                    if other:
                        consider(ik, other)

        if not scores:
            with self._lock:
                self.recommends_miss += 1
            return None

        best_token = max(scores.keys(), key=lambda t: (scores[t], support.get(t, 0)))
        best_score = scores[best_token]
        if best_score < threshold:
            with self._lock:
                self.recommends_miss += 1
            return None

        with self._lock:
            self.recommends_hit += 1
        return RecommendResult(
            token=best_token,
            score=best_score,
            domain=domain,
            content_key=act_keys.get(best_token, self.action_content_key(domain, best_token)),
            support=support.get(best_token, 1),
        )

    def note_valence(
        self,
        *,
        content_key: Optional[str] = None,
        thought_id: Optional[str] = None,
        channel: Optional[str] = None,
        delta: float,
        recent: int = 8,
    ) -> None:
        """
        Feeling: adjust valence on a key, thought, or recent keys (coach bridge).

        channel=\"board\" fans delta across recent Observation keys.
        """
        d = float(delta)
        with self._lock:
            keys: list[str] = []
            if content_key:
                keys.append(content_key)
            if thought_id:
                ck = self._thought_to_key.get(thought_id)
                if ck:
                    keys.append(ck)
            if channel == "board" or (not keys and recent > 0):
                keys.extend(self._recent_keys[-max(1, recent) :])
            if not keys:
                return
            seen: set[str] = set()
            for ck in keys:
                if ck in seen:
                    continue
                seen.add(ck)
                v = self._valence.get(ck, 0.0) + d
                self._valence[ck] = max(self.valence_floor, min(self.valence_ceil, v))

    def _touch_channel(self, sensor_id: str, content_key: str) -> None:
        keys = self._channel_keys.setdefault(sensor_id, [])
        if content_key in keys:
            keys.remove(content_key)
        keys.append(content_key)
        while len(keys) > int(self.max_registry_per_channel):
            old = keys.pop(0)
            # Evict lowest-valence among overflow only if still first
            self._maybe_evict(sensor_id, old)

    def _maybe_evict(self, sensor_id: str, content_key: str) -> None:
        """Drop registry entry if low valence and not last key."""
        if self._last_key.get(sensor_id) == content_key:
            return
        v = self._valence.get(content_key, 0.0)
        if v >= 0.5:
            return
        obs = self._observations.pop(content_key, None)
        self._valence.pop(content_key, None)
        if obs is not None:
            self._thought_to_key.pop(obs.id, None)

    def _register(
        self,
        content_key: str,
        observation: Thought,
        *,
        sensor_id: str,
        valence: float = 0.0,
    ) -> None:
        self._observations[content_key] = observation
        self._thought_to_key[observation.id] = content_key
        self._valence[content_key] = max(
            self.valence_floor, min(self.valence_ceil, float(valence))
        )
        self._touch_channel(sensor_id, content_key)
        self._recent_keys.append(content_key)
        if len(self._recent_keys) > 64:
            del self._recent_keys[:-64]

    def admit_input(
        self,
        sensor_id: str,
        sense: Optional[dict[str, Any]],
        *,
        host_id: str = "host",
        with_labels: bool = True,
    ) -> AdmitResult:
        """
        Reflection + Maker: decide skip | reuse | mint for this Input.

        When recognition is off or Mind disabled, always mint with a unique
        generation-style id (legacy growth behaviour).
        """
        sense = sense or {}
        if not self.enabled or not self.recognition_enabled:
            # Legacy: unique observation each call (Interface still bumps gen)
            gen = sense.get("sample") or sense.get("tick") or id(sense)
            ck = f"{sensor_id}:legacy:{gen}"
            oid = f"{host_id}:obs:legacy:{self._hash_key(ck)}"
            fid = f"{host_id}:form:{sensor_id}:legacy:{self._hash_key(ck)}"
            with self._lock:
                self.admits_mint += 1
            return AdmitResult(
                action="mint",
                content_key=ck,
                observation_id=oid,
                formation_id=fid,
                kind="observation",
            )

        ck = self.content_key(sensor_id, sense)
        with self._lock:
            prev = self._last_key.get(sensor_id)
            if prev == ck:
                streak = self._streak.get(sensor_id, 0) + 1
            else:
                streak = 1
            self._last_key[sensor_id] = ck
            self._streak[sensor_id] = streak

            known = self._observations.get(ck)
            if known is not None and streak >= int(self.habituate_after):
                self.admits_skip += 1
                # mild decay on ignored re-presentation
                v = self._valence.get(ck, 0.0) - self.reuse_valence_decay * 0.5
                self._valence[ck] = max(self.valence_floor, v)
                return AdmitResult(
                    action="skip",
                    content_key=ck,
                    observation=known,
                    observation_id=known.id,
                    formation_id=self.formation_id_for(host_id, sensor_id, ck),
                    kind="observation",
                )

            if known is not None:
                self.admits_reuse += 1
                v = self._valence.get(ck, 0.0) - self.reuse_valence_decay
                self._valence[ck] = max(self.valence_floor, v)
                self._touch_channel(sensor_id, ck)
                self._recent_keys.append(ck)
                if len(self._recent_keys) > 64:
                    del self._recent_keys[:-64]
                return AdmitResult(
                    action="reuse",
                    content_key=ck,
                    observation=known,
                    observation_id=known.id,
                    formation_id=self.formation_id_for(host_id, sensor_id, ck),
                    kind="observation",
                )

            # mint
            oid = self.observation_id_for(host_id, ck)
            fid = self.formation_id_for(host_id, sensor_id, ck)
            lab = None
            if with_labels:
                val = sense.get("value")
                if val is None and sense.get("reading") is not None:
                    lab = f"{sensor_id}:{sense['reading']}"
                else:
                    lab = str(val) if val is not None else ck
            obs = Thought(id=oid, label=lab, transient=True)
            self._register(
                ck, obs, sensor_id=sensor_id, valence=self.surprise_valence
            )
            self.admits_mint += 1
            return AdmitResult(
                action="mint",
                content_key=ck,
                observation=obs,
                observation_id=oid,
                formation_id=fid,
                kind="observation",
            )

    def _evict_registry_hard(
        self,
        store: dict[str, Any],
        order: list[str],
        streak: dict[str, int],
        max_n: int,
        *,
        keep_key: Optional[str] = None,
    ) -> None:
        """
        Hard-cap a content registry by lowest valence (then oldest).

        Unlike soft eviction, high-valence keys are still dropped when over cap
        so the map cannot grow unbounded.
        """
        max_n = max(1, int(max_n))
        while len(store) > max_n:
            candidates = [k for k in store.keys() if k != keep_key]
            if not candidates:
                break
            order_index = {k: i for i, k in enumerate(order)}

            def _rank(k: str) -> tuple[float, int]:
                return (
                    float(self._valence.get(k, 0.0)),
                    order_index.get(k, 0),
                )

            old = min(candidates, key=_rank)
            store.pop(old, None)
            streak.pop(old, None)
            # Keep valence on Action poles / observations elsewhere; drop pair valence
            if old.startswith("follows:") or old.startswith("int:"):
                self._valence.pop(old, None)
            if old in order:
                order.remove(old)

    def _touch_follows(self, follows_key: str, sync_id: str) -> None:
        self._follows[follows_key] = sync_id
        if follows_key in self._follows_order:
            self._follows_order.remove(follows_key)
        self._follows_order.append(follows_key)
        self._evict_registry_hard(
            self._follows,
            self._follows_order,
            self._follows_streak,
            int(self.max_follows_registry),
            keep_key=follows_key,
        )

    def admit_follows(
        self,
        observation_a: Thought,
        observation_b: Thought,
        *,
        host_id: str = "host",
    ) -> AdmitResult:
        """
        Maker for lateral co-occurrence: Follows / FollowedBy.

        Content key = undirected pair of Observation content keys.
        mint → new stable sync_id; reuse → same sync_id, no new scaffolding;
        skip → habituated re-presentation (no handoff / no re-integrate).
        """
        if observation_a.id == observation_b.id:
            return AdmitResult(
                action="skip",
                content_key="follows:self",
                kind="follows",
            )

        if not self.enabled or not self.recognition_enabled:
            # Legacy: unique sync id each time
            tickish = id(observation_a) ^ id(observation_b) ^ len(self._follows)
            ck = f"follows:legacy:{observation_a.id}:{observation_b.id}:{tickish}"
            sid = f"{host_id}:sync:legacy:{self._hash_key(ck)}"
            with self._lock:
                self.admits_mint += 1
                self.follows_mint += 1
            return AdmitResult(
                action="mint",
                content_key=ck,
                formation_id=sid,
                kind="follows",
            )

        ck = self.follows_content_key(observation_a, observation_b)
        with self._lock:
            streak = self._follows_streak.get(ck, 0) + 1
            self._follows_streak[ck] = streak
            known_sid = self._follows.get(ck)

            if known_sid is not None and streak >= int(self.habituate_after):
                self.admits_skip += 1
                self.follows_skip += 1
                v = self._valence.get(ck, 0.0) - self.reuse_valence_decay * 0.5
                self._valence[ck] = max(self.valence_floor, v)
                return AdmitResult(
                    action="skip",
                    content_key=ck,
                    formation_id=known_sid,
                    kind="follows",
                )

            if known_sid is not None:
                self.admits_reuse += 1
                self.follows_reuse += 1
                v = self._valence.get(ck, 0.0) - self.reuse_valence_decay
                self._valence[ck] = max(self.valence_floor, v)
                self._touch_follows(ck, known_sid)
                return AdmitResult(
                    action="reuse",
                    content_key=ck,
                    formation_id=known_sid,
                    kind="follows",
                )

            # mint new co-occurrence association
            sid = self.sync_id_for(host_id, ck)
            self._valence[ck] = self.surprise_valence
            self._touch_follows(ck, sid)
            self.admits_mint += 1
            self.follows_mint += 1
            return AdmitResult(
                action="mint",
                content_key=ck,
                formation_id=sid,
                kind="follows",
            )

    def _touch_integrates(self, integrates_key: str, integrate_id: str) -> None:
        self._integrates[integrates_key] = integrate_id
        if integrates_key in self._integrates_order:
            self._integrates_order.remove(integrates_key)
        self._integrates_order.append(integrates_key)
        self._evict_registry_hard(
            self._integrates,
            self._integrates_order,
            self._integrates_streak,
            int(self.max_integrates_registry),
            keep_key=integrates_key,
        )

    def admit_integrates(
        self,
        observation_a: Thought,
        observation_b: Thought,
        *,
        host_id: str = "host",
        channel: str = "_",
        depth_parents: Optional[tuple[str, str]] = None,
    ) -> AdmitResult:
        """
        Maker for Rodin-halving Integrates / IntegratedBy.

        Content key = undirected pole pair + awareness channel (+ depth parents).
        mint → stable integrate_id; reuse / skip avoid reminting scaffolding.
        """
        if observation_a.id == observation_b.id and not depth_parents:
            return AdmitResult(
                action="skip",
                content_key="int:self",
                kind="integrates",
            )

        if not self.enabled or not self.recognition_enabled:
            tickish = id(observation_a) ^ id(observation_b) ^ len(self._integrates)
            ck = (
                f"int:legacy:{channel}:{observation_a.id}:{observation_b.id}:{tickish}"
            )
            iid = f"{host_id}:int:legacy:{self._hash_key(ck)}"
            with self._lock:
                self.admits_mint += 1
                self.integrates_mint += 1
            return AdmitResult(
                action="mint",
                content_key=ck,
                formation_id=iid,
                kind="integrates",
            )

        ck = self.integrates_content_key(
            observation_a,
            observation_b,
            channel=channel,
            depth_parents=depth_parents,
        )
        with self._lock:
            streak = self._integrates_streak.get(ck, 0) + 1
            self._integrates_streak[ck] = streak
            known_id = self._integrates.get(ck)

            if known_id is not None and streak >= int(self.habituate_after):
                self.admits_skip += 1
                self.integrates_skip += 1
                v = self._valence.get(ck, 0.0) - self.reuse_valence_decay * 0.5
                self._valence[ck] = max(self.valence_floor, v)
                return AdmitResult(
                    action="skip",
                    content_key=ck,
                    formation_id=known_id,
                    kind="integrates",
                )

            if known_id is not None:
                self.admits_reuse += 1
                self.integrates_reuse += 1
                v = self._valence.get(ck, 0.0) - self.reuse_valence_decay
                self._valence[ck] = max(self.valence_floor, v)
                self._touch_integrates(ck, known_id)
                return AdmitResult(
                    action="reuse",
                    content_key=ck,
                    formation_id=known_id,
                    kind="integrates",
                )

            iid = self.integrate_id_for(host_id, ck)
            self._valence[ck] = self.surprise_valence
            self._touch_integrates(ck, iid)
            self.admits_mint += 1
            self.integrates_mint += 1
            return AdmitResult(
                action="mint",
                content_key=ck,
                formation_id=iid,
                kind="integrates",
            )

    def summary(self) -> str:
        with self._lock:
            return (
                f"mind mint={self.admits_mint} reuse={self.admits_reuse} "
                f"skip={self.admits_skip} registry={len(self._observations)} "
                f"follows={len(self._follows)}"
                f"(m{self.follows_mint}/r{self.follows_reuse}/s{self.follows_skip}) "
                f"int={len(self._integrates)}"
                f"(m{self.integrates_mint}/r{self.integrates_reuse}/s{self.integrates_skip}) "
                f"act={len(self._actions)} rec={self.recommends_hit}/{self.recommends_miss} "
                f"hebb={self.hebb_updates}"
            )
