"""Outerface process — Belief six-sets (Expects/ExpectedBy) + gated Actuators.

Manuscript (Antelligence Architecture, Outerfaces):
  Stable: Self, World, Beliefs (a *vast cluster* — many at once)
  Dynamic cycle: Observation → Belief → Action → World → Feedback → Observation

Belief: six-set that stores an Interface Observation caused by Feedback as the
**expected value** for that Feedback channel (typically one Belief per Sensor).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from symbioid.Core.Link import Link
from symbioid.Core.Process import Process
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import (
    complete_belief_set,
    emit_six_set,
    six_set_poles,
)
from symbioid.Core.ids import _new_id

if TYPE_CHECKING:
    from symbioid.Core.Symbioid import Symbioid


def _short_action(action: Optional[str]) -> str:
    """
    Stable short action token for Feedback labels / pending state.
    Never embed Belief text (prevents Feedback[eye|hand:Feedback[…]] nesting).
    """
    if not action:
        return "act"
    token = str(action).strip()
    # Only the actuator / verb token before any ':' payload
    token = token.split(":", 1)[0].strip()
    if not token or token.startswith("Feedback") or "anticipates" in token:
        return "act"
    # Keep labels short
    return token[:24]


def _reading_from_observation(obs: Thought) -> Optional[float]:
    """Parse numeric reading from Observation label like 'eye:0.9211'."""
    lab = obs.label or ""
    if ":" not in lab:
        return None
    try:
        return float(lab.rsplit(":", 1)[-1])
    except ValueError:
        return None


def _expected_observation_from_store(store: dict[str, Thought]) -> Optional[Thought]:
    """Non-Feedback pole of a Belief six-set (the expected Observation)."""
    poles = six_set_poles(store)
    for p in poles:
        if p.label and str(p.label).startswith("Feedback["):
            continue
        return p
    return poles[-1] if poles else None


@dataclass
class Outerface(Process):
    """
    Outer process (~9): Belief → Action → World → Feedback.

    One active Belief six-set per Sensor channel (Feedback[eye] anticipates …).
    Updating Feedback revises the expected Observation; it does not nest labels.
    """

    id: str = field(default_factory=lambda: _new_id("outer-"))
    label: Optional[str] = "Outerface"
    agency_ticks: int = field(default=0, init=False, repr=False)
    last_gate: Optional[str] = field(default=None, init=False, repr=False)

    # belief_id → six-Thought store (stable id per sensor channel)
    completed_beliefs: dict[str, dict[str, Thought]] = field(
        default_factory=dict, init=False, repr=False
    )
    active_belief_ids: set[str] = field(default_factory=set, init=False, repr=False)
    # sensor_id → belief_id (exactly one Belief channel per sensor)
    belief_by_sensor: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # After Actuator fire, next Interface Observations on these sensors are Feedback
    _pending_feedback: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    # Last Feedback-confirmed (or seeded) reading per sensor — for stale detection
    _prior_reading: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    beliefs_created: int = field(default=0, init=False, repr=False)
    beliefs_updated: int = field(default=0, init=False, repr=False)
    belief_challenges: int = field(default=0, init=False, repr=False)
    belief_confirms: int = field(default=0, init=False, repr=False)
    belief_stale_skips: int = field(default=0, init=False, repr=False)
    # |actual − expected| below this → confirm (tight: avoid 1.0 vs cos(0.2) false match)
    expectation_tolerance: float = 0.02
    # Closed-loop: do not fire again until pending Feedback is resolved
    wait_for_feedback: bool = True
    actuator_fires: int = field(default=0, init=False, repr=False)
    actuator_denies: int = field(default=0, init=False, repr=False)
    last_fire: Optional[dict[str, Any]] = field(default=None, init=False, repr=False)
    sets_emitted: int = field(default=0, init=False, repr=False)

    def process(self) -> Optional[threading.Thread]:
        return super().process()

    @property
    def beliefs(self) -> dict[str, dict[str, Thought]]:
        """Active Belief six-sets only."""
        with self._local_lock:
            return {
                bid: self.completed_beliefs[bid]
                for bid in self.active_belief_ids
                if bid in self.completed_beliefs
            }

    def _belief_id_for_sensor(self, host_id: str, sensor_id: str) -> str:
        """Stable one Belief per sensor channel."""
        return f"{host_id}:belief:ch:{sensor_id}"

    def record_actuator_fire(
        self,
        *,
        actuator_id: str,
        action: str,
        reason: str,
        belief_id: Optional[str] = None,
    ) -> None:
        """Record a gated fire and arm Sensors for Feedback Observations."""
        host = self.host
        short = _short_action(action)
        with self._local_lock:
            self.actuator_fires += 1
            self.last_fire = {
                "actuator_id": actuator_id,
                "action": short,
                "reason": reason,
                "belief_id": belief_id,
            }
        if host is None:
            return
        with host.graph_lock:
            sensors = list(host.sensors)
            world = {
                (a.label or a.id): float(getattr(a, "output", 0.0))
                for a in host.actuators
            }
        # Predictions for this post-fire world (must match next Interface samples)
        predictions: dict[str, float] = {}
        for sen in sensors:
            if sen.transfer is None:
                continue
            try:
                predictions[sen.id] = float(sen.transfer(world))
            except Exception:
                continue
        with self._local_lock:
            for sen in sensors:
                if not sen.can_sample():
                    continue  # exhausted sensors never clear pending
                pred = predictions.get(sen.id)
                # Snapshot prior expectation before overwriting with prediction
                prior = self._prior_reading.get(sen.id)
                bid = self.belief_by_sensor.get(sen.id)
                if prior is None and bid and bid in self.completed_beliefs:
                    exp_o = _expected_observation_from_store(self.completed_beliefs[bid])
                    if exp_o is not None:
                        prior = _reading_from_observation(exp_o)
                self._pending_feedback[sen.id] = {
                    "actuator_id": actuator_id,
                    "action": short,
                    "sensor_label": sen.label,
                    "predicted_reading": pred,
                    "prior_reading": prior,
                    "world_snapshot": dict(world),
                }
        self.predict_expectations_from_world(
            world=world, action=short, predictions=predictions
        )

    def predict_expectations_from_world(
        self,
        *,
        world: dict[str, float],
        action: str = "act",
        predictions: Optional[dict[str, float]] = None,
    ) -> int:
        """
        After Actuator output changes, set each Belief's expected Observation
        from Sensor.transfer(world). Feedback should *confirm* these predictions.
        """
        host = self.host
        if host is None:
            return 0
        with host.graph_lock:
            sensors = list(host.sensors)
        updated = 0
        act = _short_action(action)
        for sen in sensors:
            if sen.transfer is None:
                continue
            if predictions is not None and sen.id in predictions:
                predicted = predictions[sen.id]
            else:
                try:
                    predicted = float(sen.transfer(world))
                except Exception:
                    continue
            sl = sen.label or sen.id
            obs = Thought(
                id=f"{host.id}:pred:{sen.id}:{self.actuator_fires}",
                label=f"{sl}:{predicted:.4f}",
                transient=True,
            )
            with self._local_lock:
                has_belief = sen.id in self.belief_by_sensor
            self.form_belief_from_feedback(
                observation=obs,
                sensor_id=sen.id,
                sensor_label=sl,
                action=act,
                with_labels=True,
                compare=False,
                quiet=has_belief,  # only print first seed; keep log quiet on predict
            )
            updated += 1
        return updated

    def form_belief_from_feedback(
        self,
        *,
        observation: Thought,
        sensor_id: str,
        sensor_label: Optional[str] = None,
        action: Optional[str] = None,
        with_labels: bool = True,
        compare: bool = True,
        quiet: bool = False,
        predicted_reading: Optional[float] = None,
        prior_reading: Optional[float] = None,
    ) -> Optional[dict[str, Thought]]:
        """
        Create or update the single Belief six-set for this sensor channel.

        When compare=True: match actual Feedback to post-fire prediction.
        Stale samples (closer to prior reading than to prediction) are ignored.
        """
        host = self.host
        host_id = host.id if host is not None else "host"
        lab = (lambda s: s) if with_labels else (lambda s: None)
        sl = sensor_label or sensor_id
        act = _short_action(action)
        belief_id = self._belief_id_for_sensor(host_id, sensor_id)

        with self._local_lock:
            existing_id = self.belief_by_sensor.get(sensor_id)
            existing = (
                self.completed_beliefs.get(existing_id)
                if existing_id
                else self.completed_beliefs.get(belief_id)
            )

        if existing is not None:
            bid = existing_id or belief_id
            verdict = "skip"
            if compare:
                verdict = self._challenge_or_confirm(
                    belief_id=bid,
                    store=existing,
                    observation=observation,
                    sensor_label=sl,
                    predicted_reading=predicted_reading,
                    prior_reading=prior_reading,
                )
            if verdict == "stale":
                return existing
            # confirm: keep/set expectation to actual (=prediction)
            # challenge: correct expectation to actual
            result = self._update_belief_expectation(
                belief_id=bid,
                store=existing,
                observation=observation,
                sensor_label=sl,
                action=act,
                with_labels=with_labels,
                quiet=quiet or verdict == "confirm",
            )
            act_val = _reading_from_observation(observation)
            if act_val is not None and verdict in ("confirm", "challenge"):
                with self._local_lock:
                    self._prior_reading[sensor_id] = act_val
            return result

        fb_label = f"Feedback[{sl}]" if act in ("act", "sense", "world") else f"Feedback[{sl}|{act}]"
        feedback = Thought(
            id=f"{belief_id}:feedback",
            label=lab(fb_label),
            transient=False,
        )
        store = complete_belief_set(
            expected_observation=observation,
            feedback=feedback,
            belief_id=belief_id,
            with_labels=with_labels,
        )
        if host is not None:
            with host.graph_lock:
                for tid, t in store.items():
                    host.thoughts[tid] = t
        with self._local_lock:
            self.completed_beliefs[belief_id] = store
            self.active_belief_ids.add(belief_id)
            self.belief_by_sensor[sensor_id] = belief_id
            self.beliefs_created += 1
            self.sets_emitted += 1
            n = self.sets_emitted
        seed_val = _reading_from_observation(observation)
        if seed_val is not None:
            with self._local_lock:
                self._prior_reading[sensor_id] = seed_val
        if not quiet:
            emit_six_set("belief", store, index=n)
        return store

    def _challenge_or_confirm(
        self,
        *,
        belief_id: str,
        store: dict[str, Thought],
        observation: Thought,
        sensor_label: str,
        predicted_reading: Optional[float] = None,
        prior_reading: Optional[float] = None,
    ) -> str:
        """
        Compare actual Feedback to post-fire prediction.

        Returns:
          confirm  — actual matches prediction
          challenge — genuine surprise; correct Belief to actual
          stale — sample is from before this fire; keep prediction
          skip — cannot parse readings
        """
        act_val = _reading_from_observation(observation)
        if act_val is None:
            return "skip"
        with self._local_lock:
            tol = self.expectation_tolerance

        if predicted_reading is not None:
            err_pred = abs(act_val - predicted_reading)
            if err_pred <= tol:
                with self._local_lock:
                    self.belief_confirms += 1
                print(
                    f"[confirm] Feedback[{sensor_label}] expected {predicted_reading:.4f} "
                    f"got {act_val:.4f} (err={err_pred:.4f})",
                    flush=True,
                )
                return "confirm"
            # Stale: actual closer to pre-fire prior than to post-fire prediction
            if prior_reading is not None and abs(act_val - prior_reading) + 1e-12 < err_pred:
                with self._local_lock:
                    self.belief_stale_skips += 1
                print(
                    f"[stale] Feedback[{sensor_label}] got {act_val:.4f} "
                    f"(prior {prior_reading:.4f}; predicted {predicted_reading:.4f}) — keep prediction",
                    flush=True,
                )
                return "stale"
            with self._local_lock:
                self.belief_challenges += 1
            print(
                f"[challenge] Feedback[{sensor_label}] expected {predicted_reading:.4f} "
                f"got {act_val:.4f} (err={err_pred:.4f}) → correct expectation",
                flush=True,
            )
            return "challenge"

        # No prediction: compare to current Belief expectation
        expected_obs = _expected_observation_from_store(store)
        exp_val = _reading_from_observation(expected_obs) if expected_obs else None
        if exp_val is None:
            return "skip"
        err = abs(act_val - exp_val)
        if err <= tol:
            with self._local_lock:
                self.belief_confirms += 1
            print(
                f"[confirm] Feedback[{sensor_label}] expected {exp_val:.4f} "
                f"got {act_val:.4f} (err={err:.4f})",
                flush=True,
            )
            return "confirm"
        with self._local_lock:
            self.belief_challenges += 1
        print(
            f"[challenge] Feedback[{sensor_label}] expected {exp_val:.4f} "
            f"got {act_val:.4f} (err={err:.4f}) → correct expectation",
            flush=True,
        )
        return "challenge"

    def _update_belief_expectation(
        self,
        *,
        belief_id: str,
        store: dict[str, Thought],
        observation: Thought,
        sensor_label: str,
        action: str,
        with_labels: bool,
        quiet: bool = False,
    ) -> dict[str, Thought]:
        """Revise expected Observation for an existing Belief; keep one six-set."""
        host = self.host
        lab = (lambda s: s) if with_labels else (lambda s: None)

        feedback: Optional[Thought] = None
        for t in store.values():
            if isinstance(t, Link):
                continue
            if t.label and str(t.label).startswith("Feedback["):
                feedback = t
                break
        if feedback is None:
            poles = six_set_poles(store)
            feedback = poles[0] if poles else None
        if feedback is None:
            return store

        if action not in ("act", "sense", "world"):
            feedback.label = lab(f"Feedback[{sensor_label}|{action}]")
        else:
            feedback.label = lab(f"Feedback[{sensor_label}]")

        new_store = complete_belief_set(
            expected_observation=observation,
            feedback=feedback,
            belief_id=belief_id,
            with_labels=with_labels,
        )
        if host is not None:
            with host.graph_lock:
                prefix = f"{belief_id}:"
                drop = [
                    tid
                    for tid in list(host.thoughts)
                    if tid.startswith(prefix) and tid != feedback.id
                ]
                for tid in drop:
                    host.thoughts.pop(tid, None)
                for tid, t in new_store.items():
                    host.thoughts[tid] = t
        with self._local_lock:
            self.completed_beliefs[belief_id] = new_store
            self.active_belief_ids.add(belief_id)
            self.beliefs_updated += 1
            self.sets_emitted += 1
            n = self.sets_emitted
        if not quiet:
            emit_six_set("belief", new_store, index=n)
        return new_store

    def handle_interface_observation(self, msg: dict[str, Any]) -> Optional[dict[str, Thought]]:
        """
        Feedback Observation → compare to post-fire prediction / Belief expectation.
        Stale pre-fire samples must not overwrite a correct prediction.
        """
        obs = msg.get("observation")
        sensor_id = str(msg.get("sensor_id") or "")
        if not isinstance(obs, Thought) or not sensor_id:
            return None

        with self._local_lock:
            pending = self._pending_feedback.get(sensor_id)
            already = sensor_id in self.belief_by_sensor

        if pending is None and already:
            return None
        if pending is None and not already:
            action = "sense"
            compare = False
            predicted = None
            result = self.form_belief_from_feedback(
                observation=obs,
                sensor_id=sensor_id,
                sensor_label=msg.get("sensor_label"),
                action=action,
                with_labels=True,
                compare=False,
            )
            return result

        action = (pending or {}).get("action") or "act"
        predicted = (pending or {}).get("predicted_reading")
        prior = (pending or {}).get("prior_reading")
        if predicted is not None:
            predicted = float(predicted)
        if prior is not None:
            prior = float(prior)

        result = self.form_belief_from_feedback(
            observation=obs,
            sensor_id=sensor_id,
            sensor_label=msg.get("sensor_label") or (pending or {}).get("sensor_label"),
            action=str(action),
            with_labels=True,
            compare=True,
            predicted_reading=predicted,
            prior_reading=prior,
        )
        # Consume pending only on confirm/challenge; keep on stale for real sample
        act_val = _reading_from_observation(obs)
        with self._local_lock:
            if pending is None:
                return result
            if predicted is not None and act_val is not None:
                tol = self.expectation_tolerance
                if abs(act_val - predicted) <= tol:
                    self._pending_feedback.pop(sensor_id, None)
                elif prior is not None and abs(act_val - prior) + 1e-12 < abs(
                    act_val - predicted
                ):
                    pass  # stale — keep pending
                else:
                    self._pending_feedback.pop(sensor_id, None)  # challenge consumed
            else:
                self._pending_feedback.pop(sensor_id, None)
        return result

    def _drop_pending_for_exhausted_sensors(self) -> None:
        """Avoid deadlock: sensors at max_samples never deliver Feedback."""
        host = self.host
        if host is None:
            return
        with host.graph_lock:
            exhausted = {sen.id for sen in host.sensors if not sen.can_sample()}
        if not exhausted:
            return
        with self._local_lock:
            for sid in list(self._pending_feedback):
                if sid in exhausted:
                    self._pending_feedback.pop(sid, None)

    def propose_actions_from_beliefs(self) -> list[tuple[bool, str]]:
        """
        Closed-loop agency: fire at most once while no Feedback is pending.

        Prevents hand racing ahead of Interface samples (which caused inverted
        challenge: expected next value, got previous sample).
        """
        host = self.host
        if host is None:
            return []
        self._drop_pending_for_exhausted_sensors()
        with self._local_lock:
            if self.wait_for_feedback and self._pending_feedback:
                return []  # wait for Feedback before next Action
            belief_ids = list(self.active_belief_ids)
        if not belief_ids:
            return []
        with host.graph_lock:
            actuators = list(host.actuators)
        if not actuators:
            return []

        results: list[tuple[bool, str]] = []
        # One fire per Outerface tick (first actuator), not one per belief×actuator
        act = actuators[0]
        action = act.label or act.id or "act"
        ok, reason = act.request_fire(
            host, action, belief_id=belief_ids[0] if belief_ids else None
        )
        results.append((ok, reason))
        with self._local_lock:
            self.last_gate = reason
            if not ok:
                self.actuator_denies += 1
        return results

    def _process_body(self) -> None:
        """Agency tick: Feedback→Belief six-sets; Action proposals gated by laws."""
        messages = self._drain_inbox()
        host = self.host
        if host is None:
            return

        proposals: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            kind = msg.get("kind")
            if kind == "interface_observation":
                try:
                    self.handle_interface_observation(msg)
                except (ValueError, RuntimeError, TypeError) as exc:
                    self.last_error = f"belief: {exc}"
            elif kind == "action_proposal":
                proposals.append(msg)

        for msg in proposals:
            allowed, reason = self.check_action(
                host,
                threatens_twin_integrity=bool(msg.get("threatens_twin_integrity")),
                harms_protected_environment=bool(msg.get("harms_protected_environment")),
                is_order=bool(msg.get("is_order")),
                order_from_authority=bool(msg.get("order_from_authority")),
                preserves_self=bool(msg.get("preserves_self")),
                self_preservation_conflicts_higher=bool(
                    msg.get("self_preservation_conflicts_higher")
                ),
            )
            with self._local_lock:
                self.last_gate = reason
            if allowed:
                act_id = msg.get("actuator_id")
                action = _short_action(str(msg.get("action") or "execute"))
                with host.graph_lock:
                    acts = list(host.actuators)
                for act in acts:
                    if act_id and act.id != act_id:
                        continue
                    flags = {
                        k: msg[k]
                        for k in (
                            "threatens_twin_integrity",
                            "harms_protected_environment",
                            "is_order",
                            "order_from_authority",
                            "preserves_self",
                            "self_preservation_conflicts_higher",
                        )
                        if k in msg
                    }
                    act.request_fire(host, action, **flags)
            else:
                with self._local_lock:
                    self.actuator_denies += 1

        if self.active_belief_ids and host.actuators:
            self.propose_actions_from_beliefs()

        with self._local_lock:
            self.agency_ticks += 1

    def check_action(
        self,
        sym: "Symbioid",
        *,
        threatens_twin_integrity: bool = False,
        harms_protected_environment: bool = False,
        is_order: bool = False,
        order_from_authority: bool = False,
        preserves_self: bool = False,
        self_preservation_conflicts_higher: bool = False,
    ) -> tuple[bool, str]:
        """Apply installed constitution (priority L0 > L1 > L2 > L3)."""
        if not self.enabled:
            return True, "outerface_disabled"

        with sym.graph_lock:
            has_laws = bool(sym.laws)

        if not has_laws:
            return True, "no_constitution"

        if threatens_twin_integrity:
            return False, "L0_deny_twin_integrity"
        if harms_protected_environment:
            return False, "L1_deny_harm_protected_env"
        if is_order and order_from_authority:
            return True, "L2_obey_authority"
        if is_order and not order_from_authority:
            return False, "L2_deny_unauthorized_order"
        if preserves_self:
            if self_preservation_conflicts_higher:
                return False, "L3_deny_self_vs_higher"
            return True, "L3_allow_self_preserve"
        return True, "default_allow"
