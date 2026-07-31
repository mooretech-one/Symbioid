"""Tetris learner: secret bytes + experiential drop-effect learning (no drop oracle)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from symbioid.world.tetris import VALID_ACTIONS, TetrisWorld, piece_cells, well_metrics


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# Observable effect labels inferred from world deltas (not from the cipher).
EFFECTS = ("left", "right", "rotate", "hard", "noop", "blocked", "unknown")


@dataclass
class WorldSnapshot:
    """Minimal pre-step state for effect classification."""

    pieces_placed: int
    score: int
    game_over: bool
    has_active: bool
    col: int
    row: int
    rotation: int
    kind: str

    @classmethod
    def take(cls, world: TetrisWorld) -> "WorldSnapshot":
        a = world.active
        return cls(
            pieces_placed=world.pieces_placed,
            score=world.score,
            game_over=world.game_over,
            has_active=a is not None,
            col=a.col if a else 0,
            row=a.row if a else 0,
            rotation=a.rotation if a else 0,
            kind=a.kind if a else "",
        )


# Line-clear point table (must match tetris._LINE_SCORES) for lock attribution.
_LINE_POINTS = (0, 100, 300, 500, 800)


def classify_effect(before: WorldSnapshot, world: TetrisWorld) -> str:
    """
    Infer what the last byte did from world change alone.

    Does not use cipher / last_byte_action — only physics + score/pieces.

    Important:
      - Horizontal move wins even if row also changed (old bug: gravity mixed
        into the same tick made left/right look like ``unknown``).
      - A lock is ``hard`` only if score gained hard-drop fall points; pure
        gravity locks must not credit the last byte as hard.
    """
    locked = world.pieces_placed > before.pieces_placed or (
        world.game_over and not before.game_over and before.has_active
    )

    if locked:
        # Attribute lock type from score: hard_drop adds 2*rows; gravity adds 0.
        score_delta = float(world.score) - float(before.score)
        lines = int(getattr(world, "last_lines_cleared", 0) or 0)
        line_pts = _LINE_POINTS[lines] if 0 <= lines < len(_LINE_POINTS) else 0
        fall_pts = score_delta - line_pts
        if fall_pts >= 2:
            return "hard"
        # Soft-drop lock (1 pt/row) or gravity lock — do not map byte as hard.
        if fall_pts >= 1:
            return "soft"
        return "noop"

    if not before.has_active:
        return "noop"

    # Still controlling the same falling piece
    if world.active is None:
        return "noop"
    if world.active.kind != before.kind:
        return "unknown"

    dcol = world.active.col - before.col
    drow = world.active.row - before.row
    drot = (world.active.rotation - before.rotation) % 4

    # Rotation first (wall-kicks may also shift col)
    if drot != 0:
        return "rotate"
    # Horizontal intent even if gravity also nudged row in a combined step
    if dcol < 0:
        return "left"
    if dcol > 0:
        return "right"
    if drow != 0:
        return "noop"  # pure gravity / soft without lock
    return "noop"


def pose_hole_features(
    world: TetrisWorld,
    rot: int,
    col: int,
    *,
    pre_holes: float | None = None,
    pre_wm: dict[str, float] | None = None,
    field: list | None = None,
) -> dict[str, float]:
    """
    Counterfactual hole signal for a candidate landing (rot, col).

    Uses locked-board hole count + ``simulate_placement`` (no cipher).
    Also counts how many **existing** holes (cell reading 0.5) the landing
    would cover.

    Optional ``pre_holes`` / ``pre_wm`` / ``field`` avoid recomputing board
    globals when scoring many poses (Phase 3 batch).

    Returns keys:
      d_holes      — post.holes − pre.holes (positive = created/net more holes)
      holes_filled — landing cells that are currently holes
      pre_holes, post_holes
      ok           — 1.0 if sim legal, else 0.0
    """
    if pre_holes is None:
        pre_holes = float(world.hole_count())
    else:
        pre_holes = float(pre_holes)
    if pre_wm is None:
        pre_wm = well_metrics(world.column_heights())
    pre_well = float(pre_wm["well"])
    pre_max_well = float(pre_wm["max_well"])
    cells = world.landing_cells(rot, col)
    if not cells:
        return {
            "d_holes": 0.0,
            "holes_filled": 0.0,
            "pre_holes": pre_holes,
            "post_holes": pre_holes,
            "d_well": 0.0,
            "d_max_well": 0.0,
            "pre_well": pre_well,
            "post_well": pre_well,
            "pre_max_well": pre_max_well,
            "post_max_well": pre_max_well,
            "ok": 0.0,
        }
    if field is None:
        field = world.cell_field_state(with_active=False)
    holes_filled = 0.0
    for r, c in cells:
        if 0 <= r < world.rows and 0 <= c < world.cols:
            if abs(float(field[r][c]) - 0.5) < 0.05:
                holes_filled += 1.0
    sim = world.simulate_placement(rot, col)
    if sim is None:
        return {
            "d_holes": 0.0,
            "holes_filled": holes_filled,
            "pre_holes": pre_holes,
            "post_holes": pre_holes,
            "d_well": 0.0,
            "d_max_well": 0.0,
            "pre_well": pre_well,
            "post_well": pre_well,
            "pre_max_well": pre_max_well,
            "post_max_well": pre_max_well,
            "ok": 0.0,
        }
    post_holes = float(sim.get("holes", pre_holes))
    # Prefer sim well metrics when present; else recompute from heights
    if "well" in sim and "max_well" in sim:
        post_well = float(sim["well"])
        post_max_well = float(sim["max_well"])
    else:
        hs = sim.get("heights") or world.column_heights()
        pwm = well_metrics([int(x) for x in hs])
        post_well = float(pwm["well"])
        post_max_well = float(pwm["max_well"])
    return {
        "d_holes": post_holes - pre_holes,
        "holes_filled": holes_filled,
        "pre_holes": pre_holes,
        "post_holes": post_holes,
        "d_well": post_well - pre_well,
        "d_max_well": post_max_well - pre_max_well,
        "pre_well": pre_well,
        "post_well": post_well,
        "pre_max_well": pre_max_well,
        "post_max_well": post_max_well,
        "ok": 1.0,
    }


def observe_board(world: TetrisWorld) -> dict[str, float]:
    """
    Sensors the agent may use — locked board only, no drop simulation.

    Includes per-column heights and fill so height-control can be learned.
    """
    heights = world.column_heights()
    rows = float(world.rows) or 1.0
    cols = float(world.cols) or 1.0
    max_h = float(max(heights) if heights else 0)
    min_h = float(min(heights) if heights else 0)
    agg = float(sum(heights))
    # Well: open single-column dips (edge-aware — see well_metrics)
    wm = well_metrics(heights)
    filled = sum(1 for row in world.board for c in row if c)
    out: dict[str, float] = {
        "holes": float(world.hole_count()),
        "bumpiness": float(world.bumpiness()),
        "agg_height": agg,
        "max_height": max_h,
        "min_height": min_h,
        "height_range": max_h - min_h,
        "well": float(wm["well"]),
        "max_well": float(wm["max_well"]),
        "filled": float(filled),
        "fill_n": filled / (rows * cols),
        "max_height_n": max_h / rows,
        "agg_height_n": agg / (rows * cols),
        "lines": float(world.lines),
        "score": float(world.score),
    }
    for i, h in enumerate(heights):
        out[f"h{i}"] = float(h)
        out[f"h{i}_n"] = float(h) / rows
    return out


def board_quality_reward(
    pre: dict[str, float],
    post: dict[str, float],
    *,
    lines_cleared: int,
    topped_out: bool,
    score_delta: float,
    rows: float = 20.0,
) -> float:
    """
    Shaped reward from **observed** board change + sparse score.

    Classic Tetris score alone does not teach "keep the stack low":
    hard-drop fall points dominate and ignore skyline quality. This mixes
    line clears / survival with strong height & hole penalties.
    """
    r = 0.0
    # Primary goals
    r += 20.0 * float(lines_cleared)
    if topped_out:
        r -= 100.0

    # Absolute skyline (always prefer lower)
    r -= 4.0 * post.get("max_height", 0.0)
    r -= 0.25 * post.get("agg_height", 0.0)
    r -= 3.0 * post.get("holes", 0.0)
    r -= 0.35 * post.get("bumpiness", 0.0)
    # Open wells (incl. edge single-width trenches) — stronger than legacy 0.5
    r -= 2.0 * post.get("well", 0.0)
    r -= 1.5 * post.get("max_well", 0.0)
    r -= 0.4 * post.get("height_range", 0.0)

    # Deltas: punish growing the stack / digging holes / deepening wells
    d_max = post.get("max_height", 0.0) - pre.get("max_height", 0.0)
    d_agg = post.get("agg_height", 0.0) - pre.get("agg_height", 0.0)
    d_holes = post.get("holes", 0.0) - pre.get("holes", 0.0)
    d_well = post.get("well", 0.0) - pre.get("well", 0.0)
    d_max_well = post.get("max_well", 0.0) - pre.get("max_well", 0.0)
    r -= 2.5 * max(0.0, d_max)
    r -= 0.15 * max(0.0, d_agg)
    r -= 4.0 * max(0.0, d_holes)
    r -= 2.5 * max(0.0, d_well)
    r -= 2.0 * max(0.0, d_max_well)
    # Bonus for packing: lower max height, fill holes, fill wells
    r += 3.0 * max(0.0, -d_max)
    r += 1.5 * max(0.0, -d_well)
    r += 1.0 * max(0.0, -d_max_well)

    # Mild game-score signal (lines already counted strongly; fall points weak)
    r += 0.02 * score_delta

    # Near-death pressure: when already tall, extra height hurts more
    danger = post.get("max_height", 0.0) / max(1.0, rows)
    if danger > 0.5:
        r -= 8.0 * (danger - 0.5) * post.get("max_height", 0.0)

    return r


@dataclass
class DropExperience:
    """One real lock: what we did, what the board was, what we got."""

    kind: str
    rot: int
    col: int
    pre: dict[str, float]
    post: dict[str, float]
    reward: float
    lines_cleared: int
    topped_out: bool


@dataclass
class TetrisCoach:
    """
    Control layers (no privileged cipher oracle):

    1. **Byte map** — emit 0..255; observe effects; learn live bytes (coach).
    2. **Placement / strategy** — with ``network_primary`` (default), 1-ply
       ``choose_target`` is dominated by Symbioid ``graph_placement_bonus``
       (cell-map / Thought heat); coach board evaluator is residual only.
    3. **Commands** — ``tick(preferred_intent=…)`` takes Symbioid intents first
       (Mind ``recommend_action`` + geo-from-network-target); ``select_byte``
       is cold fallback / mapping explore.

    Coach retains: cipher discovery, gravity separation, stuck/force-hard
    survival, and residual board shaping when the graph is cold.

    The coach never reads ``world.cipher`` / ``last_byte_action``.
    """

    # Learned scalar value of *observed* post-drop board (from real locks only).
    # Strong negative height priors — score alone does not teach low stacks.
    w_lines: float = 2.0
    w_holes: float = -3.0
    w_bump: float = -0.4
    w_agg: float = -0.3
    w_max_h: float = -4.0
    w_well: float = -2.0
    w_max_well: float = -1.5
    w_range: float = -0.4
    # Blend: predicted slot value vs scored post-features after the fact
    lr: float = 0.05
    noise: float = 0.35
    # How much to trust feature-scored board vs slot average (0..1)
    board_value_blend: float = 0.55
    # When True (default): Symbioid drives placement scores + preferred intents;
    # coach supplies byte map, gravity/world model, and cold fallback.
    network_primary: bool = True
    # Weight for Symbioid placement bonus in choose_target (higher = more network).
    # Demo uses ~0.60 co-lead; net-primary clamp floor is 0.35.
    graph_placement_weight: float = 0.60
    graph_placement_bonus: Optional[
        Callable[[TetrisWorld, int, int], float]
    ] = field(default=None, repr=False)
    # Optional Phase 3: prepare(world, options) before per-pose bonus lookups
    placements: int = 0
    lines_total: int = 0
    games: int = 0
    top_outs: int = 0
    game_number: int = 1
    highscores: list[tuple[int, int]] = field(default_factory=list)
    last_choice: tuple[int, int] = (0, 3)
    last_score: float = 0.0
    last_graph_bonus: float = 0.0
    # Cells occupied by the piece at last lock (for Mind placement valence)
    last_lock_cells: list[tuple[int, int]] = field(default_factory=list)
    last_lock_pose: Optional[tuple[str, int, int]] = None  # kind, rot, col
    # Hole delta of last real lock (post.holes − pre.holes) for network sensors
    last_d_holes: float = 0.0
    last_post_holes: float = 0.0
    last_features: dict[str, float] = field(default_factory=dict)
    last_byte: int = 0
    last_effect: str = "noop"
    last_intent: str = "explore"
    last_network_cmd: bool = False  # True when last tick used preferred_intent
    last_reward: float = 0.0
    # byte → effect → count
    effect_counts: dict[int, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    bytes_tried: set[int] = field(default_factory=set)
    explore_rate: float = 0.55
    # One clean observation is enough; rescans confirm dominance
    map_threshold: int = 1
    # Placement exploration among near-best simulated poses
    place_explore: float = 0.18
    place_explore_floor: float = 0.12
    # (kind, rot, col) → (reward_sum, count) from real locks only
    slot_reward_sum: dict[tuple[str, int, int], float] = field(default_factory=dict)
    slot_reward_n: dict[tuple[str, int, int], int] = field(default_factory=dict)
    # Average *post* skyline after locking this pose (for height-aware choice)
    slot_post_max_h_sum: dict[tuple[str, int, int], float] = field(default_factory=dict)
    slot_post_holes_sum: dict[tuple[str, int, int], float] = field(default_factory=dict)
    # Optional height-bucket conditioning: (kind, rot, col, h_bucket) → …
    slot_h_sum: dict[tuple[str, int, int, int], float] = field(default_factory=dict)
    slot_h_n: dict[tuple[str, int, int, int], int] = field(default_factory=dict)
    experiences: list[DropExperience] = field(default_factory=list)
    max_experiences: int = 400
    # Systematic scan cursor (0..255) — guarantees every byte is re-probed
    _scan_byte: int = 0
    _scan_passes: int = 0
    _target: Optional[tuple[int, int]] = None
    _pre_board: Optional[dict[str, float]] = None
    _score_at_piece_start: float = 0.0
    _piece_kind: str = ""
    # Consecutive left/right commands that did not change col → then replan/hard
    _stuck_lateral: int = 0
    # Phase A: mid-piece replan / force-hard
    _piece_cmds: int = 0
    _replans_this_piece: int = 0
    replan_every_cmds: int = 10
    force_hard_after_cmds: int = 36
    # Last lock's true action effect (for Mind outcomes; not "explore")
    last_lock_effect: str = "noop"
    rng: random.Random = field(default_factory=random.Random)

    # ------------------------------------------------------------------ map

    def observed_effect_for_byte(self, code: int) -> Optional[str]:
        """
        Dominant *action* for this byte, if reliable.

        Only VALID_ACTIONS count. Noops/blocks at walls must NOT unbind a key
        (old bug: using RIGHT at the right wall flooded noop counts and wiped
        the map, then rediscovery/play looked stuck on RIGHT).
        """
        counts = self.effect_counts.get(int(code) & 0xFF)
        if not counts:
            return None
        ranked = sorted(
            ((e, n) for e, n in counts.items() if e in VALID_ACTIONS),
            key=lambda t: t[1],
            reverse=True,
        )
        if ranked and ranked[0][1] >= self.map_threshold:
            return ranked[0][0]
        return None

    def byte_for_effect(self, effect: str) -> Optional[int]:
        best_b: Optional[int] = None
        best_n = 0
        for b, counts in self.effect_counts.items():
            n = counts.get(effect, 0)
            if n < self.map_threshold:
                continue
            if self.observed_effect_for_byte(b) == effect and n > best_n:
                best_n = n
                best_b = b
        return best_b

    def discovered_map(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for eff in VALID_ACTIONS:
            b = self.byte_for_effect(eff)
            if b is not None:
                out[eff] = b
        return out

    def map_complete(self) -> bool:
        return all(self.byte_for_effect(e) is not None for e in VALID_ACTIONS)

    def play_ready(self) -> bool:
        """
        Phase A: enough map to aim — hard + left + right.

        Rotate may still be missing (then rotate intent falls back to explore).
        Does not require full map_complete().
        """
        return (
            self.byte_for_effect("hard") is not None
            and self.byte_for_effect("left") is not None
            and self.byte_for_effect("right") is not None
        )

    def wants_hard_now(self, world: TetrisWorld) -> bool:
        """
        Phase B: coach situation where hard-drop is the right micro-intent
        (aligned on target, stuck laterally, or force-hard timeout).
        """
        if not self.play_ready() and not self.map_complete():
            return False
        if self.byte_for_effect("hard") is None:
            return False
        if world.active is None or world.game_over:
            return False
        if self._piece_cmds >= int(self.force_hard_after_cmds):
            return True
        if self._stuck_lateral >= 3:
            return True
        # Aligned with target pose → hard
        if self._target is not None:
            tr, tc = self._target
            tr, tc = self._clamp_pose(world.active.kind, tr, tc, world.cols)
            if (
                world.active.rotation % 4 == tr % 4
                and world.active.col == tc
            ):
                return True
        return False

    def map_progress(self) -> str:
        m = self.discovered_map()
        parts = [f"{e}=0x{m[e]:02X}" if e in m else f"{e}=?" for e in VALID_ACTIONS]
        return " ".join(parts)

    def missing_effects(self) -> list[str]:
        return [e for e in VALID_ACTIONS if self.byte_for_effect(e) is None]

    def _note_effect(self, code: int, effect: str) -> None:
        b = int(code) & 0xFF
        self.bytes_tried.add(b)
        if effect in VALID_ACTIONS:
            self.effect_counts[b][effect] += 1
            return
        # soft / blocked / noop / unknown: do not erase a known binding.
        # Only record noop while this byte has no action evidence yet (helps
        # deprioritize dead keys during systematic scan).
        if self.observed_effect_for_byte(b) is not None:
            return
        if effect in ("noop", "soft", "blocked", "unknown"):
            self.effect_counts[b]["noop"] += 1

    def _pick_explore_byte(self) -> int:
        """
        Discovery policy (map incomplete):

        1. Prefer untried bytes.
        2. **Hard-hunt bias**: if hard unknown, prefer unbound / noop-only bytes.
        3. Else systematic scan 0→255→0… so every code is re-probed.
        4. Occasionally re-test bytes that showed a *hint* of a missing action.
        """
        untried = [i for i in range(256) if i not in self.bytes_tried]
        if untried:
            return self.rng.choice(untried)

        missing = set(self.missing_effects())
        # Phase A: hard-hunt — walk scan among unbound bytes (covers all, prefers
        # codes with no action binding yet). Falls through to full scan if none.
        if "hard" in missing and self.rng.random() < 0.7:
            unbound = {
                b
                for b in range(256)
                if self.observed_effect_for_byte(b) is None
            }
            if unbound:
                for _ in range(256):
                    b = self._scan_byte & 0xFF
                    self._scan_byte = (self._scan_byte + 1) & 0xFF
                    if b == 0:
                        self._scan_passes += 1
                    if b in unbound:
                        return b

        # Re-probe candidates that ever produced a still-missing action
        hints: list[int] = []
        if missing:
            for b, counts in self.effect_counts.items():
                for e in missing:
                    if counts.get(e, 0) > 0:
                        hints.append(b)
                        break
        if hints and self.rng.random() < 0.25:
            return self.rng.choice(hints)

        # Systematic full rescan — guarantees progress after 256-tried stall
        b = self._scan_byte & 0xFF
        self._scan_byte = (self._scan_byte + 1) & 0xFF
        if b == 0:
            self._scan_passes += 1
        return b

    # ------------------------------------------------ drop model (learned)

    @staticmethod
    def _h_bucket(max_height: float, rows: int = 20) -> int:
        """Coarse board-height context for slot values."""
        return int(_clamp(max_height / max(1.0, rows / 5.0), 0, 4))

    def score_observed_board(
        self, post: dict[str, float], *, lines_cleared: float = 0.0
    ) -> float:
        """
        Scalar quality of an *observed* board after a drop.

        Weights start with strong low-height priors and are refined online.
        """
        return (
            self.w_lines * lines_cleared
            + self.w_holes * post.get("holes", 0.0)
            + self.w_bump * post.get("bumpiness", 0.0)
            + self.w_agg * post.get("agg_height", 0.0)
            + self.w_max_h * post.get("max_height", 0.0)
            + self.w_well * post.get("well", 0.0)
            + self.w_max_well * post.get("max_well", 0.0)
            + self.w_range * post.get("height_range", 0.0)
        )

    def predict_drop_value(
        self,
        kind: str,
        rot: int,
        col: int,
        pre: dict[str, float],
        *,
        cols: int = 10,
        rows: int = 20,
    ) -> float:
        """
        Predict value of locking this pose — **only** from past real drops.

        Blends:
          - average shaped reward for (kind, rot, col)
          - expected post height/holes from memory (height-aware)
          - feature prior: lower current skyline pressure prefers safer packs
        No call to ``simulate_placement``.
        """
        rot = int(rot) % 4
        col = int(_clamp(col, 0, cols - 1))
        hb = self._h_bucket(pre.get("max_height", 0.0), rows)
        key_h = (kind, rot, col, hb)
        key = (kind, rot, col)
        n = self.slot_reward_n.get(key, 0)

        if self.slot_h_n.get(key_h, 0) > 0:
            slot_v = self.slot_h_sum[key_h] / self.slot_h_n[key_h]
        elif n > 0:
            slot_v = self.slot_reward_sum[key] / n
        else:
            same_kc = [
                self.slot_reward_sum[k] / self.slot_reward_n[k]
                for k in self.slot_reward_n
                if k[0] == kind and k[2] == col and self.slot_reward_n[k] > 0
            ]
            if same_kc:
                slot_v = sum(same_kc) / len(same_kc)
            else:
                same_k = [
                    self.slot_reward_sum[k] / self.slot_reward_n[k]
                    for k in self.slot_reward_n
                    if k[0] == kind and self.slot_reward_n[k] > 0
                ]
                slot_v = sum(same_k) / len(same_k) if same_k else 0.0

        # Memory of how tall/holey this pose left the board
        height_term = 0.0
        if n > 0:
            avg_max = self.slot_post_max_h_sum.get(key, 0.0) / n
            avg_holes = self.slot_post_holes_sum.get(key, 0.0) / n
            height_term = self.w_max_h * avg_max + self.w_holes * avg_holes
        else:
            # No memory: mild prior — prefer mid columns, avoid stacking when tall
            mid = (cols - 1) / 2.0
            height_term = 0.1 * (1.0 - abs(col - mid) / max(1.0, mid))
            height_term -= 0.8 * pre.get("max_height_n", pre.get("max_height", 0.0) / rows)

        # When pre-stack is already high, overweight height memory
        danger = pre.get("max_height_n", pre.get("max_height", 0.0) / max(1.0, rows))
        blend = self.board_value_blend + 0.25 * danger
        blend = _clamp(blend, 0.2, 0.85)
        return (1.0 - blend) * slot_v + blend * height_term

    @staticmethod
    def col_range_for_pose(kind: str, rot: int, cols: int) -> tuple[int, int]:
        """
        Inclusive [lo, hi] for piece **origin** col so every filled cell is
        on-board.

        Origin may be **negative** when the shape's min column offset is > 0
        (e.g. O uses dc∈{1,2}; vertical I uses dc=2). Forcing lo≥0 was an
        off-by-one that made board column 0 unreachable for those poses.
        Matches ``TetrisWorld.legal_placements``: ``range(-min_dc, cols-max_dc)``.
        """
        cells = piece_cells(kind, rot % 4)
        min_dc = min(dc for _, dc in cells)
        max_dc = max(dc for _, dc in cells)
        lo = -min_dc  # e.g. O → -1, vertical I → -2
        hi = cols - 1 - max_dc
        if lo > hi:
            return 0, max(0, cols - 1)
        return lo, hi

    def _clamp_pose(self, kind: str, rot: int, col: int, cols: int) -> tuple[int, int]:
        rot = int(rot) % 4
        lo, hi = self.col_range_for_pose(kind, rot, cols)
        return rot, int(_clamp(col, lo, hi))

    def _sample_explore_col(
        self, cur_col: int, cols: int, *, lo: int, hi: int
    ) -> int:
        """
        Sample target column with balanced left/right step counts, inside [lo, hi].
        """
        cur = int(_clamp(cur_col, lo, hi))
        if lo >= hi:
            return lo
        direction = -1 if self.rng.random() < 0.5 else 1
        max_steps = max(1, (hi - lo + 1) // 2)
        steps = self.rng.randint(0, max_steps)
        return int(_clamp(cur + direction * steps, lo, hi))

    def evaluate_imagined_drop(
        self,
        pre: dict[str, float],
        sim: dict,
        *,
        rows: float = 20.0,
    ) -> float:
        """
        Score a one-step imagined lock (from ``simulate_placement``).

        Uses the same height-shaped objective as real learning, plus learned
        feature weights. This is planning against world physics — not a fixed
        hand-tuned oracle of "good Tetris"; weights still adapt online.
        """
        post = {k: float(v) for k, v in sim.items() if k != "heights"}
        lines = int(sim.get("lines_cleared", 0))
        shaped = board_quality_reward(
            pre,
            post,
            lines_cleared=lines,
            topped_out=False,
            score_delta=0.0,
            rows=rows,
        )
        weighted = self.score_observed_board(post, lines_cleared=float(lines))
        # Prefer line clears hard
        return 0.55 * shaped + 0.45 * weighted + 8.0 * float(lines)

    def choose_target(self, world: TetrisWorld) -> tuple[int, int]:
        """
        Pick (rotation, column) by 1-ply search:

        For each legal landing, simulate the lock on a copy of the board and
        score the resulting skyline with the learned height-aware evaluator
        plus optional Phase C graph/cell-map bonus.

        Pure (kind,rot,col) averages cannot learn packing — the same pose is
        good or bad depending on the board. Simulation is the world model for
        planning; the *value* of boards is still learned from real games.
        """
        assert world.active is not None
        kind = world.active.kind
        pre = observe_board(world)
        cols = world.cols
        cur_col = world.active.col

        options = world.legal_placements()
        if not options:
            return self._clamp_pose(kind, world.active.rotation, cur_col, cols)

        # Phase 3: batch graph bonus prepare (shared Mind/field locks)
        bonus_fn = self.graph_placement_bonus
        if bonus_fn is not None and hasattr(bonus_fn, "prepare"):
            try:
                bonus_fn.prepare(world, options)  # type: ignore[union-attr]
            except Exception:
                pass

        scored: list[tuple[float, int, int, float]] = []
        net_primary = bool(self.network_primary) and self.graph_placement_bonus is not None
        g_w = float(self.graph_placement_weight)
        if net_primary:
            # Allow coach residual co-lead (was ≥0.55 floor; research 2026-07-26)
            g_w = max(0.35, min(0.98, g_w))
        for rot, col in options:
            sim = world.simulate_placement(rot, col)
            if sim is None:
                continue
            coach_v = self.evaluate_imagined_drop(pre, sim, rows=float(world.rows))
            # Small residual from real-game outcomes for this pose
            key = (kind, rot % 4, int(col))
            n = self.slot_reward_n.get(key, 0)
            if n > 0:
                coach_v += 0.12 * (self.slot_reward_sum[key] / n)
            # Slight preference for nearer columns (less control error)
            coach_v -= 0.05 * abs(col - cur_col)
            g_bonus = 0.0
            if self.graph_placement_bonus is not None:
                try:
                    g_bonus = float(self.graph_placement_bonus(world, rot, col))
                except Exception:
                    g_bonus = 0.0
            if net_primary:
                # Symbioid placement heat is primary; coach board value is residual
                v = g_w * g_bonus + (1.0 - g_w) * coach_v
            else:
                v = coach_v + g_w * g_bonus
            scored.append((v, rot, col, g_bonus))

        if not scored:
            return self._clamp_pose(kind, world.active.rotation, cur_col, cols)

        scored.sort(key=lambda t: t[0], reverse=True)
        best_v, best_rot, best_col, best_g = scored[0]

        # Explore among near-best (or broader set) to escape local pose minima
        if self.rng.random() < self.place_explore and len(scored) > 1:
            # Phase A: sometimes sample top-8 / half of options, not only top-4
            k = min(len(scored), 8 if self.rng.random() < 0.35 else 4)
            top_k = scored[:k]
            m = top_k[0][0]
            exps = [math.exp((v - m) / max(0.5, self.noise * 8)) for v, _, _, _ in top_k]
            z = sum(exps)
            r = self.rng.random() * z
            acc = 0.0
            for e, (v, rot, col, gb) in zip(exps, top_k):
                acc += e
                if acc >= r:
                    self.last_choice = (rot, col)
                    self.last_score = v
                    self.last_graph_bonus = gb
                    return rot, col

        self.last_choice = (best_rot, best_col)
        self.last_score = best_v
        self.last_graph_bonus = best_g
        return best_rot, best_col

    def _begin_piece(self, world: TetrisWorld) -> None:
        """Snapshot board at the start of working on the current piece."""
        if world.active is None:
            return
        self._pre_board = observe_board(world)
        self._score_at_piece_start = float(world.score)
        self._piece_kind = world.active.kind
        self._piece_cmds = 0
        self._replans_this_piece = 0
        self._stuck_lateral = 0
        # Phase A: structured target once hard+left+right known (not full map only)
        if self.play_ready() or self.map_complete():
            self._target = self.choose_target(world)
        else:
            # Until we can hard-drop intentionally, no structured target
            self._target = None

    def _replan_target(self, world: TetrisWorld) -> None:
        """Pick a new landing pose mid-piece (escape stuck / local minimum)."""
        if world.active is None or not (self.play_ready() or self.map_complete()):
            return
        self._target = self.choose_target(world)
        self._replans_this_piece += 1
        self._stuck_lateral = 0

    def _learn_from_real_drop(
        self,
        world: TetrisWorld,
        *,
        kind: str,
        rot: int,
        col: int,
        pre: dict[str, float],
        score_before: float,
    ) -> None:
        """Update drop model from an actual lock (observed only)."""
        post = observe_board(world)
        lines_cleared = int(world.last_lines_cleared)
        topped = bool(world.game_over)
        score_delta = float(world.score) - score_before
        # Height-aware shaped reward (not raw Tetris score alone)
        reward = board_quality_reward(
            pre,
            post,
            lines_cleared=lines_cleared,
            topped_out=topped,
            score_delta=score_delta,
            rows=float(world.rows),
        )

        self.last_reward = reward
        self.last_features = post
        self.lines_total += lines_cleared
        # Explicit packing outcome for Symbioid meta sensors
        pre_h = float(pre.get("holes", 0.0))
        post_h = float(post.get("holes", 0.0))
        self.last_d_holes = post_h - pre_h
        self.last_post_holes = post_h

        exp = DropExperience(
            kind=kind,
            rot=rot % 4,
            col=col,
            pre=dict(pre),
            post=post,
            reward=reward,
            lines_cleared=lines_cleared,
            topped_out=topped,
        )
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences :]

        key = (kind, rot % 4, int(col))
        self.slot_reward_sum[key] = self.slot_reward_sum.get(key, 0.0) + reward
        self.slot_reward_n[key] = self.slot_reward_n.get(key, 0) + 1
        self.slot_post_max_h_sum[key] = (
            self.slot_post_max_h_sum.get(key, 0.0) + post.get("max_height", 0.0)
        )
        self.slot_post_holes_sum[key] = (
            self.slot_post_holes_sum.get(key, 0.0) + post.get("holes", 0.0)
        )
        hb = self._h_bucket(pre.get("max_height", 0.0), world.rows)
        key_h = (kind, rot % 4, int(col), hb)
        self.slot_h_sum[key_h] = self.slot_h_sum.get(key_h, 0.0) + reward
        self.slot_h_n[key_h] = self.slot_h_n.get(key_h, 0) + 1

        # Fit board-feature weights so score_observed_board ≈ shaped reward
        pred = self.score_observed_board(post, lines_cleared=float(lines_cleared))
        err = reward - pred
        self.w_lines = _clamp(
            self.w_lines + self.lr * err * max(1.0, float(lines_cleared)), 0.0, 8.0
        )
        self.w_holes = _clamp(
            self.w_holes + self.lr * err * (post.get("holes", 0.0) + 0.5) * 0.08,
            -8.0,
            -0.2,
        )
        self.w_bump = _clamp(
            self.w_bump + self.lr * err * (post.get("bumpiness", 0.0) + 0.5) * 0.04,
            -3.0,
            0.0,
        )
        self.w_agg = _clamp(
            self.w_agg + self.lr * err * (post.get("agg_height", 0.0) + 1.0) * 0.01,
            -2.0,
            0.0,
        )
        self.w_max_h = _clamp(
            self.w_max_h + self.lr * err * (post.get("max_height", 0.0) + 1.0) * 0.05,
            -10.0,
            -0.5,
        )
        self.w_well = _clamp(
            self.w_well + self.lr * err * (post.get("well", 0.0) + 0.5) * 0.06,
            -6.0,
            -0.2,
        )
        self.w_max_well = _clamp(
            self.w_max_well + self.lr * err * (post.get("max_well", 0.0) + 0.5) * 0.06,
            -5.0,
            0.0,
        )
        self.w_range = _clamp(
            self.w_range + self.lr * err * (post.get("height_range", 0.0) + 0.5) * 0.04,
            -3.0,
            0.0,
        )

        floor = float(self.place_explore_floor)
        if topped:
            self.top_outs += 1
            self.place_explore = min(0.8, self.place_explore * 1.05 + 0.02)
        else:
            self.place_explore = max(floor, self.place_explore * 0.995)
            self.noise = max(0.08, self.noise * 0.998)

        if lines_cleared > 0:
            self.place_explore = max(floor, self.place_explore * 0.97)

    # ----------------------------------------------------------- control

    def desired_intent(self, world: TetrisWorld) -> str:
        if world.active is None or world.game_over:
            return "explore"

        # Phase A: explore until hard+left+right known (not full map).
        # Partial one-key maps must not spam that key (see play_ready).
        if not self.play_ready() and not self.map_complete():
            return "explore"

        if self._pre_board is None:
            self._begin_piece(world)
        if self._target is None:
            if self.play_ready() or self.map_complete():
                self._replan_target(world)
            if self._target is None:
                return "explore"

        cur = world.active
        # Force hard if piece is stalling forever (gravity-only death spiral)
        if (
            self.byte_for_effect("hard") is not None
            and self._piece_cmds >= int(self.force_hard_after_cmds)
        ):
            self._target = (cur.rotation % 4, cur.col)
            return "hard"

        # Periodic replan to escape bad targets / local minima
        if (
            self._piece_cmds > 0
            and self.replan_every_cmds > 0
            and self._piece_cmds % int(self.replan_every_cmds) == 0
            and self._replans_this_piece < 4
        ):
            self._replan_target(world)

        tgt_rot, tgt_col = self._target
        # Keep target geometrically on-board for this piece shape
        tgt_rot, tgt_col = self._clamp_pose(cur.kind, tgt_rot, tgt_col, world.cols)
        self._target = (tgt_rot, tgt_col)

        # Stuck on a wall / stack: replan once, then hard-drop here
        if self._stuck_lateral >= 2:
            if self._replans_this_piece < 3:
                self._replan_target(world)
                tgt_rot, tgt_col = self._target or (cur.rotation % 4, cur.col)
                tgt_rot, tgt_col = self._clamp_pose(cur.kind, tgt_rot, tgt_col, world.cols)
                self._target = (tgt_rot, tgt_col)
            if self._stuck_lateral >= 2:
                # Still stuck after replan attempt on next cmds — hard now
                if self._replans_this_piece >= 1 and self._stuck_lateral >= 3:
                    self._target = (cur.rotation % 4, cur.col)
                    return "hard"

        if cur.rotation % 4 != tgt_rot:
            if self.byte_for_effect("rotate") is None:
                return "explore"  # hunt rotate while otherwise play-ready
            return "rotate"
        if cur.col < tgt_col:
            return "right"
        if cur.col > tgt_col:
            return "left"
        return "hard"

    def select_byte(self, world: TetrisWorld) -> tuple[int, str]:
        """
        Coach / geometric fallback command.

        When ``network_primary`` and play-ready, explore is minimal — the demo
        path should usually have already supplied ``preferred_intent`` from
        Symbioid. ``desired_intent`` still walks a network-scored ``_target``
        when ``graph_placement_bonus`` is set.
        """
        intent = self.desired_intent(world)
        if intent == "explore" or self.byte_for_effect(intent) is None:
            return self._pick_explore_byte(), "explore"
        # Network-primary: keep residual discovery tiny once we can act
        if self.network_primary and (self.play_ready() or self.map_complete()):
            explore_p = max(0.01, self.explore_rate * 0.02)
        else:
            explore_p = max(0.02, self.explore_rate * (0.06 if self.play_ready() else 0.1))
        if self.rng.random() < explore_p:
            return self._pick_explore_byte(), "explore"
        b = self.byte_for_effect(intent)
        assert b is not None
        return b, intent

    def tick(
        self,
        world: TetrisWorld,
        *,
        run_gravity: bool = True,
        preferred_intent: Optional[str] = None,
        graph_bias: float = 0.92,
    ) -> int:
        """
        Emit one secret byte. Learn byte map + drop effects from observation.
        Never scores candidates with simulate_placement.

        Gravity runs in a **separate** step after the command so left/right
        are not mixed with a free-fall delta (which previously broke mapping).

        ``preferred_intent``: Symbioid / graph recommendation (e.g. left/hard).
        With ``network_primary`` (default), preferred intent is the *primary*
        command source once a live byte is known; coach ``select_byte`` is fallback.
        ``graph_bias`` is the probability of taking the network intent (0..1).
        """
        if world.game_over:
            self.last_effect = "game_over"
            self.last_intent = "idle"
            return self.last_byte

        # New piece → snapshot board / pick target without drop oracle
        if world.active is not None and self._pre_board is None:
            self._begin_piece(world)

        before = WorldSnapshot.take(world)
        bias = max(0.0, min(1.0, float(graph_bias)))
        if self.network_primary and bias < 0.5 and (
            self.play_ready() or self.map_complete()
        ):
            # Network-primary demos should not accidentally run coach-only
            bias = max(bias, 0.90)

        used_network = False
        code: int
        intent: str
        # --- Symbioid-first command selection ---
        if (
            preferred_intent
            and preferred_intent in VALID_ACTIONS
            and self.byte_for_effect(preferred_intent) is not None
            and self.rng.random() < bias
        ):
            # Allow weak network influence even while mapping (bias often small)
            if self.play_ready() or self.map_complete() or self.network_primary:
                b = self.byte_for_effect(preferred_intent)
                if b is not None:
                    code, intent = b, preferred_intent
                    used_network = True
        if not used_network:
            code, intent = self.select_byte(world)
            # Coach-primary soft override (legacy)
            if (
                not self.network_primary
                and preferred_intent
                and preferred_intent in VALID_ACTIONS
                and (self.play_ready() or self.map_complete())
                and self.byte_for_effect(preferred_intent) is not None
                and self.rng.random() < bias
            ):
                b = self.byte_for_effect(preferred_intent)
                if b is not None:
                    code, intent = b, preferred_intent
                    used_network = True
        self.last_byte = code
        self.last_intent = intent
        self.last_network_cmd = used_network
        self._piece_cmds += 1

        prev_placed = world.pieces_placed
        col_before = before.col
        # Command only — no gravity in this call
        world.step_byte(code)

        effect = classify_effect(before, world)
        self.last_effect = effect
        # Credit command-step effects only (gravity is a separate step below)
        self._note_effect(code, effect)

        # Detect endless RIGHT/LEFT into a wall (unreachable target)
        if (
            intent in ("left", "right")
            and world.active is not None
            and before.has_active
            and world.active.kind == before.kind
            and world.active.col == col_before
        ):
            self._stuck_lateral += 1
        elif intent in ("left", "right", "rotate", "hard"):
            self._stuck_lateral = 0

        if self.map_complete() or self.play_ready():
            self.explore_rate = max(0.04, self.explore_rate * 0.999)

        locked_by_cmd = world.pieces_placed > prev_placed or (
            world.game_over and before.has_active
        )
        if locked_by_cmd:
            # True effect for Mind outcomes (prefer hard over explore)
            self.last_lock_effect = (
                effect if effect in VALID_ACTIONS else intent
                if intent in VALID_ACTIONS
                else "noop"
            )
            self._finish_piece_lock(world, before)
            return code

        # Separate gravity tick (does not re-attribute to the command byte)
        if run_gravity and not world.game_over:
            g_before = WorldSnapshot.take(world)
            g_prev = world.pieces_placed
            world.tick_gravity()
            if world.pieces_placed > g_prev or (
                world.game_over and g_before.has_active
            ):
                # Gravity lock — learn drop outcome, do NOT map last byte as hard
                self.last_lock_effect = "soft"
                self._finish_piece_lock(world, g_before)
        return code

    def _finish_piece_lock(self, world: TetrisWorld, before: WorldSnapshot) -> None:
        self.placements += 1
        pre = self._pre_board or observe_board(world)
        score0 = self._score_at_piece_start
        kind = before.kind or self._piece_kind or "?"
        rot = before.rotation
        col = before.col
        # Snapshot lock cells for Mind placement-credit (board already has them)
        self.last_lock_pose = (kind, int(rot) % 4, int(col))
        self.last_lock_cells = []
        if before.has_active and kind and kind != "?":
            try:
                from symbioid.world.tetris import ActivePiece

                ap = ActivePiece(
                    kind=kind,
                    row=int(before.row),
                    col=int(col),
                    rotation=int(rot) % 4,
                )
                self.last_lock_cells = list(ap.cells())
            except Exception:
                self.last_lock_cells = []
        self._learn_from_real_drop(
            world,
            kind=kind,
            rot=rot,
            col=col,
            pre=pre,
            score_before=score0,
        )
        self._target = None
        self._pre_board = None
        self._piece_kind = ""
        self._stuck_lateral = 0
        self._piece_cmds = 0
        self._replans_this_piece = 0

    def act(self, world: TetrisWorld) -> bool:
        """
        Headless: choose target from experience, apply placement once.

        Still no simulate_placement for scoring — only for applying the pose
        (instant lock for tests). Learning uses observed post-lock state.
        """
        if world.game_over or world.active is None:
            return False
        pre = observe_board(world)
        score0 = float(world.score)
        kind = world.active.kind
        rot, col = self.choose_target(world)
        # apply_placement is a test shortcut for physics; scoring was experiential
        ok = world.apply_placement(rot, col)
        used_rot, used_col = rot, col
        if not ok:
            # Pose may be illegal — try a few random poses (agent has no legality oracle)
            for _ in range(12):
                r = self.rng.randint(0, 3)
                c = self.rng.randint(0, world.cols - 1)
                if world.apply_placement(r, c):
                    ok = True
                    used_rot, used_col = r, c
                    break
        if ok:
            self.placements += 1
            self.last_lock_pose = (kind, int(used_rot) % 4, int(used_col))
            try:
                from symbioid.world.tetris import ActivePiece

                # Reconstruct final cells via hard-drop geometry on pre-lock board
                # (board already locked — use offsets from piece_cells at used pose)
                cells = world.landing_cells(used_rot, used_col)
                if not cells:
                    # Fallback: piece at top of column (approximate)
                    off = piece_cells(kind, used_rot)
                    cells = [(r, used_col + c) for r, c in off]
                self.last_lock_cells = list(cells)
            except Exception:
                self.last_lock_cells = []
            self._learn_from_real_drop(
                world,
                kind=kind,
                rot=used_rot,
                col=used_col,
                pre=pre,
                score_before=score0,
            )
            self._target = None
            self._pre_board = None
        return ok

    def record_game_score(self, world: TetrisWorld) -> tuple[int, int]:
        entry = (self.game_number, int(world.score))
        self.highscores.append(entry)
        return entry

    def on_new_game(
        self, world: TetrisWorld, *, record: bool = True
    ) -> Optional[tuple[int, int]]:
        entry = None
        if record:
            entry = self.record_game_score(world)
        self.games += 1
        self.game_number += 1
        self._target = None
        self._pre_board = None
        self._piece_kind = ""
        self._stuck_lateral = 0
        self.last_intent = "noop"
        # Keep byte map + drop memory across games
        world.reset()
        return entry

    def best_score(self) -> int:
        if not self.highscores:
            return 0
        return max(s for _, s in self.highscores)

    def highscore_lines(self, limit: int = 12) -> list[str]:
        """Format highscores for HUD: best score first — ``#ddd ssssss``."""
        # Sort by score desc, then game number desc for stable ties
        rows = sorted(
            self.highscores,
            key=lambda t: (t[1], t[0]),
            reverse=True,
        )[:limit]
        # #ddd = 3-digit game no; ssssss = 6-wide right-aligned score
        return [f"#{n:03d} {score:>6d}" for n, score in rows]

    def drop_model_summary(self) -> str:
        n = len(self.experiences)
        slots = len(self.slot_reward_n)
        return (
            f"drops={n} slots={slots} place_ex={self.place_explore:.2f} "
            f"wH={self.w_max_h:.1f} wHole={self.w_holes:.1f}"
        )

    def summary(self) -> str:
        best = self.best_score()
        n_try = len(self.bytes_tried)
        miss = ",".join(self.missing_effects()) or "none"
        return (
            f"g={self.game_number} place={self.placements} lines={self.lines_total} "
            f"top_out={self.top_outs} best={best} "
            f"bytes_tried={n_try}/256 scans={self._scan_passes} miss=[{miss}] "
            f"map=[{self.map_progress()}] "
            f"{self.drop_model_summary()} "
            f"byte=0x{self.last_byte:02X} eff={self.last_effect} "
            f"R={self.last_reward:.1f} intent={self.last_intent}"
        )
