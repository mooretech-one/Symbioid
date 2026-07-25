"""Minimal Tetris physics for Symbioid closed-loop demos (no rendering)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Secret byte → action cipher
# Controllers only emit 0..255. Most bytes are dead (noop); a few map to moves.
# The learner must discover the mapping by observing the world — it is never
# given the table (and sensor_world does not expose decoded actions).
# ---------------------------------------------------------------------------

VALID_ACTIONS = ("left", "right", "rotate", "hard")


@dataclass
class ActionCipher:
    """
    Opaque mapping: byte → action name.

    Only ``live`` entries produce real game commands; every other byte is noop.
    """

    mapping: dict[int, str] = field(default_factory=dict)

    @classmethod
    def random(
        cls,
        rng: Optional[random.Random] = None,
        actions: tuple[str, ...] = VALID_ACTIONS,
    ) -> "ActionCipher":
        r = rng or random.Random()
        codes = r.sample(range(256), len(actions))
        return cls({c: a for c, a in zip(codes, actions)})

    @classmethod
    def fixed(cls, mapping: dict[int, str]) -> "ActionCipher":
        return cls({int(k) & 0xFF: v for k, v in mapping.items()})

    def decode(self, code: int) -> str:
        return self.mapping.get(int(code) & 0xFF, "noop")

    def live_bytes(self) -> list[int]:
        return sorted(self.mapping.keys())

    def __len__(self) -> int:
        return len(self.mapping)


# ---------------------------------------------------------------------------
# Pieces: each rotation is a list of (row, col) offsets in a 4×4 box.
# Colors are RGB triples for the pygame demo (ignored by headless tests).
# ---------------------------------------------------------------------------

PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")

# Rotation states 0..3 for each piece (row increases downward).
_PIECE_CELLS: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "I": (
        ((1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 1), (1, 1), (2, 1), (3, 1)),
    ),
    "O": (
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
    ),
    "T": (
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ),
    "S": (
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ),
    "Z": (
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 2), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    "J": (
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ),
    "L": (
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ),
}

PIECE_COLORS: dict[str, tuple[int, int, int]] = {
    "I": (80, 220, 220),
    "O": (240, 220, 80),
    "T": (180, 100, 220),
    "S": (80, 200, 120),
    "Z": (220, 80, 100),
    "J": (80, 120, 220),
    "L": (240, 160, 60),
}

# Line-clear score by number of lines (classic-ish).
_LINE_SCORES = (0, 100, 300, 500, 800)


def piece_cells(kind: str, rotation: int) -> tuple[tuple[int, int], ...]:
    return _PIECE_CELLS[kind][rotation % 4]


@dataclass
class ActivePiece:
    kind: str
    row: int = 0
    col: int = 3
    rotation: int = 0

    def cells(self) -> list[tuple[int, int]]:
        return [(self.row + dr, self.col + dc) for dr, dc in piece_cells(self.kind, self.rotation)]


@dataclass
class TetrisWorld:
    """
    Standard-ish 10×20 Tetris with 7-bag randomizer.

    Control is either:
      - low-level: step_action("left"|"right"|"rotate"|"soft"|"hard"|"noop")
      - high-level: apply_placement(rotation, col) then hard-drop semantics

    Coordinates: row 0 is the top (spawn), row grows downward.
    Board cells store piece kind char or "" for empty.
    """

    cols: int = 10
    rows: int = 20
    gravity_interval: int = 12  # ticks between automatic soft drops
    # Secret control map — agent sees only bytes, not this table.
    cipher: Optional[ActionCipher] = None
    # state
    board: list[list[str]] = field(default_factory=list)
    active: Optional[ActivePiece] = None
    next_kind: str = "T"
    bag: list[str] = field(default_factory=list)
    score: int = 0
    lines: int = 0
    pieces_placed: int = 0
    ticks: int = 0
    gravity_counter: int = 0
    game_over: bool = False
    last_event: str = ""
    last_lines_cleared: int = 0
    last_byte: int = 0
    last_byte_action: str = "noop"  # ground truth for tests only; not in sensors
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.cipher is None:
            self.cipher = ActionCipher.random(self.rng)
        if not self.board:
            self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        self.board = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        self.bag.clear()
        self.score = 0
        self.lines = 0
        self.pieces_placed = 0
        self.ticks = 0
        self.gravity_counter = 0
        self.game_over = False
        self.last_event = "reset"
        self.last_lines_cleared = 0
        self.next_kind = self._draw_kind()
        self.active = None
        self._spawn()

    def _refill_bag(self) -> None:
        bag = list(PIECE_NAMES)
        self.rng.shuffle(bag)
        self.bag.extend(bag)

    def _draw_kind(self) -> str:
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def _spawn(self) -> bool:
        """Spawn next piece. Returns False on top-out."""
        kind = self.next_kind
        self.next_kind = self._draw_kind()
        # Center the 4-wide piece box on the board (cols//2 - 2) so left/right
        # travel distances are balanced (old col=3 biased RIGHT on 10-wide).
        spawn_col = max(0, self.cols // 2 - 2)
        piece = ActivePiece(kind=kind, row=0, col=spawn_col, rotation=0)
        if not self._fits(piece):
            self.active = piece
            self.game_over = True
            self.last_event = "top_out"
            return False
        self.active = piece
        self.last_event = "spawn"
        self.gravity_counter = 0
        return True

    # -------------------------------------------------------------- collisions

    def _fits(self, piece: ActivePiece) -> bool:
        for r, c in piece.cells():
            if c < 0 or c >= self.cols or r >= self.rows:
                return False
            if r < 0:
                continue
            if self.board[r][c]:
                return False
        return True

    def _occupied(self, r: int, c: int) -> bool:
        if c < 0 or c >= self.cols or r >= self.rows:
            return True
        if r < 0:
            return False
        return bool(self.board[r][c])

    # ---------------------------------------------------------------- actions

    def try_move(self, d_row: int, d_col: int) -> bool:
        if self.game_over or self.active is None:
            return False
        trial = ActivePiece(
            kind=self.active.kind,
            row=self.active.row + d_row,
            col=self.active.col + d_col,
            rotation=self.active.rotation,
        )
        if self._fits(trial):
            self.active = trial
            return True
        return False

    def try_rotate(self, direction: int = 1) -> bool:
        """Rotate CW (direction=+1) or CCW (-1). Simple kicks: 0, ±1, +1 col."""
        if self.game_over or self.active is None:
            return False
        new_rot = (self.active.rotation + direction) % 4
        for kick in (0, -1, 1, -2, 2):
            trial = ActivePiece(
                kind=self.active.kind,
                row=self.active.row,
                col=self.active.col + kick,
                rotation=new_rot,
            )
            if self._fits(trial):
                self.active = trial
                return True
        return False

    def soft_drop(self) -> bool:
        """Move one row down; lock if blocked. Returns True if moved."""
        if self.game_over or self.active is None:
            return False
        if self.try_move(1, 0):
            self.score += 1
            self.last_event = "soft_drop"
            return True
        self._lock()
        return False

    def hard_drop(self) -> int:
        """Drop to bottom and lock. Returns rows fallen."""
        if self.game_over or self.active is None:
            return 0
        fallen = 0
        while self.try_move(1, 0):
            fallen += 1
        self.score += 2 * fallen
        self._lock()
        return fallen

    def step_byte(self, code: int) -> str:
        """
        Only control path for the agent: emit a raw byte 0..255.

        Most bytes do nothing. A few (secret) map to left/right/rotate/hard.
        Returns the decoded action for tests/logging — **do not** feed this
        back into the learner as privileged knowledge; use world state deltas.
        """
        self.last_byte = int(code) & 0xFF
        assert self.cipher is not None
        action = self.cipher.decode(self.last_byte)
        self.last_byte_action = action
        self.step_action(action)
        return action

    def step_action(self, action: str = "noop", *, apply_gravity: bool = False) -> None:
        """
        One control command (internal / privileged).

        Prefer ``step_byte`` for agent control. Gravity is **not** mixed into
        the same step by default — that would scramble effect observation
        (left+gravity in one tick looks like an unknown diagonal).

        action: noop | left | right | rotate | rotate_ccw | soft | hard
        """
        if self.game_over:
            self.last_event = "game_over"
            return

        self.ticks += 1
        act = (action or "noop").lower()
        if act == "left":
            if self.try_move(0, -1):
                self.last_event = "left"
            else:
                self.last_event = "blocked"
        elif act == "right":
            if self.try_move(0, 1):
                self.last_event = "right"
            else:
                self.last_event = "blocked"
        elif act in ("rotate", "rotate_cw", "cw"):
            if self.try_rotate(1):
                self.last_event = "rotate"
            else:
                self.last_event = "blocked"
        elif act in ("rotate_ccw", "ccw"):
            if self.try_rotate(-1):
                self.last_event = "rotate_ccw"
            else:
                self.last_event = "blocked"
        elif act in ("soft", "soft_drop", "down"):
            self.soft_drop()
            return
        elif act in ("hard", "hard_drop", "drop"):
            self.hard_drop()
            return
        else:
            self.last_event = "noop"

        if apply_gravity:
            self.tick_gravity()

    def tick_gravity(self) -> bool:
        """
        Apply one gravity step if the interval elapsed.

        Returns True if gravity moved or locked the piece.
        Separated from commands so byte→effect learning stays clean.
        """
        if self.game_over or self.active is None:
            return False
        self.gravity_counter += 1
        if self.gravity_counter < self.gravity_interval:
            return False
        self.gravity_counter = 0
        if self.try_move(1, 0):
            self.last_event = "gravity"
            return True
        self._lock()
        return True

    def apply_placement(self, rotation: int, col: int) -> bool:
        """
        High-level action: set piece rotation/column, hard-drop, lock.
        Returns False if placement illegal or game already over.
        """
        if self.game_over or self.active is None:
            return False
        trial = ActivePiece(
            kind=self.active.kind,
            row=0,
            col=col,
            rotation=rotation % 4,
        )
        # Drop from top until resting on something (or invalid).
        if not self._fits(trial):
            # try a few row starts if spawn box is blocked mid-board rare
            return False
        while True:
            nxt = ActivePiece(
                kind=trial.kind,
                row=trial.row + 1,
                col=trial.col,
                rotation=trial.rotation,
            )
            if self._fits(nxt):
                trial = nxt
            else:
                break
        self.active = trial
        self._lock()
        return True

    # ------------------------------------------------------------- lock/clear

    def _lock(self) -> None:
        if self.active is None:
            return
        for r, c in self.active.cells():
            if r < 0:
                self.game_over = True
                self.last_event = "top_out"
                self.active = None
                return
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.board[r][c] = self.active.kind
        self.pieces_placed += 1
        cleared = self._clear_lines()
        # Safety: never leave a full row on the board after a lock
        extra = self._clear_lines()
        while extra:
            cleared += extra
            extra = self._clear_lines()
        self.last_lines_cleared = cleared
        if cleared:
            self.lines += cleared
            # Cap score table index (classic max 4-line)
            self.score += _LINE_SCORES[min(cleared, 4)]
            self.last_event = "line_clear"
        else:
            self.last_event = "lock"
        self.active = None
        if not self.game_over:
            self._spawn()

    def _row_is_full(self, row: list[str]) -> bool:
        """True only for a complete filled row of width ``cols``."""
        if len(row) != self.cols:
            return False
        return all(bool(cell) for cell in row)

    def _clear_lines(self) -> int:
        """
        Remove every completely filled row and pad empty rows at the top.

        Counts cleared rows by scanning (not ``rows - len(kept)``), so a
        drifted board length cannot invent phantom clears or skip padding
        when the difference is negative (negative ints are truthy in Python).
        """
        self.board, cleared = _clear_rows(self.board, self.cols, self.rows)
        return cleared

    # --------------------------------------------------------------- features

    def column_heights(self) -> list[int]:
        """Height of each column (0 = empty, rows = full)."""
        heights = []
        for c in range(self.cols):
            h = 0
            for r in range(self.rows):
                if self.board[r][c]:
                    h = self.rows - r
                    break
            heights.append(h)
        return heights

    def hole_count(self) -> int:
        holes = 0
        for c in range(self.cols):
            seen = False
            for r in range(self.rows):
                if self.board[r][c]:
                    seen = True
                elif seen:
                    holes += 1
        return holes

    def bumpiness(self) -> int:
        h = self.column_heights()
        return sum(abs(h[i] - h[i + 1]) for i in range(len(h) - 1))

    def aggregate_height(self) -> int:
        return sum(self.column_heights())

    def max_height(self) -> int:
        hs = self.column_heights()
        return max(hs) if hs else 0

    def board_with_active(self) -> list[list[str]]:
        """Copy of board with active piece painted in."""
        out = [row[:] for row in self.board]
        if self.active is not None:
            for r, c in self.active.cells():
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    out[r][c] = self.active.kind
        return out

    def _column_locked(self, c: int) -> list[bool]:
        """Per-row locked (settled) occupancy for column ``c``."""
        return [bool(self.board[r][c]) for r in range(self.rows)]

    def _column_active_rows(self, c: int) -> set[int]:
        """Rows of the active piece in column ``c``."""
        rows: set[int] = set()
        if self.active is None:
            return rows
        for r, cc in self.active.cells():
            if cc == c and 0 <= r < self.rows:
                rows.add(r)
        return rows

    def cell_field_state(self, *, with_active: bool = True) -> list[list[float]]:
        """
        Full board map for sensors: each cell is one of

          1.0 — block (locked, or active piece if with_active)
          0.5 — hole  (empty with a **locked** fill above — not under the falling piece)
          0.0 — open  (empty, no locked fill above)

        Holes use locked board only so a falling piece does not invent
        column-long hole storms under itself.
        Row 0 is the top of the playfield.
        """
        rows, cols = self.rows, self.cols
        out: list[list[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for c in range(cols):
            locked = self._column_locked(c)
            active_rows = self._column_active_rows(c) if with_active else set()
            seen_locked = False
            for r in range(rows):
                if locked[r] or r in active_rows:
                    out[r][c] = 1.0
                    if locked[r]:
                        seen_locked = True
                elif seen_locked:
                    out[r][c] = 0.5
                else:
                    out[r][c] = 0.0
        return out

    def sky_row(self, *, with_active: bool = True) -> int:
        """
        First row from the top that is not pure open sky (locked or active).

        Rows ``0 .. sky_row-1`` are completely empty and can be skipped for
        cell sampling. Returns ``rows`` when the whole board is empty.
        """
        active_rows_by_c: list[set[int]] = [set() for _ in range(self.cols)]
        if with_active and self.active is not None:
            for r, c in self.active.cells():
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    active_rows_by_c[c].add(r)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] or r in active_rows_by_c[c]:
                    return r
        return self.rows

    def solid_floor_start_row(self) -> int:
        """
        First row of a contiguous full-width **locked** solid slab from the bottom.

        Rows ``solid_floor_start_row .. rows-1`` are entirely locked (no empties).
        Those cells cannot change until a line clear. Returns ``rows`` when the
        bottom row is not fully solid (common mid-game).

        Note: fully filled rows normally clear immediately on lock; this still
        covers the brief post-lock window and any deferred-clear setups.
        """
        r = self.rows - 1
        while r >= 0 and all(bool(self.board[r][c]) for c in range(self.cols)):
            r -= 1
        return r + 1

    def active_cells_set(self) -> set[tuple[int, int]]:
        """Board cells occupied by the active piece (in-bounds only)."""
        out: set[tuple[int, int]] = set()
        if self.active is None:
            return out
        for r, c in self.active.cells():
            if 0 <= r < self.rows and 0 <= c < self.cols:
                out.add((r, c))
        return out

    def cell_sample_roi(
        self, *, with_active: bool = True
    ) -> tuple[int, int]:
        """
        Inclusive-exclusive row band ``[r_lo, r_hi)`` for cell sampling.

        - ``r_lo`` = :meth:`sky_row` (skip empty top)
        - ``r_hi`` = :meth:`solid_floor_start_row` (skip solid full base)

        Empty board → ``(rows, rows)`` (empty band). Active piece always expands
        sky via ``with_active``.
        """
        r_lo = self.sky_row(with_active=with_active)
        r_hi = self.solid_floor_start_row()
        if r_lo > r_hi:
            r_lo = r_hi
        return r_lo, r_hi

    def cell_reading(self, r: int, c: int, *, with_active: bool = True) -> float:
        """Single-cell reading (block=1.0, hole=0.5, open=0.0)."""
        if r < 0 or c < 0 or r >= self.rows or c >= self.cols:
            return 0.0
        locked = self._column_locked(c)
        if locked[r]:
            return 1.0
        if with_active and r in self._column_active_rows(c):
            return 1.0
        if any(locked[rr] for rr in range(r)):
            return 0.5
        return 0.0

    def ghost_row(self) -> Optional[int]:
        """Row of active piece if hard-dropped (top-left box row)."""
        if self.active is None:
            return None
        trial = ActivePiece(
            kind=self.active.kind,
            row=self.active.row,
            col=self.active.col,
            rotation=self.active.rotation,
        )
        while True:
            nxt = ActivePiece(
                kind=trial.kind,
                row=trial.row + 1,
                col=trial.col,
                rotation=trial.rotation,
            )
            if self._fits(nxt):
                trial = nxt
            else:
                return trial.row

    def legal_placements(self) -> list[tuple[int, int]]:
        """
        Enumerate (rotation, col) pairs that can be hard-dropped from the top
        without intersecting filled cells. Used by the placement learner.
        """
        if self.active is None or self.game_over:
            return []
        kind = self.active.kind
        found: list[tuple[int, int]] = []
        rots = 1 if kind == "O" else 4
        for rot in range(rots):
            # Column range: keep all cells on board horizontally.
            min_dc = min(dc for _, dc in piece_cells(kind, rot))
            max_dc = max(dc for _, dc in piece_cells(kind, rot))
            for col in range(-min_dc, self.cols - max_dc):
                trial = ActivePiece(kind=kind, row=0, col=col, rotation=rot)
                if not self._fits(trial):
                    continue
                # Simulate drop
                while True:
                    nxt = ActivePiece(
                        kind=kind, row=trial.row + 1, col=col, rotation=rot
                    )
                    if self._fits(nxt):
                        trial = nxt
                    else:
                        break
                # Must land with all cells on board (not above)
                cells = trial.cells()
                if any(r < 0 for r, _ in cells):
                    continue
                found.append((rot, col))
        return found

    def landing_cells(self, rotation: int, col: int) -> list[tuple[int, int]]:
        """
        Cells filled if current piece hard-drops at (rotation, col).
        Empty list if illegal. Does not mutate self.
        """
        if self.active is None or self.game_over:
            return []
        kind = self.active.kind
        trial = ActivePiece(kind=kind, row=0, col=col, rotation=rotation % 4)
        if not self._fits(trial):
            return []
        while True:
            nxt = ActivePiece(
                kind=kind, row=trial.row + 1, col=col, rotation=trial.rotation
            )
            if self._fits(nxt):
                trial = nxt
            else:
                break
        cells = trial.cells()
        if any(r < 0 for r, _ in cells):
            return []
        return cells

    def simulate_placement(self, rotation: int, col: int) -> Optional[dict[str, float]]:
        """
        Soft-clone board, apply placement, return board features after lock
        (before next spawn matters). Does not mutate self.
        """
        if self.active is None or self.game_over:
            return None
        # clone board
        board = [row[:] for row in self.board]
        kind = self.active.kind
        trial = ActivePiece(kind=kind, row=0, col=col, rotation=rotation % 4)
        if not self._cell_fits(board, trial):
            return None
        while True:
            nxt = ActivePiece(kind=kind, row=trial.row + 1, col=col, rotation=trial.rotation)
            if self._cell_fits(board, nxt):
                trial = nxt
            else:
                break
        cells = trial.cells()
        if any(r < 0 for r, _ in cells):
            return None
        for r, c in cells:
            board[r][c] = kind
        board, cleared = _clear_rows(board, self.cols, self.rows)
        return _board_features(board, self.cols, self.rows, lines_cleared=cleared)

    @staticmethod
    def _cell_fits(board: list[list[str]], piece: ActivePiece) -> bool:
        rows, cols = len(board), len(board[0])
        for r, c in piece.cells():
            if c < 0 or c >= cols or r >= rows:
                return False
            if r < 0:
                continue
            if board[r][c]:
                return False
        return True

    def sensor_world(self) -> dict[str, float]:
        """Normalized features for Sensor.transfer / Interface world map."""
        heights = self.column_heights()
        max_h = float(self.rows) or 1.0
        agg = float(self.aggregate_height())
        mx = float(self.max_height())
        out: dict[str, float] = {
            "score": float(self.score),
            "lines": float(self.lines),
            "pieces": float(self.pieces_placed),
            "holes": float(self.hole_count()),
            "bumpiness": float(self.bumpiness()),
            "agg_height": agg,
            "max_height": mx,
            "game_over": 1.0 if self.game_over else 0.0,
            "last_lines": float(self.last_lines_cleared),
            # Raw control byte only — never the decoded action name.
            "last_byte": float(self.last_byte),
            "last_byte_n": float(self.last_byte) / 255.0,
        }
        for i, h in enumerate(heights):
            out[f"h{i}"] = h / max_h
        out["max_height_n"] = mx / max_h
        out["agg_height_n"] = agg / (max_h * float(self.cols))
        out["holes_n"] = self.hole_count() / 40.0
        out["bump_n"] = self.bumpiness() / 40.0
        if heights:
            out["height_range_n"] = (max(heights) - min(heights)) / max_h
        else:
            out["height_range_n"] = 0.0
        if self.active is not None:
            out["piece_x"] = self.active.col / max(1, self.cols - 1)
            out["piece_y"] = self.active.row / max(1, self.rows - 1)
            out["piece_rot"] = self.active.rotation / 3.0
            out["piece_id"] = float(PIECE_NAMES.index(self.active.kind)) / 6.0
        else:
            out["piece_x"] = 0.0
            out["piece_y"] = 0.0
            out["piece_rot"] = 0.0
            out["piece_id"] = 0.0
        out["next_id"] = float(PIECE_NAMES.index(self.next_kind)) / 6.0
        return out


def _clear_rows(
    board: list[list[str]], cols: int, rows: int
) -> tuple[list[list[str]], int]:
    """Pure line-clear helper used by simulate + world (keeps height == rows)."""
    kept: list[list[str]] = []
    cleared = 0
    for row in board:
        if len(row) < cols:
            row = list(row) + [""] * (cols - len(row))
        elif len(row) > cols:
            row = list(row[:cols])
        else:
            row = list(row)
        if len(row) == cols and all(bool(cell) for cell in row):
            cleared += 1
        else:
            kept.append(row)
    if cleared:
        pad = [["" for _ in range(cols)] for _ in range(cleared)]
        board = pad + kept
    else:
        board = kept
    if len(board) < rows:
        board = [["" for _ in range(cols)] for _ in range(rows - len(board))] + board
    elif len(board) > rows:
        board = board[-rows:]
    return board, cleared


def _board_features(
    board: list[list[str]],
    cols: int,
    rows: int,
    *,
    lines_cleared: int = 0,
) -> dict[str, Any]:
    heights: list[int] = []
    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[r][c]:
                h = rows - r
                break
        heights.append(h)
    holes = 0
    for c in range(cols):
        seen = False
        for r in range(rows):
            if board[r][c]:
                seen = True
            elif seen:
                holes += 1
    bump = sum(abs(heights[i] - heights[i + 1]) for i in range(cols - 1))
    max_h = float(max(heights) if heights else 0)
    min_h = float(min(heights) if heights else 0)
    well = 0.0
    for i, h in enumerate(heights):
        left = heights[i - 1] if i > 0 else h
        right = heights[i + 1] if i + 1 < len(heights) else h
        well += max(0.0, float(min(left, right) - h))
    filled = sum(1 for row in board for cell in row if cell)
    return {
        "agg_height": float(sum(heights)),
        "holes": float(holes),
        "bumpiness": float(bump),
        "max_height": max_h,
        "min_height": min_h,
        "height_range": max_h - min_h,
        "well": well,
        "filled": float(filled),
        "fill_n": filled / float(max(1, rows * cols)),
        "max_height_n": max_h / float(max(1, rows)),
        "agg_height_n": float(sum(heights)) / float(max(1, rows * cols)),
        "lines_cleared": float(lines_cleared),
        "heights": heights,
    }
