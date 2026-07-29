"""Spectral substrate — fixed-size bank for optional FFT mix/store.

Phase 1: SpectralBank pack / FFT / iFFT / unpack / bind.
Phase 2: ``apply_mix_filter`` + residual unpack (FNet-shaped global mix).
Phase 3+: HolonomicStore (not yet).

No claim of biological equivalence — numerical utilities only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


class _HostLike(Protocol):
    """Minimal host surface for pack/unpack (Symbioid or test double)."""

    thoughts: dict[str, Any]


def ceil_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n & (n - 1) == 0:
        return n
    return 1 << n.bit_length()


@dataclass
class SpectralBank:
    """
    Fixed channel bank: ordered slots ↔ Thought ids + real time signal + rFFT cache.

    ``size`` should be a power of two for clean FFT lengths (auto-rounded up if
    ``pad_pow2=True``, the default).
    """

    size: int = 64
    pad_pow2: bool = True
    time_signal: np.ndarray = field(init=False, repr=False)
    spectrum: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    slot_to_thought: list[Optional[str]] = field(init=False, repr=False)
    thought_to_slot: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        n = int(self.size)
        if n < 2:
            raise ValueError("spectral bank size must be >= 2")
        if self.pad_pow2:
            n = ceil_pow2(n)
        self.size = n
        self.time_signal = np.zeros(self.size, dtype=np.float32)
        self.slot_to_thought = [None] * self.size
        self.spectrum = None

    # ------------------------------------------------------------------ bind
    def bind(self, thought_id: str, prefer_slot: Optional[int] = None) -> int:
        """
        Map ``thought_id`` to a stable slot index.

        If already bound, return existing slot (prefer_slot ignored).
        If prefer_slot is free (or already this id), use it; else first free;
        else overwrite lowest-index slot (stable, documented).
        """
        tid = str(thought_id)
        if tid in self.thought_to_slot:
            return int(self.thought_to_slot[tid])

        slot: Optional[int] = None
        if prefer_slot is not None:
            ps = int(prefer_slot)
            if 0 <= ps < self.size:
                occ = self.slot_to_thought[ps]
                if occ is None or occ == tid:
                    slot = ps

        if slot is None:
            for i, occ in enumerate(self.slot_to_thought):
                if occ is None:
                    slot = i
                    break

        if slot is None:
            # Full: reclaim slot 0 (deterministic); drop previous occupant
            slot = 0
            old = self.slot_to_thought[0]
            if old is not None and old in self.thought_to_slot:
                del self.thought_to_slot[old]

        # If prefer_slot reclaims a different occupant
        prev = self.slot_to_thought[slot]
        if prev is not None and prev != tid and prev in self.thought_to_slot:
            del self.thought_to_slot[prev]

        self.slot_to_thought[slot] = tid
        self.thought_to_slot[tid] = slot
        return int(slot)

    def unbind(self, thought_id: str) -> bool:
        tid = str(thought_id)
        slot = self.thought_to_slot.pop(tid, None)
        if slot is None:
            return False
        if self.slot_to_thought[slot] == tid:
            self.slot_to_thought[slot] = None
            self.time_signal[slot] = 0.0
        return True

    def clear_bindings(self) -> None:
        self.slot_to_thought = [None] * self.size
        self.thought_to_slot.clear()
        self.time_signal[:] = 0.0
        self.spectrum = None

    # ------------------------------------------------------------------ pack / unpack
    def pack_from_activations(self, host: _HostLike) -> None:
        """Fill ``time_signal`` from bound Thoughts' ``activation`` (unbound → 0)."""
        thoughts = getattr(host, "thoughts", {}) or {}
        for i, tid in enumerate(self.slot_to_thought):
            if tid is None:
                self.time_signal[i] = 0.0
                continue
            t = thoughts.get(tid)
            if t is None:
                self.time_signal[i] = 0.0
            else:
                self.time_signal[i] = float(getattr(t, "activation", 0.0) or 0.0)
        self.spectrum = None  # invalidate cache

    def unpack_to_activations(
        self,
        host: _HostLike,
        gain: float = 1.0,
        *,
        mode: str = "set",
    ) -> int:
        """
        Write bank values back onto bound Thoughts.

        mode:
          - ``set`` — activation = gain * time_signal[slot] (pack/unpack identity at gain=1)
          - ``add`` — activation += gain * time_signal[slot] (Phase 2 residual mix)
        Returns number of Thoughts updated.
        """
        if mode not in ("set", "add"):
            raise ValueError("mode must be 'set' or 'add'")
        g = float(gain)
        thoughts = getattr(host, "thoughts", {}) or {}
        n = 0
        for i, tid in enumerate(self.slot_to_thought):
            if tid is None:
                continue
            t = thoughts.get(tid)
            if t is None or not getattr(t, "dynamics_enabled", True):
                continue
            val = g * float(self.time_signal[i])
            if mode == "set":
                # Prefer receive/clamp path if available
                amax = float(getattr(t, "activation_max", 3.0))
                t.activation = max(0.0, min(amax, val))
            else:
                if hasattr(t, "receive"):
                    t.receive(val)
                else:
                    amax = float(getattr(t, "activation_max", 3.0))
                    t.activation = max(0.0, min(amax, float(t.activation) + val))
            n += 1
        return n

    # ------------------------------------------------------------------ FFT
    def fft(self) -> np.ndarray:
        """rFFT of ``time_signal`` → cache ``spectrum`` (complex64)."""
        self.spectrum = np.fft.rfft(self.time_signal).astype(np.complex64)
        return self.spectrum

    def ifft(self) -> np.ndarray:
        """iFFT of ``spectrum`` → real ``time_signal`` (float32)."""
        if self.spectrum is None:
            self.fft()
        assert self.spectrum is not None
        recon = np.fft.irfft(self.spectrum, n=self.size)
        self.time_signal = np.asarray(recon, dtype=np.float32)
        return self.time_signal

    def apply_mix_filter(
        self,
        *,
        soft_threshold: float = 0.05,
        lowpass: float = 0.75,
        bin_gains: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        rFFT → soft-threshold weak bins → optional low-pass → optional bin gains → iFFT.

        Pure rFFT/iFFT is identity; the filters create global (non-local) time
        coupling so residual unpack can move energy across unbound graph edges.
        """
        S = self.fft().copy()
        mag = np.abs(S)
        peak = float(mag.max()) if mag.size else 0.0
        top_bin = int(np.argmax(mag)) if mag.size else 0
        thr = float(soft_threshold)
        if peak > 0.0 and thr > 0.0:
            keep = mag >= thr * peak
            S = np.where(keep, S, 0.0)
        lp = float(lowpass)
        if 0.0 < lp < 1.0 and S.size > 1:
            cutoff = max(1, int(len(S) * lp))
            if cutoff < len(S):
                S[cutoff:] = 0.0
        if bin_gains is not None:
            g = np.asarray(bin_gains, dtype=np.float32).reshape(-1)
            if g.size == S.size:
                S = S * g.astype(np.complex64)
            elif g.size > 0:
                n = min(g.size, S.size)
                S = S.copy()
                S[:n] *= g[:n].astype(np.complex64)
        self.spectrum = S.astype(np.complex64)
        self.ifft()
        e = self.energy_time()
        return {
            "mix_energy": e,
            "top_bin": top_bin,
            "peak_mag": peak,
            "n_bins": int(S.size),
        }

    def sync_phases_to_thoughts(self, host: _HostLike) -> int:
        """Set Thought.spectral_phase = 2π · slot / size for each bound id."""
        import math

        thoughts = getattr(host, "thoughts", {}) or {}
        n = 0
        two_pi = 2.0 * math.pi
        for slot, tid in enumerate(self.slot_to_thought):
            if tid is None:
                continue
            t = thoughts.get(tid)
            if t is None:
                continue
            t.spectral_phase = two_pi * float(slot) / float(max(1, self.size))
            n += 1
        return n

    def energy_time(self) -> float:
        """Sum of squares of time_signal."""
        return float(np.dot(self.time_signal, self.time_signal))

    def energy_freq(self) -> float:
        """
        Parseval-consistent energy for numpy default rFFT/irfft pair.

        For real signal x of length N: (1/N) * ( |X[0]|^2 + |X[N/2]|^2 (if even)
        + 2 * sum |X[k]|^2 for k=1..N/2-1 ) ≈ sum |x|^2
        """
        if self.spectrum is None:
            self.fft()
        assert self.spectrum is not None
        n = self.size
        s = self.spectrum
        e = float(np.abs(s[0]) ** 2)
        if n % 2 == 0 and len(s) > 1:
            e += float(np.abs(s[-1]) ** 2)
            mid = s[1:-1]
        else:
            mid = s[1:]
        if mid.size:
            e += 2.0 * float(np.sum(np.abs(mid) ** 2))
        return e / float(n)

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "n_bound": len(self.thought_to_slot),
            "energy_time": self.energy_time(),
        }


def abs_phase_diff(a: float, b: float) -> float:
    """Smallest absolute difference of two phases in radians, in [0, π]."""
    import math

    d = abs(float(a) - float(b)) % (2.0 * math.pi)
    if d > math.pi:
        d = 2.0 * math.pi - d
    return float(d)


# ---------------------------------------------------------------------------
# Holonomic / holographic interference store (Phase 3)
# ---------------------------------------------------------------------------


def key_to_vector(key: str, n: int) -> np.ndarray:
    """
    Deterministic bipolar embedding of a content key into length-n real vector.

    Hash-expanded so capacity is fixed (holonomic); not learned.
    """
    import hashlib

    n = int(n)
    if n < 2:
        raise ValueError("n must be >= 2")
    out = np.empty(n, dtype=np.float32)
    seed = hashlib.sha256(str(key).encode("utf-8")).digest()
    for i in range(n):
        block = hashlib.sha256(seed + int(i).to_bytes(4, "little")).digest()
        out[i] = 1.0 if (block[0] & 1) else -1.0
    # zero-mean-ish unit energy
    out -= float(out.mean())
    norm = float(np.linalg.norm(out))
    if norm > 1e-12:
        out /= norm
    return out


@dataclass
class HolonomicStore:
    """
    Fixed-size complex spectral interference memory.

    Writes accumulate ``strength * rFFT(key_vector)`` into one buffer.
    Reads correlate a probe via ``ifft(conj(rFFT(probe)) * buffer)``.
    Capacity does not grow with number of writes (holonomic property).
    """

    capacity: int = 64  # real-signal length (power of 2 recommended)
    pad_pow2: bool = True
    write_gain: float = 1.0
    read_gain: float = 1.0
    decay: float = 0.002  # multiplicative leak per decay_step / write
    buffer: np.ndarray = field(init=False, repr=False)
    n_writes: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        n = int(self.capacity)
        if n < 2:
            raise ValueError("holonomic capacity must be >= 2")
        if self.pad_pow2:
            n = ceil_pow2(n)
        self.capacity = n
        # rFFT length = n//2+1 complex bins
        self.buffer = np.zeros(n // 2 + 1, dtype=np.complex64)

    def decay_step(self, amount: Optional[float] = None) -> None:
        d = float(self.decay if amount is None else amount)
        if d <= 0.0:
            return
        self.buffer *= np.float32(max(0.0, 1.0 - d))

    def write(self, key_vector: np.ndarray, strength: float = 1.0) -> None:
        """buffer += strength * write_gain * rFFT(normalized key_vector)."""
        x = np.asarray(key_vector, dtype=np.float32).reshape(-1)
        if x.size != self.capacity:
            # pad or truncate
            v = np.zeros(self.capacity, dtype=np.float32)
            m = min(self.capacity, x.size)
            v[:m] = x[:m]
            x = v
        nrm = float(np.linalg.norm(x))
        if nrm > 1e-12:
            x = x / nrm
        self.decay_step()
        s = float(strength) * float(self.write_gain)
        self.buffer += (s * np.fft.rfft(x)).astype(np.complex64)
        self.n_writes += 1

    def write_key(self, key: str, strength: float = 1.0) -> None:
        self.write(key_to_vector(key, self.capacity), strength=strength)

    def correlation(self, probe: np.ndarray) -> np.ndarray:
        """Real circular correlation of probe against store (length = capacity)."""
        x = np.asarray(probe, dtype=np.float32).reshape(-1)
        if x.size != self.capacity:
            v = np.zeros(self.capacity, dtype=np.float32)
            m = min(self.capacity, x.size)
            v[:m] = x[:m]
            x = v
        nrm = float(np.linalg.norm(x))
        if nrm > 1e-12:
            x = x / nrm
        Xp = np.fft.rfft(x)
        corr = np.fft.irfft(np.conj(Xp) * self.buffer, n=self.capacity)
        return np.asarray(corr, dtype=np.float32) * float(self.read_gain)

    def score_probe(self, probe: np.ndarray) -> float:
        """Scalar match strength = max |correlation|."""
        c = self.correlation(probe)
        if c.size == 0:
            return 0.0
        return float(np.max(np.abs(c)))

    def score_key(self, key: str) -> float:
        return self.score_probe(key_to_vector(key, self.capacity))

    def top_matches(
        self,
        probe: np.ndarray,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Top-k lag indices by |correlation| score."""
        c = self.correlation(probe)
        if c.size == 0 or k <= 0:
            return []
        abs_c = np.abs(c)
        kk = min(int(k), abs_c.size)
        # argpartition for top-k
        idx = np.argpartition(-abs_c, kk - 1)[:kk]
        idx = sorted(idx, key=lambda i: -float(abs_c[i]))
        return [(int(i), float(abs_c[i])) for i in idx]

    def match_keys(
        self,
        keys: list[str],
        *,
        probe_key: Optional[str] = None,
        probe: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Score listed content keys against store (or against probe embedding)."""
        if probe is None:
            if probe_key is None:
                raise ValueError("need probe or probe_key")
            probe = key_to_vector(probe_key, self.capacity)
        # Self-similarity of probe to store (not pairwise): score each key by
        # writing-style probe of that key's embedding against the buffer.
        scored = [(ck, self.score_key(ck)) for ck in keys]
        scored.sort(key=lambda p: -p[1])
        return scored[: max(0, int(k))]

    def energy(self) -> float:
        return float(np.sum(np.abs(self.buffer) ** 2))

    def clear(self) -> None:
        self.buffer[:] = 0
        self.n_writes = 0

    def to_serializable(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "write_gain": float(self.write_gain),
            "read_gain": float(self.read_gain),
            "decay": float(self.decay),
            "n_writes": int(self.n_writes),
            "buffer_real": self.buffer.real.astype(np.float64).tolist(),
            "buffer_imag": self.buffer.imag.astype(np.float64).tolist(),
        }

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> "HolonomicStore":
        store = cls(
            capacity=int(data.get("capacity", 64)),
            pad_pow2=False,  # respect exact capacity from snapshot
            write_gain=float(data.get("write_gain", 1.0)),
            read_gain=float(data.get("read_gain", 1.0)),
            decay=float(data.get("decay", 0.002)),
        )
        # from_serializable may pad; force exact length
        re = np.asarray(data.get("buffer_real") or [], dtype=np.float64)
        im = np.asarray(data.get("buffer_imag") or [], dtype=np.float64)
        n = min(len(re), len(im), store.buffer.size)
        if n > 0:
            store.buffer[:n] = (re[:n] + 1j * im[:n]).astype(np.complex64)
        store.n_writes = int(data.get("n_writes", 0))
        return store

    def as_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "n_writes": self.n_writes,
            "energy": self.energy(),
        }
