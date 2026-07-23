"""Base Process — concurrent face worker thread."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from symbioid.Core.ids import _new_id

if TYPE_CHECKING:
    from symbioid.Core.Symbioid import Symbioid


@dataclass
class Process:
    """
    Base for concurrent process facades on a Symbioid.

    Call process() to start an independent worker thread. Subclasses override
    process() and must invoke super().process() before face-specific setup;
    override _process_body() for the actual loop run inside the thread.
    """

    id: str = field(default_factory=_new_id)
    label: Optional[str] = None
    enabled: bool = True
    # host Symbioid (set by Symbioid.__post_init__); not required at construct time
    host: Optional["Symbioid"] = field(default=None, repr=False)
    tick_interval: float = 0.05

    # runtime (not in init)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lifecycle_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _local_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    inbox: queue.Queue = field(default_factory=queue.Queue, init=False, repr=False)
    cycles: int = field(default=0, init=False, repr=False)
    last_error: Optional[str] = field(default=None, init=False, repr=False)

    def __repr__(self) -> str:
        lab = f" label={self.label!r}" if self.label else ""
        alive = self.is_alive()
        return (
            f"{type(self).__name__}(id={self.id!r}{lab} enabled={self.enabled} "
            f"alive={alive} cycles={self.cycles})"
        )

    def process(self) -> Optional[threading.Thread]:
        """
        Start independent processing on a new daemon thread.

        Subclasses that override process() must call super().process() first,
        then perform any face-specific startup (not the main loop — use
        _process_body for that).
        """
        if not self.enabled:
            return None
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._stop.clear()
            self.last_error = None
            name = f"{type(self).__name__}-{self.label or self.id[:8]}"
            self._thread = threading.Thread(
                target=self._thread_main,
                name=name,
                daemon=True,
            )
            self._thread.start()
            return self._thread

    def stop(self, timeout: Optional[float] = 1.0) -> None:
        """Signal the worker to stop and optionally join."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and timeout is not None:
            t.join(timeout=timeout)

    def is_alive(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def _thread_main(self) -> None:
        try:
            while not self._stop.is_set():
                if not self.enabled:
                    self._stop.wait(self.tick_interval)
                    continue
                try:
                    self._process_body()
                except Exception as exc:  # keep thread alive; record last error
                    self.last_error = f"{type(exc).__name__}: {exc}"
                self.cycles += 1
                self._stop.wait(self.tick_interval)
        finally:
            pass

    def _process_body(self) -> None:
        """
        One unit of work per tick. Override in subclasses.
        Base implementation drains inbox under local lock only (no host lock).
        """
        self._drain_inbox()

    def _drain_inbox(self, max_items: int = 32) -> list[Any]:
        """Non-blocking drain of thread-safe inbox (no host lock held)."""
        items: list[Any] = []
        for _ in range(max_items):
            try:
                items.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return items

    def post(self, message: Any) -> None:
        """Enqueue a message for this process (thread-safe)."""
        self.inbox.put(message)

    def with_host_lock(self):
        """
        Context manager for Symbioid shared data.
        Prefer short critical sections; never block on another Process lock
        while holding the host lock (deadlock avoidance).
        """
        if self.host is None:
            return self._local_lock
        return self.host.graph_lock
