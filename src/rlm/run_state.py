"""Per-invocation state shared within one RLM recursion tree."""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple

from .budget import RunBudget
from .results import TrajectoryEvent
from .stats import UsageTracker


class RunState:
    """Own usage, budget, loop, and iteration data for one completion tree."""

    def __init__(
        self,
        usage: UsageTracker,
        budget: RunBudget,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        event_handler: Optional[Callable[[TrajectoryEvent], None]] = None,
    ) -> None:
        self.usage = usage
        self.budget = budget
        self._loop = loop
        self._iterations_by_depth: Dict[int, int] = {}
        self._started_at = time.monotonic()
        self._next_node_number = 0
        self._events: list[TrajectoryEvent] = []
        self._event_handler = event_handler
        self._lock = Lock()

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Return the event loop that owns this invocation."""
        with self._lock:
            return self._loop

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach a loop when protected call helpers are used directly."""
        with self._lock:
            if self._loop is None:
                self._loop = loop

    def record_iteration(self, depth: int) -> None:
        """Record one iteration globally and for a specific RLM depth."""
        self.usage.record_iteration()
        with self._lock:
            self._iterations_by_depth[depth] = self._iterations_by_depth.get(depth, 0) + 1

    def iterations_at(self, depth: int) -> int:
        """Return loop iterations performed by one RLM depth."""
        with self._lock:
            return self._iterations_by_depth.get(depth, 0)

    def next_node_id(self, prefix: str) -> str:
        """Return a deterministic unique node identifier for this run."""
        with self._lock:
            self._next_node_number += 1
            return f"{prefix}-{self._next_node_number}"

    def record_event(
        self,
        kind: str,
        depth: int,
        node_id: str,
        parent_id: str = "",
        **data: Any,
    ) -> TrajectoryEvent:
        """Append an event and notify the optional best-effort handler."""
        with self._lock:
            event = TrajectoryEvent(
                sequence=len(self._events) + 1,
                kind=kind,
                elapsed_seconds=round(time.monotonic() - self._started_at, 6),
                depth=depth,
                node_id=node_id,
                parent_id=parent_id,
                data=deepcopy(data),
            )
            self._events.append(event)
        if self._event_handler is not None:
            try:
                self._event_handler(event)
            except Exception:
                pass
        return event

    def trajectory(self) -> Tuple[TrajectoryEvent, ...]:
        """Return a detached immutable snapshot of all events so far."""
        with self._lock:
            return tuple(
                TrajectoryEvent(
                    sequence=event.sequence,
                    kind=event.kind,
                    elapsed_seconds=event.elapsed_seconds,
                    depth=event.depth,
                    node_id=event.node_id,
                    parent_id=event.parent_id,
                    data=deepcopy(event.data),
                )
                for event in self._events
            )
