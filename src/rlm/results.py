"""Structured completion and trajectory results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TrajectoryEvent:
    """One ordered event from an RLM completion tree."""

    sequence: int
    kind: str
    elapsed_seconds: float
    depth: int
    node_id: str
    parent_id: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-serializable representation."""
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "elapsed_seconds": self.elapsed_seconds,
            "depth": self.depth,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "data": deepcopy(self.data),
        }


@dataclass(frozen=True)
class CompletionResult:
    """Answer, exact per-run usage, and the full recursion trajectory."""

    answer: str
    stats: Dict[str, Any]
    trajectory: Tuple[TrajectoryEvent, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-serializable representation."""
        return {
            "answer": self.answer,
            "stats": deepcopy(self.stats),
            "trajectory": [event.to_dict() for event in self.trajectory],
        }
