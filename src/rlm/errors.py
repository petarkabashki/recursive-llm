"""Public RLM exceptions."""

from __future__ import annotations

from typing import Any, Dict, Optional


class RLMError(Exception):
    """Base error for RLM."""


class MaxIterationsError(RLMError):
    """Maximum root or child RLM iterations exceeded."""


class MaxDepthError(RLMError):
    """An invalid internal RLM depth was requested."""


class BudgetExceededError(RLMError):
    """A shared completion-tree budget was exhausted."""

    abort_repl = True

    def __init__(self, metric: str, limit: float, observed: float) -> None:
        self.metric = metric
        self.limit = limit
        self.observed = observed
        self.stats: Optional[Dict[str, Any]] = None
        super().__init__(
            f"Run budget exceeded for {metric}: limit={limit:g}, observed={observed:g}"
        )
