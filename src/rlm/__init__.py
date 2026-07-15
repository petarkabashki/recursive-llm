"""Recursive Language Models for unbounded context processing."""

from .budget import RunBudget
from .core import RLM
from .errors import BudgetExceededError, MaxDepthError, MaxIterationsError, RLMError
from .repl import REPLError, REPLTimeoutError, WorkerResourceLimits
from .results import CompletionResult, TrajectoryEvent

__version__ = "0.1.0"

__all__ = [
    "RLM",
    "RLMError",
    "MaxIterationsError",
    "MaxDepthError",
    "BudgetExceededError",
    "RunBudget",
    "CompletionResult",
    "TrajectoryEvent",
    "REPLError",
    "REPLTimeoutError",
    "WorkerResourceLimits",
]
