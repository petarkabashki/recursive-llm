"""Concurrency tests for reusable RLM instances."""

import asyncio
import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from rlm import RLM


class MockResponse:
    """Minimal LiteLLM-compatible response."""

    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = None
        self._hidden_params = {}


@pytest.mark.asyncio
async def test_same_instance_async_completions_keep_usage_and_repl_state_isolated() -> None:
    """Interleaved invocations must not reset or aggregate each other's state."""

    async def completion(*, messages, **_kwargs):
        query = messages[1]["content"]
        await asyncio.sleep(0.03 if query == "slow" else 0.01)
        if len(messages) == 2:
            return MockResponse("value = query")
        return MockResponse("FINAL_VAR(value)")

    with patch("rlm.core.litellm.acompletion", side_effect=completion) as mocked:
        rlm = RLM(model="test-model")
        results = await asyncio.gather(
            rlm.acomplete("slow", "Shared context"),
            rlm.acomplete("fast", "Shared context"),
        )

    assert results == ["slow", "fast"]
    assert mocked.call_count == 4
    assert rlm.stats["llm_calls"] == 2
    assert rlm.stats["iterations"] == 2


def test_same_instance_sync_completions_use_their_own_callback_loops() -> None:
    """Concurrent sync calls from separate threads must route callbacks correctly."""

    async def completion(*, messages, **_kwargs):
        await asyncio.sleep(0.02)
        system_prompt = messages[0]["content"]
        if system_prompt.startswith("Answer the subproblem"):
            task = messages[1]["content"].splitlines()[1]
            return MockResponse(f"leaf:{task}")
        if len(messages) == 2:
            return MockResponse("result = llm_query(query, context)")
        return MockResponse("FINAL_VAR(result)")

    with patch("rlm.core.litellm.acompletion", side_effect=completion) as mocked:
        rlm = RLM(model="root", recursive_model="leaf", max_depth=1, repl_timeout=3)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(rlm.complete, query, "Shared context")
                for query in ("alpha", "beta")
            ]
            results = [future.result(timeout=15) for future in futures]

    assert results == ["leaf:alpha", "leaf:beta"]
    assert mocked.call_count == 6
    assert rlm.stats["llm_calls"] == 3
    assert rlm.stats["root_calls"] == 2
    assert rlm.stats["leaf_calls"] == 1
