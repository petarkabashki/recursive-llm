"""Core Recursive Language Model implementation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from threading import Lock
from typing import Any, Callable, Coroutine, Dict, List, NoReturn, Optional, Sequence, TypeVar, cast

import litellm

from .budget import RunBudget
from .errors import BudgetExceededError, MaxDepthError, MaxIterationsError, RLMError
from .parser import extract_final, extract_final_var_name
from .prompts import build_system_prompt
from .repl import REPLError, REPLExecutor, WorkerResourceLimits
from .results import CompletionResult, TrajectoryEvent
from .run_state import RunState
from .stats import UsageTracker
from .types import Message

T = TypeVar("T")


def _run_sync(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run an awaitable from synchronous code, including inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(awaitable)).result()


class RLM:
    """Recursive Language Model with paper-aligned depth semantics."""

    def __init__(
        self,
        model: str,
        recursive_model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_depth: int = 1,
        max_iterations: int = 30,
        repl_timeout: float = 5,
        max_output_chars: int = 2000,
        repl_memory_limit_mb: Optional[int] = None,
        repl_cpu_time_limit_seconds: Optional[int] = None,
        repl_max_open_files: Optional[int] = None,
        max_concurrent_subcalls: int = 4,
        max_total_calls: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        max_total_cost_usd: Optional[float] = None,
        max_elapsed_seconds: Optional[float] = None,
        capture_trajectory_content: bool = False,
        event_handler: Optional[Callable[[TrajectoryEvent], None]] = None,
        _current_depth: int = 0,
        _run_state: Optional[RunState] = None,
        _node_id: str = "",
        _parent_node_id: str = "",
        **llm_kwargs: Any,
    ) -> None:
        """Initialize an RLM.

        ``max_depth`` describes available subcall capability, not the number of
        RLM objects. At depth 0 the root has a REPL but no subcalls. At depth 1
        it can call a plain LM. At depth 2 it can create one child RLM, whose
        boundary falls back to a plain LM call.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")
        if repl_timeout <= 0:
            raise ValueError("repl_timeout must be greater than zero")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than zero")
        if max_concurrent_subcalls <= 0:
            raise ValueError("max_concurrent_subcalls must be greater than zero")
        if _current_depth < 0:
            raise ValueError("_current_depth must be zero or greater")

        self.model = model
        self.recursive_model = recursive_model or model
        self.api_base = api_base
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.repl_timeout = repl_timeout
        self.max_output_chars = max_output_chars
        self.repl_resource_limits = WorkerResourceLimits(
            memory_mb=repl_memory_limit_mb,
            cpu_time_seconds=repl_cpu_time_limit_seconds,
            max_open_files=repl_max_open_files,
        )
        self.max_concurrent_subcalls = max_concurrent_subcalls
        self.max_total_calls = max_total_calls
        self.max_total_tokens = max_total_tokens
        self.max_total_cost_usd = max_total_cost_usd
        self.max_elapsed_seconds = max_elapsed_seconds
        self.capture_trajectory_content = capture_trajectory_content
        self.event_handler = event_handler
        self._current_depth = _current_depth
        self._inherited_run_state = _run_state
        self._inherited_node_id = _node_id
        self._inherited_parent_node_id = _parent_node_id
        self.llm_kwargs = llm_kwargs
        self._state_lock = Lock()
        self._direct_run_state = self._new_run_state()
        self._last_run_state = self._direct_run_state

    def complete(self, query: str = "", context: str = "", **kwargs: Any) -> str:
        """Synchronously complete a query over an external context."""
        return _run_sync(self.acomplete(query, context, **kwargs))

    def complete_result(
        self, query: str = "", context: str = "", **kwargs: Any
    ) -> CompletionResult:
        """Synchronously return an answer with exact stats and trajectory."""
        return _run_sync(self.acomplete_result(query, context, **kwargs))

    async def acomplete(self, query: str = "", context: str = "", **kwargs: Any) -> str:
        """Complete a query and return only its answer for compatibility."""
        result = await self.acomplete_result(query, context, **kwargs)
        return result.answer

    async def acomplete_result(
        self, query: str = "", context: str = "", **kwargs: Any
    ) -> CompletionResult:
        """Complete a query and return structured per-run diagnostics."""
        if query and not context:
            context = query
            query = ""

        if self._current_depth > 0 and self._current_depth >= self.max_depth:
            raise MaxDepthError(
                f"RLM depth {self._current_depth} is not available with max_depth={self.max_depth}"
            )

        loop = asyncio.get_running_loop()
        if self._current_depth == 0:
            run_state = self._new_run_state(loop)
        else:
            run_state = self._inherited_run_state or self._new_run_state(loop)
            run_state.attach_loop(loop)

        node_id = self._inherited_node_id or run_state.next_node_id("rlm")
        parent_id = self._inherited_parent_node_id
        start_data = self._content_data(query=query, context=context)
        if self._current_depth == 0:
            run_state.record_event("run_start", 0, node_id, **start_data)
        run_state.record_event("rlm_start", self._current_depth, node_id, parent_id, **start_data)

        try:
            answer = await self._acomplete_impl(query, context, kwargs, run_state, node_id)
        except BaseException as exc:
            run_state.record_event(
                "rlm_error",
                self._current_depth,
                node_id,
                parent_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if self._current_depth == 0:
                run_state.record_event(
                    "run_error",
                    0,
                    node_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        else:
            answer_data = self._content_data(answer=answer)
            run_state.record_event(
                "rlm_end", self._current_depth, node_id, parent_id, **answer_data
            )
            if self._current_depth == 0:
                run_state.record_event("run_end", 0, node_id, **answer_data)
            return CompletionResult(
                answer=answer,
                stats=self._stats_snapshot(run_state),
                trajectory=run_state.trajectory(),
            )
        finally:
            self._publish_run_state(run_state)

    async def _acomplete_impl(
        self,
        query: str,
        context: str,
        kwargs: Dict[str, Any],
        run_state: RunState,
        node_id: str,
    ) -> str:
        """Run one RLM loop within an already-created invocation state."""

        repl_env = self._build_repl_env(query, context, run_state, node_id)
        repl = REPLExecutor(
            timeout=self.repl_timeout,
            max_output_chars=self.max_output_chars,
            resource_limits=self.repl_resource_limits,
        )
        system_prompt = build_system_prompt(
            len(context),
            depth=self._current_depth,
            max_depth=self.max_depth,
        )
        messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            for iteration in range(self.max_iterations):
                self._check_budget_deadline(run_state)
                run_state.record_iteration(self._current_depth)
                response = await self._call_llm(
                    messages, _run_state=run_state, _node_id=node_id, **kwargs
                )

                if not response.strip():
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your response was empty. Return one short executable Python step "
                                "or a standalone final directive."
                            ),
                        }
                    )
                    continue

                direct_answer = extract_final(response)
                if direct_answer is not None:
                    run_state.record_event(
                        "final_answer",
                        self._current_depth,
                        node_id,
                        method="directive",
                        **self._content_data(answer=direct_answer),
                    )
                    return direct_answer

                final_var_name = extract_final_var_name(response)
                if final_var_name is not None:
                    try:
                        found, value = await asyncio.to_thread(repl.get_variable, final_var_name)
                    except REPLError:
                        if final_var_name in repl_env:
                            answer = str(repl_env[final_var_name])
                            run_state.record_event(
                                "final_answer",
                                self._current_depth,
                                node_id,
                                method="parent_snapshot",
                                variable=final_var_name,
                                **self._content_data(answer=answer),
                            )
                            return answer
                    else:
                        if found:
                            answer = str(value)
                            run_state.record_event(
                                "final_answer",
                                self._current_depth,
                                node_id,
                                method="worker_variable",
                                variable=final_var_name,
                                **self._content_data(answer=answer),
                            )
                            return answer
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Error: variable {final_var_name!r} was not found",
                        }
                    )
                    continue

                try:
                    exec_result = await asyncio.to_thread(repl.execute, response, repl_env)
                    self._record_repl_step(run_state, node_id, response, exec_result, status="ok")
                    published_answer = repl.pop_final_answer()
                    if published_answer is not None:
                        run_state.record_event(
                            "final_answer",
                            self._current_depth,
                            node_id,
                            method="answer_object",
                            **self._content_data(answer=published_answer),
                        )
                        return published_answer
                except REPLError as exc:
                    exec_result = f"Error: {exc}"
                    self._record_repl_step(
                        run_state, node_id, response, exec_result, status="error"
                    )
                except BudgetExceededError:
                    raise
                except Exception as exc:
                    exec_result = f"Unexpected error: {exc}"
                    self._record_repl_step(
                        run_state, node_id, response, exec_result, status="error"
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": exec_result})
        finally:
            await asyncio.to_thread(repl.close)

        raise MaxIterationsError(
            f"Max iterations ({self.max_iterations}) exceeded without a final answer"
        )

    def _content_data(self, **values: str) -> Dict[str, Any]:
        """Return content or redacted lengths for trajectory payloads."""
        if self.capture_trajectory_content:
            return dict(values)
        return {f"{name}_chars": len(value) for name, value in values.items()}

    def _record_repl_step(
        self,
        run_state: RunState,
        node_id: str,
        code: str,
        output: str,
        *,
        status: str,
    ) -> None:
        """Record one restricted-code step with content redaction by default."""
        run_state.record_event(
            "repl_step",
            self._current_depth,
            node_id,
            status=status,
            **self._content_data(code=code, output=output),
        )

    async def _call_llm(
        self,
        messages: List[Message],
        *,
        _run_state: Optional[RunState] = None,
        _node_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Call the root or child RLM model and record its usage."""
        run_state = self._state_for_call(_run_state)
        parent_node_id = _node_id or run_state.next_node_id("direct")
        default_model = self.model if self._current_depth == 0 else self.recursive_model
        model = cast(str, kwargs.get("model", default_model))
        call_overrides = dict(kwargs)
        call_overrides.pop("model", None)
        return await self._run_model_call(
            run_state,
            parent_node_id,
            model,
            self._current_depth,
            messages,
            call_overrides,
            is_leaf=False,
        )

    async def _call_leaf(
        self,
        sub_query: str,
        sub_context: str = "",
        model: Optional[str] = None,
        *,
        _run_state: Optional[RunState] = None,
        _node_id: str = "",
    ) -> str:
        """Call a plain LM without creating another REPL loop."""
        run_state = self._state_for_call(_run_state)
        parent_node_id = _node_id or run_state.next_node_id("direct")
        selected_model = model or self.recursive_model
        user_content = sub_query
        if sub_context:
            user_content = f"Task:\n{sub_query}\n\nContext:\n{sub_context}"
        messages: List[Message] = [
            {
                "role": "system",
                "content": (
                    "Answer the subproblem using only the supplied context. "
                    "Return the answer directly and do not emit REPL code or FINAL directives."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        call_depth = self._current_depth + 1
        return await self._run_model_call(
            run_state,
            parent_node_id,
            selected_model,
            call_depth,
            messages,
            {},
            is_leaf=True,
        )

    async def _run_model_call(
        self,
        run_state: RunState,
        parent_node_id: str,
        model: str,
        depth: int,
        messages: List[Message],
        overrides: Dict[str, Any],
        *,
        is_leaf: bool,
    ) -> str:
        """Execute and trace one root, recursive, or leaf model call."""
        call_id = run_state.next_node_id("llm")
        request_data: Dict[str, Any] = {
            "model": model,
            "is_leaf": is_leaf,
            "message_count": len(messages),
        }
        if self.capture_trajectory_content:
            request_data["messages"] = [dict(message) for message in messages]
        else:
            request_data["message_chars"] = sum(len(message["content"]) for message in messages)
        run_state.record_event("model_call_start", depth, call_id, parent_node_id, **request_data)
        try:
            self._reserve_model_call(run_state, model, depth, is_leaf=is_leaf)
            response = await self._request_completion(run_state, model, messages, overrides)
            self._record_response(run_state, model, response)
            text = self._response_text(response)
        except BaseException as exc:
            run_state.record_event(
                "model_call_error",
                depth,
                call_id,
                parent_node_id,
                model=model,
                is_leaf=is_leaf,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        run_state.record_event(
            "model_call_end",
            depth,
            call_id,
            parent_node_id,
            model=model,
            is_leaf=is_leaf,
            **self._content_data(response=text),
        )
        return text

    def _completion_kwargs(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge common provider arguments for one LiteLLM request."""
        call_kwargs = {**self.llm_kwargs, **overrides}
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        return call_kwargs

    async def _request_completion(
        self,
        run_state: RunState,
        model: str,
        messages: List[Message],
        overrides: Dict[str, Any],
    ) -> Any:
        """Make one provider request within the remaining run deadline."""
        remaining = run_state.budget.remaining_seconds()
        if remaining is None:
            return await litellm.acompletion(
                model=model,
                messages=messages,
                **self._completion_kwargs(overrides),
            )
        if remaining <= 0:
            self._check_budget_deadline(run_state)
        awaitable = litellm.acompletion(
            model=model,
            messages=messages,
            **self._completion_kwargs(overrides),
        )
        try:
            return await asyncio.wait_for(awaitable, timeout=remaining)
        except asyncio.TimeoutError as exc:
            self._raise_budget_error(
                run_state,
                BudgetExceededError(
                    "elapsed_seconds",
                    cast(float, run_state.budget.max_elapsed_seconds),
                    cast(float, run_state.budget.max_elapsed_seconds),
                ),
            )
            raise AssertionError("unreachable") from exc

    def _reserve_model_call(
        self,
        run_state: RunState,
        model: str,
        depth: int,
        *,
        is_leaf: bool = False,
    ) -> None:
        """Reserve and record one provider request for the shared tree."""
        try:
            run_state.budget.reserve_call()
        except BudgetExceededError as exc:
            self._raise_budget_error(run_state, exc)
        run_state.usage.record_call(model, depth, is_leaf=is_leaf)

    def _check_budget_deadline(self, run_state: RunState) -> None:
        try:
            run_state.budget.check_deadline()
        except BudgetExceededError as exc:
            self._raise_budget_error(run_state, exc)

    def _raise_budget_error(self, run_state: RunState, error: BudgetExceededError) -> NoReturn:
        """Attach partial tree statistics before surfacing a budget error."""
        error.stats = self._stats_snapshot(run_state)
        raise error

    def _new_run_budget(self) -> RunBudget:
        """Create a fresh budget for one root completion tree."""
        return RunBudget(
            max_calls=self.max_total_calls,
            max_tokens=self.max_total_tokens,
            max_cost_usd=self.max_total_cost_usd,
            max_elapsed_seconds=self.max_elapsed_seconds,
        )

    def _new_run_state(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> RunState:
        """Create isolated mutable state for one root invocation."""
        return RunState(
            UsageTracker(), self._new_run_budget(), loop, event_handler=self.event_handler
        )

    def _state_for_call(self, run_state: Optional[RunState]) -> RunState:
        """Resolve explicit, inherited, or protected-helper state."""
        state = run_state or self._inherited_run_state or self._direct_run_state
        try:
            state.attach_loop(asyncio.get_running_loop())
        except RuntimeError:
            pass
        if run_state is None and self._inherited_run_state is None:
            self._publish_run_state(state)
        return state

    def _publish_run_state(self, run_state: RunState) -> None:
        """Expose the most recently completed or directly used state via stats."""
        with self._state_lock:
            self._last_run_state = run_state

    def _record_response(self, run_state: RunState, model: str, response: Any) -> None:
        run_state.usage.record_response(model, response, self._get_response_cost(response))
        stats = run_state.usage.snapshot()
        try:
            run_state.budget.record_usage(stats["total_tokens"], stats["estimated_cost_usd"])
        except BudgetExceededError as exc:
            self._raise_budget_error(run_state, exc)

    @staticmethod
    def _response_text(response: Any) -> str:
        content = response.choices[0].message.content
        if content is None:
            raise RLMError("LLM response did not contain text content")
        return cast(str, content)

    @staticmethod
    def _get_response_cost(response: Any) -> Optional[float]:
        """Return LiteLLM's best-effort response cost without affecting completion."""
        hidden_params = getattr(response, "_hidden_params", None)
        if isinstance(hidden_params, dict):
            response_cost = hidden_params.get("response_cost")
            if isinstance(response_cost, (int, float)):
                return float(response_cost)
        try:
            response_cost = litellm.completion_cost(completion_response=response)
        except Exception:
            return None
        if isinstance(response_cost, (int, float)):
            return float(response_cost)
        return None

    def _build_repl_env(
        self,
        query: str,
        context: str,
        run_state: Optional[RunState] = None,
        node_id: str = "",
    ) -> Dict[str, Any]:
        """Build the names exposed to restricted Python code."""
        state = self._state_for_call(run_state)
        rlm_node_id = node_id or state.next_node_id("rlm")
        env: Dict[str, Any] = {
            "context": context,
            "query": query,
            "answer": {"content": "", "ready": False},
            "re": re,
        }
        if self.max_depth == 0:
            return env

        llm_query = self._make_llm_query(state, rlm_node_id)
        rlm_query = self._make_rlm_query(state, rlm_node_id)
        env.update(
            {
                "llm_query": llm_query,
                "rlm_query": rlm_query,
                "recursive_llm": rlm_query,
                "llm_query_batched": self._make_batched_query(llm_query, state),
                "rlm_query_batched": self._make_batched_query(rlm_query, state),
            }
        )
        return env

    def _make_llm_query(
        self, run_state: Optional[RunState] = None, node_id: str = ""
    ) -> Callable[..., str]:
        """Create the direct plain-LM function exposed in the REPL."""
        state = self._state_for_call(run_state)
        parent_node_id = node_id or state.next_node_id("rlm")

        def llm_query(
            sub_query: str,
            sub_context: str = "",
            model: Optional[str] = None,
        ) -> str:
            return self._run_callback(
                self._call_leaf(
                    sub_query,
                    sub_context,
                    model,
                    _run_state=state,
                    _node_id=parent_node_id,
                ),
                state,
            )

        return llm_query

    def _make_rlm_query(
        self, run_state: Optional[RunState] = None, node_id: str = ""
    ) -> Callable[[str, str], str]:
        """Create the recursive function with a plain-LM boundary fallback."""
        state = self._state_for_call(run_state)
        parent_node_id = node_id or state.next_node_id("rlm")

        async def call(sub_query: str, sub_context: str = "") -> str:
            if self._current_depth + 1 >= self.max_depth:
                return await self._call_leaf(
                    sub_query,
                    sub_context,
                    _run_state=state,
                    _node_id=parent_node_id,
                )

            child_node_id = state.next_node_id("rlm")
            child = RLM(
                model=self.recursive_model,
                recursive_model=self.recursive_model,
                api_base=self.api_base,
                api_key=self.api_key,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                repl_timeout=self.repl_timeout,
                max_output_chars=self.max_output_chars,
                repl_memory_limit_mb=self.repl_resource_limits.memory_mb,
                repl_cpu_time_limit_seconds=self.repl_resource_limits.cpu_time_seconds,
                repl_max_open_files=self.repl_resource_limits.max_open_files,
                max_concurrent_subcalls=self.max_concurrent_subcalls,
                max_total_calls=self.max_total_calls,
                max_total_tokens=self.max_total_tokens,
                max_total_cost_usd=self.max_total_cost_usd,
                max_elapsed_seconds=self.max_elapsed_seconds,
                capture_trajectory_content=self.capture_trajectory_content,
                event_handler=self.event_handler,
                _current_depth=self._current_depth + 1,
                _run_state=state,
                _node_id=child_node_id,
                _parent_node_id=parent_node_id,
                **self.llm_kwargs,
            )
            return await child.acomplete(sub_query, sub_context)

        def rlm_query(sub_query: str, sub_context: str = "") -> str:
            return self._run_callback(call(sub_query, sub_context), state)

        return rlm_query

    def _make_batched_query(
        self,
        query_fn: Callable[[str, str], str],
        run_state: Optional[RunState] = None,
    ) -> Callable[[Sequence[str], Optional[Sequence[str]]], List[str]]:
        """Create an ordered, bounded-concurrency batch wrapper."""
        state = self._state_for_call(run_state)

        async def run_batch(
            queries: Sequence[str],
            contexts: Optional[Sequence[str]],
        ) -> List[str]:
            query_list = list(queries)
            context_list = list(contexts) if contexts is not None else [""] * len(query_list)
            if len(query_list) != len(context_list):
                raise ValueError("queries and contexts must have the same length")
            semaphore = asyncio.Semaphore(self.max_concurrent_subcalls)

            async def run_one(item_query: str, item_context: str) -> str:
                async with semaphore:
                    try:
                        return await asyncio.to_thread(query_fn, item_query, item_context)
                    except BudgetExceededError:
                        raise
                    except Exception as exc:
                        return f"Error: {exc}"

            return await asyncio.gather(
                *(
                    run_one(item_query, item_context)
                    for item_query, item_context in zip(query_list, context_list)
                )
            )

        def batched(
            queries: Sequence[str],
            contexts: Optional[Sequence[str]] = None,
        ) -> List[str]:
            return self._run_callback(run_batch(queries, contexts), state)

        return batched

    def _run_callback(self, awaitable: Coroutine[Any, Any, T], run_state: RunState) -> T:
        """Run a REPL callback on its owning completion loop when available."""
        loop = run_state.loop
        if loop is not None and loop.is_running():
            return asyncio.run_coroutine_threadsafe(awaitable, loop).result()
        return _run_sync(awaitable)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for the latest full recursion tree."""
        with self._state_lock:
            run_state = self._last_run_state
        return self._stats_snapshot(run_state)

    def _stats_snapshot(self, run_state: RunState) -> Dict[str, Any]:
        """Return current completion-tree statistics, including its budget."""
        stats = run_state.usage.snapshot()
        stats["iterations"] = run_state.iterations_at(self._current_depth)
        stats["depth"] = self._current_depth
        stats["budget"] = run_state.budget.snapshot()
        return stats
