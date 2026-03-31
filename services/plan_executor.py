"""
AERIS Plan Executor
Executes multi-step plans produced by the Planner.
Passes output from step N as context into step N+1.
Supports: sequential execution, progress reporting, step-level rollback.

Example plan:
    Step 1: open_browser
    Step 2: web_scrape (url from step 1 context)
    Step 3: generate_report (content from step 2)
    Step 4: file_write (path from step 3)
"""

import logging
import time
from typing import List, Dict, Optional, Callable
from services.tool_dispatcher import ToolDispatcher
from services.tool_base import ToolResult

logger = logging.getLogger(__name__)


class StepResult:
    def __init__(self, step: int, action: str, result: ToolResult, duration: float):
        self.step = step
        self.action = action
        self.result = result
        self.duration = duration

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "success": self.result.success,
            "data": self.result.data,
            "error": self.result.error,
            "duration": round(self.duration, 3)
        }


class PlanExecutionResult:
    def __init__(self, steps: List[StepResult], success: bool, total_time: float, stopped_at: Optional[int] = None):
        self.steps = steps
        self.success = success
        self.total_time = total_time
        self.stopped_at = stopped_at

    @property
    def last_data(self) -> dict:
        """Get output data from the last successful step."""
        for step in reversed(self.steps):
            if step.result.success and step.result.data:
                return step.result.data
        return {}

    def to_summary(self) -> dict:
        return {
            "success": self.success,
            "steps_total": len(self.steps),
            "steps_succeeded": sum(1 for s in self.steps if s.result.success),
            "total_time": round(self.total_time, 3),
            "stopped_at": self.stopped_at,
            "steps": [s.to_dict() for s in self.steps],
            "final_output": self.last_data
        }


class PlanExecutor:
    """
    Executes a multi-step plan produced by the Planner.
    Chains step outputs as context for subsequent steps.
    """

    def __init__(self, dispatcher: Optional[ToolDispatcher] = None):
        self.dispatcher = dispatcher or ToolDispatcher()
        self.progress_callback: Optional[Callable] = None

    def execute_plan(
        self,
        plan: List[dict],
        initial_context: dict = None
    ) -> PlanExecutionResult:
        """
        Execute all steps in the plan sequentially.

        Args:
            plan: List of step dicts from Planner.create_plan()
            initial_context: Optional starting context (e.g., user input, session)

        Returns:
            PlanExecutionResult with full step history
        """
        start_time = time.time()
        step_results: List[StepResult] = []
        context = dict(initial_context or {})

        logger.info(f"[PlanExecutor] Starting execution of {len(plan)}-step plan")

        for step_def in plan:
            step_num = step_def.get("step", len(step_results) + 1)
            action = step_def.get("action", "")
            params = dict(step_def.get("params", {}))

            if not action or action in {"respond", "clarify", "confirm", "cancel"}:
                # Non-dispatchable steps — handled upstream
                logger.info(f"[PlanExecutor] Step {step_num}: '{action}' is conversational, skipping dispatch")
                continue

            # Inject context from previous steps into params
            params = self._inject_context(params, context, action)

            logger.info(f"[PlanExecutor] Step {step_num}: executing '{action}' with params={params}")

            # Notify progress
            if self.progress_callback:
                self.progress_callback(step_num, len(plan), action)

            # Execute
            step_start = time.time()
            result = self.dispatcher.dispatch(action, params)
            step_duration = time.time() - step_start

            step_result = StepResult(step_num, action, result, step_duration)
            step_results.append(step_result)

            if result.success:
                # Store output in context for next steps
                if result.data:
                    context[f"step{step_num}_output"] = result.data
                    context["last_output"] = result.data
                    context["last_action"] = action
                    logger.info(f"[PlanExecutor] Step {step_num} succeeded")
            else:
                logger.warning(f"[PlanExecutor] Step {step_num} failed: {result.error}")
                # Stop on failure for now (could be made configurable)
                total_time = time.time() - start_time
                return PlanExecutionResult(
                    steps=step_results,
                    success=False,
                    total_time=total_time,
                    stopped_at=step_num
                )

        total_time = time.time() - start_time
        success = all(s.result.success for s in step_results)

        logger.info(f"[PlanExecutor] Plan complete. Success={success}, Time={total_time:.2f}s")

        return PlanExecutionResult(
            steps=step_results,
            success=success,
            total_time=total_time
        )

    def _inject_context(self, params: dict, context: dict, action: str) -> dict:
        """
        Smart context injection — passes relevant outputs from prior steps.

        Examples:
        - web_scrape step gets URL from web_navigate output
        - generate_report gets content from web_scrape output
        - file_write gets content from generate_report output
        """
        params = dict(params)
        last_output = context.get("last_output", {})

        if not last_output:
            return params

        action_lower = action.lower()

        # Web scraper can get URL from navigation step
        if action_lower in ("web_scrape", "scrape") and not params.get("url"):
            if "url" in last_output or "opened" in last_output:
                params["url"] = last_output.get("url") or last_output.get("opened", "")

        # Report generator gets content from scraper
        if action_lower in ("generate_report", "report") and not params.get("content"):
            if "plain_text" in last_output or "paragraphs" in last_output:
                params["content"] = last_output
                if not params.get("title"):
                    params["title"] = last_output.get("title", "AERIS Report")
                if not params.get("source_url"):
                    params["source_url"] = last_output.get("url", "")

        # File writer gets content from report or scraper
        if action_lower in ("file_write",) and not params.get("content"):
            if "content" in last_output:
                params["content"] = last_output["content"]
            elif "plain_text" in last_output:
                params["content"] = last_output["plain_text"]

        # Pass session context
        if "session_id" in context:
            params["_session_id"] = context["session_id"]

        return params

    def set_progress_callback(self, callback: Callable):
        """Set a callback for progress updates: callback(step_num, total_steps, action)"""
        self.progress_callback = callback
