"""
Action Executor for Agents
Executes planned steps and handles errors
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .tools import ToolRegistry, ToolResult
from .planner import Step, StepType


@dataclass
class ExecutionResult:
    """Result of executing a step"""
    step_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    start_time: datetime = None
    end_time: datetime = None
    duration: float = 0.0
    retries: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ActionExecutor:
    """
    Executes planned actions
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_parallel: int = 5,
        timeout: float = 30.0
    ):
        self.tool_registry = tool_registry
        self.max_parallel = max_parallel
        self.timeout = timeout
        
        # Execution tracking
        self.execution_history = []
        self.active_executions = {}
        
        # Thread pool for parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=max_parallel)
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    async def execute_step(
        self,
        step: Step,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a single step"""
        start_time = datetime.now()
        
        # Track active execution
        self.active_executions[step.id] = {
            'step': step,
            'start_time': start_time,
            'context': context
        }
        
        try:
            # Route to appropriate executor
            if step.type == StepType.TOOL_USE:
                result = await self._execute_tool_step(step, context)
            elif step.type == StepType.REASONING:
                result = await self._execute_reasoning_step(step, context)
            elif step.type == StepType.DECISION:
                result = await self._execute_decision_step(step, context)
            elif step.type == StepType.PARALLEL:
                result = await self._execute_parallel_step(step, context)
            elif step.type == StepType.LOOP:
                result = await self._execute_loop_step(step, context)
            elif step.type == StepType.CONDITIONAL:
                result = await self._execute_conditional_step(step, context)
            else:
                raise ValueError(f"Unknown step type: {step.type}")
            
            # Create execution result
            end_time = datetime.now()
            execution_result = ExecutionResult(
                step_id=step.id,
                success=True,
                result=result,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                metadata={'step_type': step.type.value}
            )
            
        except Exception as e:
            # Handle errors
            end_time = datetime.now()
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            self.logger.error(f"Error executing step {step.id}: {error_msg}")
            self.logger.debug(traceback.format_exc())
            
            execution_result = ExecutionResult(
                step_id=step.id,
                success=False,
                error=error_msg,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                metadata={'step_type': step.type.value, 'traceback': traceback.format_exc()}
            )
            
            # Retry if configured
            if step.retry_on_failure and execution_result.retries < step.max_retries:
                self.logger.info(f"Retrying step {step.id} (attempt {execution_result.retries + 1}/{step.max_retries})")
                await asyncio.sleep(2 ** execution_result.retries)  # Exponential backoff
                execution_result.retries += 1
                return await self.execute_step(step, context)
        
        finally:
            # Clean up active execution
            if step.id in self.active_executions:
                del self.active_executions[step.id]
        
        # Store in history
        self.execution_history.append(execution_result)
        
        return execution_result
    
    async def _execute_tool_step(
        self,
        step: Step,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a tool use step"""
        tool_name = step.action
        
        # Get tool from registry
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Prepare parameters
        params = step.parameters.copy()
        
        # Inject context if needed
        if context:
            for key, value in context.items():
                if key.startswith(f"step_") and "_result" in key:
                    # Make previous results available
                    params[f"previous_{key}"] = value
        
        # Execute tool with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(**params),
                timeout=self.timeout
            )
            
            if not result.success:
                raise RuntimeError(f"Tool execution failed: {result.error}")
            
            return result.data
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool '{tool_name}' execution timed out after {self.timeout}s")
    
    async def _execute_reasoning_step(
        self,
        step: Step,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a reasoning step"""
        # This would use the model to perform reasoning
        # For now, return a simulated result
        
        action = step.action
        params = step.parameters
        
        if action == "analyze_task":
            return {
                "task_type": "complex",
                "requirements": ["research", "synthesis"],
                "estimated_complexity": 0.7
            }
        elif action == "synthesize":
            return {
                "synthesis": f"Synthesized results for: {params.get('task', 'unknown task')}",
                "confidence": 0.85
            }
        elif action == "analyze_results":
            return {
                "key_findings": ["Finding 1", "Finding 2"],
                "quality_score": 0.8
            }
        else:
            return {"reasoning_complete": True, "action": action}
    
    async def _execute_decision_step(
        self,
        step: Step,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a decision step"""
        # This would use the model to make decisions
        # For now, return a simulated decision
        
        options = step.parameters.get("options", [])
        criteria = step.parameters.get("criteria", {})
        
        # Simulate decision making
        if options:
            # Pick best option (simplified)
            decision = {
                "selected": options[0] if options else None,
                "reasoning": "Selected based on criteria",
                "confidence": 0.75
            }
        else:
            decision = {
                "decision": "proceed",
                "reasoning": "Conditions met",
                "confidence": 0.8
            }
        
        return decision
    
    async def _execute_parallel_step(
        self,
        step: Step,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute multiple steps in parallel"""
        parallel_steps = step.parameters.get("steps", [])
        
        if not parallel_steps:
            return {"parallel_results": []}
        
        # Execute steps in parallel
        tasks = []
        for sub_step in parallel_steps:
            task = asyncio.create_task(self.execute_step(sub_step, context))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        parallel_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                parallel_results.append({
                    "step_id": parallel_steps[i].id,
                    "success": False,
                    "error": str(result)
                })
            else:
                parallel_results.append({
                    "step_id": parallel_steps[i].id,
                    "success": result.success,
                    "data": result.result if result.success else result.error
                })
        
        return {"parallel_results": parallel_results}
    
    async def _execute_loop_step(
        self,
        step: Step,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a loop step"""
        loop_steps = step.parameters.get("steps", [])
        condition = step.parameters.get("condition", {})
        max_iterations = step.parameters.get("max_iterations", 10)
        
        results = []
        iteration = 0
        
        while iteration < max_iterations:
            # Check condition
            if not self._evaluate_condition(condition, context, results):
                break
            
            # Execute loop body
            iteration_results = []
            for sub_step in loop_steps:
                result = await self.execute_step(sub_step, context)
                iteration_results.append(result)
            
            results.append({
                "iteration": iteration,
                "results": iteration_results
            })
            
            iteration += 1
        
        return {
            "loop_completed": True,
            "iterations": iteration,
            "results": results
        }
    
    async def _execute_conditional_step(
        self,
        step: Step,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a conditional step"""
        condition = step.parameters.get("condition", {})
        then_steps = step.parameters.get("then_steps", [])
        else_steps = step.parameters.get("else_steps", [])
        
        # Evaluate condition
        condition_met = self._evaluate_condition(condition, context)
        
        # Execute appropriate branch
        branch_steps = then_steps if condition_met else else_steps
        results = []
        
        for sub_step in branch_steps:
            result = await self.execute_step(sub_step, context)
            results.append(result)
        
        return {
            "condition_met": condition_met,
            "branch": "then" if condition_met else "else",
            "results": results
        }
    
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        loop_results: Optional[List[Any]] = None
    ) -> bool:
        """Evaluate a condition"""
        # Simple condition evaluation
        condition_type = condition.get("type", "always")
        
        if condition_type == "always":
            return True
        elif condition_type == "never":
            return False
        elif condition_type == "iterations":
            max_iter = condition.get("max", 10)
            current = len(loop_results) if loop_results else 0
            return current < max_iter
        elif condition_type == "context_value":
            key = condition.get("key")
            expected = condition.get("value")
            if context and key in context:
                return context[key] == expected
        elif condition_type == "result_check":
            if loop_results and loop_results[-1]:
                last_result = loop_results[-1]
                check_key = condition.get("key", "success")
                expected = condition.get("value", True)
                
                # Navigate nested results
                value = last_result
                for part in check_key.split('.'):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return False
                
                return value == expected
        
        return True
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active executions"""
        return self.active_executions.copy()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "average_duration": 0.0
            }
        
        successful = sum(1 for e in self.execution_history if e.success)
        failed = len(self.execution_history) - successful
        avg_duration = sum(e.duration for e in self.execution_history) / len(self.execution_history)
        
        # Group by step type
        type_stats = {}
        for execution in self.execution_history:
            step_type = execution.metadata.get('step_type', 'unknown')
            if step_type not in type_stats:
                type_stats[step_type] = {
                    'count': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_duration': 0.0
                }
            
            type_stats[step_type]['count'] += 1
            if execution.success:
                type_stats[step_type]['successful'] += 1
            else:
                type_stats[step_type]['failed'] += 1
            type_stats[step_type]['total_duration'] += execution.duration
        
        return {
            "total_executions": len(self.execution_history),
            "successful": successful,
            "failed": failed,
            "average_duration": avg_duration,
            "by_type": type_stats,
            "currently_active": len(self.active_executions)
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.thread_pool.shutdown(wait=True)