"""
Agent Core Module
Central agent system for Jigyasa
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import asyncio
import logging

from .agent import Agent, AgentConfig
from .tools import ToolRegistry
from .memory import AgentMemory
from .planner import TaskPlanner, Plan
from .executor import ActionExecutor


class AgentCore(nn.Module):
    """
    Core agent system integrating all agentic components
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Initialize components
        self.config = AgentConfig()
        self.tool_registry = ToolRegistry()
        self.memory = AgentMemory(embedding_dim=model_dim)
        self.planner = TaskPlanner(self, config={'model_dim': model_dim})
        self.executor = ActionExecutor(self.tool_registry)
        
        # Agent instance - create dummy model for agent
        dummy_model = nn.Linear(model_dim, model_dim)
        
        # Get tools from registry
        tools = []
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                tools.append(tool)
        
        self.agent = Agent(
            name="jigyasa_agent",
            model=dummy_model,
            tools=tools,
            config=self.config
        )
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    async def plan_task(self, task: str) -> Plan:
        """Plan a task"""
        return await self.planner.create_plan(task)
    
    async def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Execute a plan"""
        results = []
        for step in plan.steps:
            result = await self.executor.execute_step(step)
            results.append(result)
        
        return {
            'plan_id': plan.id,
            'results': results,
            'success': all(r.success for r in results)
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (for nn.Module compatibility)"""
        return x  # Pass through for now