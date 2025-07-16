"""
Main Agent Implementation
Provides proactive, intelligent agent capabilities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json
from enum import Enum

from .tools import ToolRegistry, Tool, ToolResult
from .memory import AgentMemory
from .planner import TaskPlanner, Plan
from .executor import ActionExecutor


class AgentMode(Enum):
    """Agent operation modes"""
    REACTIVE = "reactive"  # Responds to queries
    PROACTIVE = "proactive"  # Anticipates needs
    AUTONOMOUS = "autonomous"  # Self-directed
    COLLABORATIVE = "collaborative"  # Works with other agents


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str = "Jigyasa"
    mode: AgentMode = AgentMode.PROACTIVE
    max_steps: int = 10
    max_retries: int = 3
    enable_learning: bool = True
    enable_memory: bool = True
    anticipation_threshold: float = 0.7
    collaboration_enabled: bool = False


class Agent:
    """
    Individual agent with specific capabilities
    """
    
    def __init__(
        self,
        name: str,
        model,
        tools: List[Tool],
        config: Optional[AgentConfig] = None
    ):
        self.name = name
        self.model = model
        self.config = config or AgentConfig(name=name)
        
        # Initialize components
        self.tool_registry = ToolRegistry()
        for tool in tools:
            self.tool_registry.register(tool)
        
        self.memory = AgentMemory(capacity=1000) if self.config.enable_memory else None
        self.planner = TaskPlanner(model)
        self.executor = ActionExecutor(self.tool_registry)
        
        # Proactive monitoring
        self.monitoring_patterns = []
        self.anticipation_model = self._build_anticipation_model()
        
        # Execution history
        self.execution_history = []
    
    def _build_anticipation_model(self) -> nn.Module:
        """Build model for anticipating user needs"""
        if hasattr(self.model, 'config'):
            model_dim = getattr(self.model.config, 'd_model', 768)
        elif hasattr(self.model, 'in_features'):
            model_dim = self.model.in_features
        else:
            model_dim = 768
        
        return nn.Sequential(
            nn.Linear(model_dim * 2, 512),  # context + history
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user request
        """
        # Store in memory if enabled
        if self.memory:
            self.memory.add_interaction(request, context)
        
        # Plan the task
        plan = await self.planner.create_plan(request, context)
        
        # Execute the plan
        results = []
        for step in plan.steps:
            try:
                result = await self.executor.execute_step(step, context)
                results.append(result)
                
                # Update context with results
                if context is None:
                    context = {}
                context[f"step_{step.id}_result"] = result
                
            except Exception as e:
                # Handle errors gracefully
                error_result = ToolResult(
                    success=False,
                    data=None,
                    error=str(e),
                    metadata={"step_id": step.id}
                )
                results.append(error_result)
                
                if not step.optional:
                    break
        
        # Generate final response
        final_response = await self._synthesize_response(request, plan, results)
        
        # Record execution
        execution_record = {
            "request": request,
            "plan": plan,
            "results": results,
            "response": final_response,
            "timestamp": datetime.now()
        }
        self.execution_history.append(execution_record)
        
        # Check for proactive actions if enabled
        if self.config.mode == AgentMode.PROACTIVE:
            proactive_suggestions = await self._generate_proactive_suggestions(
                request, final_response, context
            )
            final_response["proactive_suggestions"] = proactive_suggestions
        
        return final_response
    
    async def _synthesize_response(
        self,
        request: str,
        plan: Plan,
        results: List[ToolResult]
    ) -> Dict[str, Any]:
        """Synthesize final response from execution results"""
        # Collect successful results
        successful_results = [r for r in results if r.success]
        
        # Create summary
        summary_prompt = f"""
        Original request: {request}
        
        Plan executed: {len(plan.steps)} steps
        Successful steps: {len(successful_results)}
        
        Results:
        {json.dumps([r.data for r in successful_results], indent=2)}
        
        Please provide a comprehensive response to the original request based on these results.
        """
        
        # Generate response (simplified - would use actual model)
        response_text = f"Based on my analysis: Completed {len(successful_results)} steps successfully."
        
        return {
            "response": response_text,
            "plan_summary": {
                "total_steps": len(plan.steps),
                "successful_steps": len(successful_results),
                "failed_steps": len(results) - len(successful_results)
            },
            "detailed_results": [
                {
                    "step": i,
                    "success": r.success,
                    "data": r.data if r.success else r.error
                }
                for i, r in enumerate(results)
            ]
        }
    
    async def _generate_proactive_suggestions(
        self,
        request: str,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate proactive suggestions based on interaction"""
        suggestions = []
        
        # Analyze patterns
        if self.memory:
            recent_interactions = self.memory.get_recent(n=5)
            
            # Look for patterns (simplified)
            if any("code" in i["request"].lower() for i in recent_interactions):
                suggestions.append({
                    "type": "follow_up",
                    "suggestion": "Would you like me to generate unit tests for this code?",
                    "confidence": 0.8
                })
            
            if any("error" in i["request"].lower() for i in recent_interactions):
                suggestions.append({
                    "type": "debugging",
                    "suggestion": "I can help debug similar issues. Would you like me to set up error monitoring?",
                    "confidence": 0.7
                })
        
        # Context-based suggestions
        if context and "project_type" in context:
            if context["project_type"] == "web":
                suggestions.append({
                    "type": "enhancement",
                    "suggestion": "Consider adding performance monitoring for your web application.",
                    "confidence": 0.6
                })
        
        return suggestions
    
    def add_monitoring_pattern(
        self,
        pattern: Dict[str, Any],
        callback: Callable
    ):
        """Add a pattern for proactive monitoring"""
        self.monitoring_patterns.append({
            "pattern": pattern,
            "callback": callback,
            "active": True
        })
    
    async def monitor(self, data: Any) -> List[Dict[str, Any]]:
        """Monitor data for patterns requiring action"""
        triggers = []
        
        for monitor in self.monitoring_patterns:
            if not monitor["active"]:
                continue
            
            # Check if pattern matches (simplified)
            if self._matches_pattern(data, monitor["pattern"]):
                # Calculate anticipation score
                anticipation_score = self._calculate_anticipation_score(data)
                
                if anticipation_score > self.config.anticipation_threshold:
                    triggers.append({
                        "pattern": monitor["pattern"],
                        "score": anticipation_score,
                        "suggested_action": monitor["callback"]
                    })
        
        return triggers
    
    def _matches_pattern(self, data: Any, pattern: Dict[str, Any]) -> bool:
        """Check if data matches pattern (simplified)"""
        # In practice, would use more sophisticated pattern matching
        if "keywords" in pattern:
            data_str = str(data).lower()
            return any(kw in data_str for kw in pattern["keywords"])
        return False
    
    def _calculate_anticipation_score(self, data: Any) -> float:
        """Calculate anticipation score for proactive action"""
        # Simplified - would use neural anticipation model
        return 0.8


class AgenticFramework:
    """
    Main agentic framework managing multiple agents
    """
    
    def __init__(self, base_model, config: Optional[Dict[str, Any]] = None):
        self.base_model = base_model
        self.config = config or {}
        
        # Agent registry
        self.agents: Dict[str, Agent] = {}
        
        # Initialize default tools
        self.default_tools = self._initialize_default_tools()
        
        # Create primary agent
        self.primary_agent = self._create_primary_agent()
        self.agents["primary"] = self.primary_agent
        
        # Collaboration coordinator
        self.collaboration_enabled = self.config.get("enable_collaboration", False)
        if self.collaboration_enabled:
            self._initialize_collaboration()
    
    def _initialize_default_tools(self) -> List[Tool]:
        """Initialize default tool set"""
        tools = []
        
        # Web search tool
        tools.append(Tool(
            name="web_search",
            description="Search the web for information",
            function=self._web_search_tool,
            parameters={
                "query": {"type": "string", "required": True},
                "max_results": {"type": "integer", "default": 5}
            }
        ))
        
        # Code execution tool
        tools.append(Tool(
            name="code_execute",
            description="Execute Python code",
            function=self._code_execute_tool,
            parameters={
                "code": {"type": "string", "required": True},
                "timeout": {"type": "integer", "default": 30}
            }
        ))
        
        # File operations tool
        tools.append(Tool(
            name="file_operations",
            description="Read, write, or modify files",
            function=self._file_operations_tool,
            parameters={
                "operation": {"type": "string", "required": True},
                "path": {"type": "string", "required": True},
                "content": {"type": "string", "required": False}
            }
        ))
        
        # RAG tool
        tools.append(Tool(
            name="rag_retrieve",
            description="Retrieve relevant information from knowledge base",
            function=self._rag_tool,
            parameters={
                "query": {"type": "string", "required": True},
                "k": {"type": "integer", "default": 5}
            }
        ))
        
        return tools
    
    def _create_primary_agent(self) -> Agent:
        """Create the primary agent"""
        config = AgentConfig(
            name="Jigyasa-Primary",
            mode=AgentMode.PROACTIVE,
            enable_learning=True,
            enable_memory=True
        )
        
        return Agent(
            name=config.name,
            model=self.base_model,
            tools=self.default_tools,
            config=config
        )
    
    def create_specialized_agent(
        self,
        name: str,
        specialization: str,
        tools: Optional[List[Tool]] = None
    ) -> Agent:
        """Create a specialized agent"""
        # Use default tools if none provided
        if tools is None:
            tools = self.default_tools
        
        config = AgentConfig(
            name=f"Jigyasa-{specialization}",
            mode=AgentMode.REACTIVE,
            enable_memory=True
        )
        
        agent = Agent(
            name=name,
            model=self.base_model,
            tools=tools,
            config=config
        )
        
        self.agents[name] = agent
        return agent
    
    async def process(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a request using specified or primary agent
        """
        # Select agent
        if agent_name and agent_name in self.agents:
            agent = self.agents[agent_name]
        else:
            agent = self.primary_agent
        
        # Process request
        result = await agent.process_request(request, context)
        
        # Check if collaboration needed
        if self.collaboration_enabled and self._needs_collaboration(request, result):
            collaborative_result = await self._collaborative_process(request, context, result)
            result["collaborative_enhancement"] = collaborative_result
        
        return result
    
    def _needs_collaboration(self, request: str, result: Dict[str, Any]) -> bool:
        """Determine if request needs multiple agents"""
        # Simplified heuristic
        complex_keywords = ["analyze", "compare", "comprehensive", "detailed", "multiple"]
        return any(kw in request.lower() for kw in complex_keywords)
    
    async def _collaborative_process(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
        initial_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request using multiple agents"""
        # This would orchestrate multiple specialized agents
        # For now, return placeholder
        return {
            "enhanced": True,
            "agents_used": ["primary"],
            "confidence": 0.9
        }
    
    def _initialize_collaboration(self):
        """Initialize multi-agent collaboration"""
        # Create specialized agents
        self.create_specialized_agent("researcher", "research")
        self.create_specialized_agent("coder", "coding")
        self.create_specialized_agent("analyst", "analysis")
    
    # Tool implementations (simplified)
    async def _web_search_tool(self, **kwargs) -> ToolResult:
        """Web search tool implementation"""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        
        # Simulated search results
        results = [
            {"title": f"Result {i+1} for {query}", "url": f"https://example.com/{i}"}
            for i in range(max_results)
        ]
        
        return ToolResult(
            success=True,
            data=results,
            metadata={"query": query, "count": len(results)}
        )
    
    async def _code_execute_tool(self, **kwargs) -> ToolResult:
        """Code execution tool implementation"""
        code = kwargs.get("code", "")
        
        # Safety check (simplified)
        if any(danger in code for danger in ["import os", "import sys", "__import__"]):
            return ToolResult(
                success=False,
                error="Code contains potentially unsafe operations"
            )
        
        # Simulated execution
        return ToolResult(
            success=True,
            data={"output": "Code executed successfully", "return_value": None}
        )
    
    async def _file_operations_tool(self, **kwargs) -> ToolResult:
        """File operations tool implementation"""
        operation = kwargs.get("operation", "")
        path = kwargs.get("path", "")
        
        # Safety check
        if ".." in path or path.startswith("/"):
            return ToolResult(
                success=False,
                error="Path traversal not allowed"
            )
        
        # Simulated file operation
        return ToolResult(
            success=True,
            data={"operation": operation, "path": path, "status": "completed"}
        )
    
    async def _rag_tool(self, **kwargs) -> ToolResult:
        """RAG retrieval tool implementation"""
        query = kwargs.get("query", "")
        k = kwargs.get("k", 5)
        
        # Simulated retrieval
        documents = [
            {"content": f"Document {i+1} relevant to: {query}", "score": 0.9 - i*0.1}
            for i in range(k)
        ]
        
        return ToolResult(
            success=True,
            data=documents,
            metadata={"query": query, "retrieved": len(documents)}
        )