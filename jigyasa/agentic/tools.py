"""
Tool Registry and Management for Agents
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import inspect
import asyncio
from datetime import datetime


@dataclass
class ToolParameter:
    """Parameter definition for a tool"""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Tool:
    """
    A tool that can be used by agents
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        requires_confirmation: bool = False,
        rate_limit: Optional[int] = None  # calls per minute
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = self._parse_parameters(parameters)
        self.requires_confirmation = requires_confirmation
        self.rate_limit = rate_limit
        
        # Usage tracking
        self.usage_count = 0
        self.last_used = None
        self.recent_calls = []
    
    def _parse_parameters(self, params: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, ToolParameter]:
        """Parse parameter definitions"""
        if params is None:
            # Try to infer from function signature
            sig = inspect.signature(self.function)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'cls']:
                    continue
                params[param_name] = {
                    'type': 'any',
                    'required': param.default == inspect.Parameter.empty
                }
        
        parsed = {}
        for name, config in params.items():
            parsed[name] = ToolParameter(
                name=name,
                type=config.get('type', 'any'),
                required=config.get('required', True),
                default=config.get('default'),
                description=config.get('description', '')
            )
        
        return parsed
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        start_time = datetime.now()
        
        # Check rate limit
        if self.rate_limit and not self._check_rate_limit():
            return ToolResult(
                success=False,
                error=f"Rate limit exceeded for {self.name}"
            )
        
        # Validate parameters
        validation_error = self._validate_parameters(kwargs)
        if validation_error:
            return ToolResult(
                success=False,
                error=validation_error
            )
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                result = self.function(**kwargs)
            
            # Track usage
            self.usage_count += 1
            self.last_used = datetime.now()
            self.recent_calls.append(start_time)
            
            # Clean old calls
            if len(self.recent_calls) > 100:
                self.recent_calls = self.recent_calls[-100:]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Return result
            if isinstance(result, ToolResult):
                result.execution_time = execution_time
                return result
            else:
                return ToolResult(
                    success=True,
                    data=result,
                    execution_time=execution_time
                )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_parameters(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Validate input parameters"""
        for param_name, param in self.parameters.items():
            if param.required and param_name not in kwargs:
                return f"Missing required parameter: {param_name}"
            
            if param_name in kwargs:
                # Type validation (simplified)
                value = kwargs[param_name]
                expected_type = param.type
                
                if expected_type == 'string' and not isinstance(value, str):
                    return f"Parameter {param_name} must be a string"
                elif expected_type == 'integer' and not isinstance(value, int):
                    return f"Parameter {param_name} must be an integer"
                elif expected_type == 'float' and not isinstance(value, (int, float)):
                    return f"Parameter {param_name} must be a number"
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    return f"Parameter {param_name} must be a boolean"
                elif expected_type == 'list' and not isinstance(value, list):
                    return f"Parameter {param_name} must be a list"
                elif expected_type == 'dict' and not isinstance(value, dict):
                    return f"Parameter {param_name} must be a dictionary"
        
        return None
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        if not self.rate_limit:
            return True
        
        # Count calls in last minute
        now = datetime.now()
        recent_count = sum(
            1 for call_time in self.recent_calls
            if (now - call_time).total_seconds() < 60
        )
        
        return recent_count < self.rate_limit
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the tool"""
        return {
            'name': self.name,
            'total_calls': self.usage_count,
            'last_used': self.last_used,
            'recent_calls_count': len(self.recent_calls),
            'rate_limit': self.rate_limit
        }


class ToolRegistry:
    """
    Registry for managing available tools
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register(self, tool: Tool, category: Optional[str] = None):
        """Register a new tool"""
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        
        self.tools[tool.name] = tool
        
        if category:
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(tool.name)
    
    def unregister(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            
            # Remove from categories
            for category, tools in self.categories.items():
                if tool_name in tools:
                    tools.remove(tool_name)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        """List all tools or tools in a category"""
        if category and category in self.categories:
            return [self.tools[name] for name in self.categories[category]]
        return list(self.tools.values())
    
    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append(tool)
        
        return results
    
    async def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        return await tool.execute(**kwargs)
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return None
        
        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': {
                param.name: {
                    'type': param.type,
                    'required': param.required,
                    'default': param.default,
                    'description': param.description
                }
                for param in tool.parameters.values()
            },
            'requires_confirmation': tool.requires_confirmation,
            'rate_limit': tool.rate_limit
        }
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools"""
        return [
            self.get_tool_schema(tool_name)
            for tool_name in self.tools
        ]
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage report for all tools"""
        return {
            tool_name: tool.get_usage_stats()
            for tool_name, tool in self.tools.items()
        }