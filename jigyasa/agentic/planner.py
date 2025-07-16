"""
Task Planning for Agents
Decomposes complex tasks into executable steps
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
try:
    from typing import Set
except ImportError:
    Set = set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from datetime import datetime


class StepType(Enum):
    """Types of steps in a plan"""
    TOOL_USE = "tool_use"
    REASONING = "reasoning"
    DECISION = "decision"
    LOOP = "loop"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class Step:
    """A single step in a plan"""
    id: str
    type: StepType
    description: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    optional: bool = False
    estimated_duration: float = 1.0
    retry_on_failure: bool = True
    max_retries: int = 3


@dataclass
class Plan:
    """A complete plan for a task"""
    id: str
    goal: str
    steps: List[Step]
    created_at: datetime
    estimated_total_duration: float
    complexity_score: float
    metadata: Optional[Dict[str, Any]] = None


class TaskPlanner:
    """
    Plans complex tasks by decomposing them into steps
    """
    
    def __init__(self, model, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        # Get model dimension
        if hasattr(model, 'config'):
            self.model_dim = getattr(model.config, 'd_model', 768)
        elif hasattr(model, 'in_features'):
            self.model_dim = model.in_features
        else:
            self.model_dim = 768
        
        # Planning components
        self.task_analyzer = self._build_task_analyzer()
        self.step_generator = self._build_step_generator()
        self.dependency_analyzer = self._build_dependency_analyzer()
        self.complexity_scorer = self._build_complexity_scorer()
        
        # Planning templates
        self.planning_templates = self._initialize_templates()
        
        # Planning history
        self.planning_history = []
    
    def _build_task_analyzer(self) -> nn.Module:
        """Build task analysis network"""
        return nn.Sequential(
            nn.Linear(self.model_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(StepType))
        )
    
    def _build_step_generator(self) -> nn.Module:
        """Build step generation network"""
        return nn.Sequential(
            nn.Linear(self.model_dim * 2, 768),  # task + context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, self.model_dim)
        )
    
    def _build_dependency_analyzer(self) -> nn.Module:
        """Build dependency analysis network"""
        return nn.Sequential(
            nn.Linear(self.model_dim * 2, 256),  # step1 + step2
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _build_complexity_scorer(self) -> nn.Module:
        """Build complexity scoring network"""
        return nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _initialize_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize planning templates for common tasks"""
        return {
            "web_research": [
                {
                    "type": StepType.TOOL_USE,
                    "action": "web_search",
                    "description": "Search for information"
                },
                {
                    "type": StepType.REASONING,
                    "action": "analyze_results",
                    "description": "Analyze search results"
                },
                {
                    "type": StepType.TOOL_USE,
                    "action": "rag_retrieve",
                    "description": "Retrieve relevant context"
                },
                {
                    "type": StepType.REASONING,
                    "action": "synthesize",
                    "description": "Synthesize information"
                }
            ],
            "code_generation": [
                {
                    "type": StepType.REASONING,
                    "action": "understand_requirements",
                    "description": "Analyze requirements"
                },
                {
                    "type": StepType.REASONING,
                    "action": "design_solution",
                    "description": "Design solution architecture"
                },
                {
                    "type": StepType.TOOL_USE,
                    "action": "code_execute",
                    "description": "Generate and test code"
                },
                {
                    "type": StepType.REASONING,
                    "action": "refine_code",
                    "description": "Refine and optimize"
                }
            ],
            "comparison": [
                {
                    "type": StepType.PARALLEL,
                    "action": "gather_data",
                    "description": "Gather data for comparison items"
                },
                {
                    "type": StepType.REASONING,
                    "action": "analyze_features",
                    "description": "Analyze features"
                },
                {
                    "type": StepType.REASONING,
                    "action": "create_comparison",
                    "description": "Create comparison matrix"
                },
                {
                    "type": StepType.DECISION,
                    "action": "make_recommendation",
                    "description": "Make recommendation"
                }
            ]
        }
    
    async def create_plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Plan:
        """Create a plan for a task"""
        # Encode task
        task_embedding = self._encode_task(task)
        
        # Analyze task type
        task_type = self._analyze_task_type(task_embedding)
        
        # Generate steps
        if task_type in self.planning_templates:
            # Use template as starting point
            template_steps = self.planning_templates[task_type]
            steps = self._adapt_template(template_steps, task, context)
        else:
            # Generate steps from scratch
            steps = self._generate_steps(task_embedding, task, context)
        
        # Analyze dependencies
        steps = self._analyze_dependencies(steps)
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(steps, task_embedding)
        
        # Create plan
        plan = Plan(
            id=f"plan_{len(self.planning_history)}",
            goal=task,
            steps=steps,
            created_at=datetime.now(),
            estimated_total_duration=sum(s.estimated_duration for s in steps),
            complexity_score=complexity_score,
            metadata=context
        )
        
        # Store in history
        self.planning_history.append(plan)
        
        return plan
    
    def _encode_task(self, task: str) -> torch.Tensor:
        """Encode task into embedding (simplified)"""
        # In practice, would use actual tokenizer and model
        return torch.randn(1, self.model_dim)
    
    def _analyze_task_type(self, task_embedding: torch.Tensor) -> str:
        """Analyze what type of task this is"""
        # Use task analyzer network
        task_logits = self.task_analyzer(task_embedding)
        
        # Map to template types (simplified)
        task_types = list(self.planning_templates.keys())
        if task_logits.shape[-1] >= len(task_types):
            task_idx = torch.argmax(task_logits[0, :len(task_types)]).item()
            return task_types[task_idx]
        
        return "general"
    
    def _adapt_template(
        self,
        template: List[Dict[str, Any]],
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Step]:
        """Adapt a template to specific task"""
        steps = []
        
        for i, template_step in enumerate(template):
            step = Step(
                id=f"step_{i}",
                type=template_step["type"],
                description=f"{template_step['description']} for: {task}",
                action=template_step["action"],
                parameters=self._generate_step_parameters(
                    template_step["action"],
                    task,
                    context
                ),
                optional=template_step.get("optional", False),
                estimated_duration=template_step.get("duration", 1.0)
            )
            steps.append(step)
        
        return steps
    
    def _generate_steps(
        self,
        task_embedding: torch.Tensor,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Step]:
        """Generate steps from scratch"""
        steps = []
        
        # Simple heuristic generation (would be neural in practice)
        # Analyze task keywords
        task_lower = task.lower()
        
        # Always start with understanding
        steps.append(Step(
            id="step_0",
            type=StepType.REASONING,
            description="Understand and analyze the task",
            action="analyze_task",
            parameters={"task": task},
            estimated_duration=0.5
        ))
        
        # Add steps based on keywords
        if any(word in task_lower for word in ["search", "find", "research"]):
            steps.append(Step(
                id="step_1",
                type=StepType.TOOL_USE,
                description="Search for information",
                action="web_search",
                parameters={"query": task},
                estimated_duration=2.0
            ))
        
        if any(word in task_lower for word in ["code", "program", "implement"]):
            steps.append(Step(
                id=f"step_{len(steps)}",
                type=StepType.TOOL_USE,
                description="Generate code",
                action="code_execute",
                parameters={"task": task},
                estimated_duration=3.0
            ))
        
        if any(word in task_lower for word in ["compare", "versus", "vs"]):
            steps.append(Step(
                id=f"step_{len(steps)}",
                type=StepType.PARALLEL,
                description="Gather comparison data",
                action="parallel_gather",
                parameters={"items": []},  # Would extract from task
                estimated_duration=2.0
            ))
        
        # Always end with synthesis
        steps.append(Step(
            id=f"step_{len(steps)}",
            type=StepType.REASONING,
            description="Synthesize results and generate response",
            action="synthesize",
            parameters={"task": task},
            estimated_duration=1.0
        ))
        
        return steps
    
    def _generate_step_parameters(
        self,
        action: str,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate parameters for a step"""
        # Simple parameter generation
        params = {}
        
        if action == "web_search":
            params["query"] = task
            params["max_results"] = 5
        elif action == "code_execute":
            params["task_description"] = task
            params["language"] = context.get("language", "python") if context else "python"
        elif action == "rag_retrieve":
            params["query"] = task
            params["k"] = 5
        
        return params
    
    def _analyze_dependencies(self, steps: List[Step]) -> List[Step]:
        """Analyze dependencies between steps"""
        # Build dependency graph
        graph = nx.DiGraph()
        
        for step in steps:
            graph.add_node(step.id)
        
        # Analyze pairwise dependencies
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps):
                if i >= j:
                    continue
                
                # Check if step2 depends on step1
                if self._check_dependency(step1, step2):
                    graph.add_edge(step1.id, step2.id)
                    if step2.dependencies is None:
                        step2.dependencies = []
                    step2.dependencies.append(step1.id)
        
        # Detect cycles
        if not nx.is_directed_acyclic_graph(graph):
            # Remove edges to break cycles
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    graph.remove_edge(cycle[-1], cycle[0])
        
        return steps
    
    def _check_dependency(self, step1: Step, step2: Step) -> bool:
        """Check if step2 depends on step1"""
        # Simple heuristics
        if step1.type == StepType.TOOL_USE and step2.type == StepType.REASONING:
            # Reasoning often depends on tool results
            return True
        
        if step1.action == "web_search" and step2.action in ["analyze_results", "synthesize"]:
            return True
        
        if step1.action == "analyze_task" and step2.type != StepType.REASONING:
            # Most steps depend on initial analysis
            return True
        
        return False
    
    def _calculate_complexity(
        self,
        steps: List[Step],
        task_embedding: torch.Tensor
    ) -> float:
        """Calculate plan complexity"""
        # Use complexity scorer
        base_complexity = self.complexity_scorer(task_embedding).item()
        
        # Adjust based on plan characteristics
        step_factor = len(steps) / 10.0  # More steps = more complex
        parallel_factor = sum(1 for s in steps if s.type == StepType.PARALLEL) * 0.2
        conditional_factor = sum(1 for s in steps if s.type == StepType.CONDITIONAL) * 0.3
        
        total_complexity = base_complexity + step_factor + parallel_factor + conditional_factor
        
        return min(total_complexity, 1.0)
    
    def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize a plan for efficiency"""
        # Identify parallelizable steps
        parallel_groups = self._identify_parallel_groups(plan.steps)
        
        # Merge parallel steps
        optimized_steps = []
        processed = set()
        
        for step in plan.steps:
            if step.id in processed:
                continue
            
            # Check if part of parallel group
            parallel_group = None
            for group in parallel_groups:
                if step.id in group:
                    parallel_group = group
                    break
            
            if parallel_group and len(parallel_group) > 1:
                # Create parallel step
                parallel_step = Step(
                    id=f"parallel_{step.id}",
                    type=StepType.PARALLEL,
                    description=f"Execute {len(parallel_group)} steps in parallel",
                    action="parallel_execute",
                    parameters={
                        "steps": [s for s in plan.steps if s.id in parallel_group]
                    },
                    estimated_duration=max(
                        s.estimated_duration for s in plan.steps if s.id in parallel_group
                    )
                )
                optimized_steps.append(parallel_step)
                processed.update(parallel_group)
            else:
                optimized_steps.append(step)
                processed.add(step.id)
        
        # Create optimized plan
        optimized_plan = Plan(
            id=f"{plan.id}_optimized",
            goal=plan.goal,
            steps=optimized_steps,
            created_at=datetime.now(),
            estimated_total_duration=sum(s.estimated_duration for s in optimized_steps),
            complexity_score=plan.complexity_score,
            metadata=plan.metadata
        )
        
        return optimized_plan
    
    def _identify_parallel_groups(self, steps: List[Step]) -> List[Set[str]]:
        """Identify groups of steps that can run in parallel"""
        # Build dependency graph
        graph = nx.DiGraph()
        
        for step in steps:
            graph.add_node(step.id)
            if step.dependencies:
                for dep in step.dependencies:
                    graph.add_edge(dep, step.id)
        
        # Find steps with no dependencies on each other
        parallel_groups = []
        
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps):
                if i >= j:
                    continue
                
                # Check if they can run in parallel
                if (not nx.has_path(graph, step1.id, step2.id) and 
                    not nx.has_path(graph, step2.id, step1.id)):
                    
                    # Find or create group
                    found = False
                    for group in parallel_groups:
                        if step1.id in group or step2.id in group:
                            group.add(step1.id)
                            group.add(step2.id)
                            found = True
                            break
                    
                    if not found:
                        parallel_groups.append({step1.id, step2.id})
        
        return parallel_groups