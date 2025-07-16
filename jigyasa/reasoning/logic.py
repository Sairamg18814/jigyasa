"""
Logic Engine for Formal Reasoning
Implements first-order logic and constraint solving
"""

import torch
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
import re
from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent
from sympy import symbols, Symbol
from sympy.logic.inference import satisfiable


class LogicOperator(Enum):
    """Logical operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    EQUIVALENT = "equivalent"
    FORALL = "forall"
    EXISTS = "exists"


@dataclass
class LogicalStatement:
    """Represents a logical statement"""
    predicate: str
    arguments: List[str]
    negated: bool = False
    quantifier: Optional[str] = None


@dataclass
class LogicalFormula:
    """Represents a logical formula"""
    statements: List[LogicalStatement]
    operators: List[LogicOperator]
    variables: Set[str]


class FirstOrderLogic:
    """
    First-order logic reasoning engine
    """
    
    def __init__(self):
        self.knowledge_base = []
        self.rules = []
        self.facts = {}
    
    def parse_statement(self, statement: str) -> LogicalStatement:
        """Parse a natural language statement into logical form"""
        # Remove extra spaces and lowercase
        statement = ' '.join(statement.split()).lower()
        
        # Check for negation
        negated = False
        if statement.startswith("not ") or statement.startswith("it is not true that "):
            negated = True
            statement = statement.replace("not ", "").replace("it is not true that ", "")
        
        # Check for quantifiers
        quantifier = None
        if statement.startswith("for all ") or statement.startswith("forall "):
            quantifier = "forall"
            statement = statement.replace("for all ", "").replace("forall ", "")
        elif statement.startswith("there exists ") or statement.startswith("exists "):
            quantifier = "exists"
            statement = statement.replace("there exists ", "").replace("exists ", "")
        
        # Extract predicate and arguments
        # Simple pattern matching - would be more sophisticated in practice
        match = re.match(r'(\w+)\(([\w,\s]+)\)', statement)
        if match:
            predicate = match.group(1)
            args = [arg.strip() for arg in match.group(2).split(',')]
        else:
            # Try to extract from natural language
            parts = statement.split()
            if len(parts) >= 2:
                predicate = parts[0]
                args = parts[1:]
            else:
                predicate = statement
                args = []
        
        return LogicalStatement(
            predicate=predicate,
            arguments=args,
            negated=negated,
            quantifier=quantifier
        )
    
    def add_fact(self, statement: Union[str, LogicalStatement]):
        """Add a fact to the knowledge base"""
        if isinstance(statement, str):
            statement = self.parse_statement(statement)
        
        self.knowledge_base.append(statement)
        
        # Index facts for faster lookup
        key = (statement.predicate, tuple(statement.arguments))
        self.facts[key] = not statement.negated
    
    def add_rule(self, premise: List[LogicalStatement], conclusion: LogicalStatement):
        """Add an inference rule"""
        self.rules.append({
            'premise': premise,
            'conclusion': conclusion
        })
    
    def query(self, query_statement: Union[str, LogicalStatement]) -> Tuple[bool, List[str]]:
        """
        Query the knowledge base
        Returns (result, explanation)
        """
        if isinstance(query_statement, str):
            query_statement = self.parse_statement(query_statement)
        
        explanation = []
        
        # Direct fact lookup
        key = (query_statement.predicate, tuple(query_statement.arguments))
        if key in self.facts:
            result = self.facts[key]
            if query_statement.negated:
                result = not result
            explanation.append(f"Direct fact: {query_statement.predicate}({', '.join(query_statement.arguments)}) = {result}")
            return result, explanation
        
        # Try forward chaining
        result, chain_explanation = self._forward_chain(query_statement)
        if result is not None:
            explanation.extend(chain_explanation)
            return result, explanation
        
        # Try backward chaining
        result, chain_explanation = self._backward_chain(query_statement)
        if result is not None:
            explanation.extend(chain_explanation)
            return result, explanation
        
        explanation.append("Could not determine truth value from knowledge base")
        return False, explanation
    
    def _forward_chain(self, goal: LogicalStatement) -> Tuple[Optional[bool], List[str]]:
        """Forward chaining inference"""
        explanation = ["Attempting forward chaining..."]
        
        # Keep applying rules until no new facts are derived
        changed = True
        iterations = 0
        max_iterations = 100
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for rule in self.rules:
                # Check if all premises are satisfied
                all_satisfied = True
                for premise in rule['premise']:
                    key = (premise.predicate, tuple(premise.arguments))
                    if key not in self.facts or self.facts[key] != (not premise.negated):
                        all_satisfied = False
                        break
                
                if all_satisfied:
                    # Apply rule
                    conclusion = rule['conclusion']
                    key = (conclusion.predicate, tuple(conclusion.arguments))
                    
                    if key not in self.facts:
                        self.facts[key] = not conclusion.negated
                        changed = True
                        explanation.append(
                            f"Applied rule: {[str(p) for p in rule['premise']]} → "
                            f"{conclusion.predicate}({', '.join(conclusion.arguments)})"
                        )
                        
                        # Check if we derived our goal
                        if (conclusion.predicate == goal.predicate and 
                            conclusion.arguments == goal.arguments):
                            return not conclusion.negated if not goal.negated else conclusion.negated, explanation
        
        return None, explanation
    
    def _backward_chain(self, goal: LogicalStatement) -> Tuple[Optional[bool], List[str]]:
        """Backward chaining inference"""
        explanation = ["Attempting backward chaining..."]
        
        # Find rules that could derive the goal
        for rule in self.rules:
            conclusion = rule['conclusion']
            if (conclusion.predicate == goal.predicate and 
                conclusion.arguments == goal.arguments):
                
                # Check if all premises can be satisfied
                all_satisfied = True
                premise_explanations = []
                
                for premise in rule['premise']:
                    result, premise_exp = self.query(premise)
                    premise_explanations.extend(premise_exp)
                    
                    if not result:
                        all_satisfied = False
                        break
                
                if all_satisfied:
                    explanation.append(
                        f"Goal {goal.predicate}({', '.join(goal.arguments)}) "
                        f"derived from rule with premises: {[str(p) for p in rule['premise']]}"
                    )
                    explanation.extend(premise_explanations)
                    return not conclusion.negated if not goal.negated else conclusion.negated, explanation
        
        return None, explanation


class LogicEngine:
    """
    Main logic engine that integrates with neural model
    """
    
    def __init__(self, model_dim: int = 768):
        self.model_dim = model_dim
        self.fol_engine = FirstOrderLogic()
        
        # Neural components for logic
        self.statement_encoder = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
        
        self.logic_classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(LogicOperator))
        )
    
    def parse_logical_text(self, text: str) -> List[LogicalStatement]:
        """Parse text into logical statements"""
        statements = []
        
        # Split by common logical connectives
        parts = re.split(r'\s+(and|or|implies|if and only if)\s+', text.lower())
        
        for part in parts:
            if part not in ['and', 'or', 'implies', 'if and only if']:
                try:
                    stmt = self.fol_engine.parse_statement(part.strip())
                    statements.append(stmt)
                except:
                    continue
        
        return statements
    
    def reason(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform logical reasoning on query with optional context
        """
        # Parse context into knowledge base
        if context:
            context_statements = self.parse_logical_text(context)
            for stmt in context_statements:
                self.fol_engine.add_fact(stmt)
        
        # Parse and process query
        query_statement = self.fol_engine.parse_statement(query)
        
        # Perform reasoning
        result, explanation = self.fol_engine.query(query_statement)
        
        # Check satisfiability
        satisfiable_result = self._check_satisfiability(query)
        
        return {
            'query': query,
            'parsed_query': str(query_statement),
            'result': result,
            'explanation': explanation,
            'satisfiable': satisfiable_result,
            'knowledge_base_size': len(self.fol_engine.knowledge_base)
        }
    
    def _check_satisfiability(self, formula_str: str) -> bool:
        """Check if a logical formula is satisfiable"""
        try:
            # Parse formula
            # Replace common words with logical operators
            formula_str = formula_str.lower()
            formula_str = formula_str.replace(' and ', ' & ')
            formula_str = formula_str.replace(' or ', ' | ')
            formula_str = formula_str.replace(' not ', ' ~')
            formula_str = formula_str.replace(' implies ', ' >> ')
            
            # Extract variables
            var_pattern = r'\b[a-z]\b'
            variables = list(set(re.findall(var_pattern, formula_str)))
            
            if variables:
                # Create sympy symbols
                syms = symbols(' '.join(variables))
                
                # Try to evaluate
                result = satisfiable(formula_str, all_models=True)
                return result is not False
            
            return True  # Empty formula is satisfiable
            
        except:
            return True  # Default to satisfiable if parsing fails
    
    def add_knowledge(self, statements: List[str]):
        """Add multiple statements to knowledge base"""
        for stmt in statements:
            try:
                self.fol_engine.add_fact(stmt)
            except:
                continue
    
    def add_inference_rule(self, rule: str):
        """
        Add an inference rule in the form "P1 and P2 and ... -> Q"
        """
        try:
            parts = rule.split('->')
            if len(parts) == 2:
                premise_str = parts[0].strip()
                conclusion_str = parts[1].strip()
                
                # Parse premises
                premise_parts = premise_str.split(' and ')
                premises = [self.fol_engine.parse_statement(p.strip()) for p in premise_parts]
                
                # Parse conclusion
                conclusion = self.fol_engine.parse_statement(conclusion_str)
                
                # Add rule
                self.fol_engine.add_rule(premises, conclusion)
        except:
            pass