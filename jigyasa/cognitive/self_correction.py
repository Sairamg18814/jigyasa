"""
Self-Correction and Introspection Module
Implements Chain-of-Verification, Reverse COT, and other self-correction techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

from ..core.model import JigyasaModel


@dataclass
class CorrectionResult:
    """Result of a self-correction process"""
    original_response: str
    corrected_response: str
    corrections_made: List[str]
    confidence_score: float
    verification_steps: List[str]
    metadata: Dict[str, Any]


class SelfCorrectionStrategy(ABC):
    """Abstract base class for self-correction strategies"""
    
    @abstractmethod
    def correct(
        self, 
        query: str, 
        response: str, 
        model: JigyasaModel,
        **kwargs
    ) -> CorrectionResult:
        """Apply self-correction to a model response"""
        pass


class ChainOfVerification(SelfCorrectionStrategy):
    """
    Chain-of-Verification (CoVe) implementation
    Separates generation from verification by having the model check its own work
    """
    
    def __init__(self, max_verification_steps: int = 5):
        self.max_verification_steps = max_verification_steps
        
        # Templates for different types of verification
        self.verification_templates = {
            "factual": """Original Question: {query}
Original Answer: {response}

Please generate {num_questions} verification questions to check the factual accuracy of this answer:

Verification Questions:""",
            
            "logical": """Original Question: {query}
Original Answer: {response}

Please check the logical consistency of this answer by generating verification questions:

1. Are there any logical contradictions in the reasoning?
2. Do the conclusions follow from the premises?
3. Are there any missing steps in the logical chain?

Verification Questions:""",
            
            "mathematical": """Original Question: {query}
Original Answer: {response}

Please verify this mathematical solution step by step:

1. Check each calculation
2. Verify the formula or method used
3. Confirm the final answer

Verification:""",
            
            "general": """Original Question: {query}
Original Answer: {response}

Please generate questions to verify the accuracy and completeness of this answer:

Verification Questions:"""
        }
    
    def correct(
        self, 
        query: str, 
        response: str, 
        model: JigyasaModel,
        verification_type: str = "general",
        **kwargs
    ) -> CorrectionResult:
        """
        Apply Chain-of-Verification to correct a response
        """
        verification_steps = []
        corrections_made = []
        current_response = response
        
        # Step 1: Generate verification questions
        verification_questions = self._generate_verification_questions(
            query, current_response, model, verification_type
        )
        verification_steps.append(f"Generated {len(verification_questions)} verification questions")
        
        # Step 2: Answer verification questions
        verification_answers = []
        for i, vq in enumerate(verification_questions):
            answer = self._answer_verification_question(vq, current_response, model)
            verification_answers.append(answer)
            verification_steps.append(f"Verification Q{i+1}: {vq}")
            verification_steps.append(f"Verification A{i+1}: {answer}")
        
        # Step 3: Detect inconsistencies
        inconsistencies = self._detect_inconsistencies(
            query, current_response, verification_questions, verification_answers, model
        )
        
        if inconsistencies:
            verification_steps.append(f"Detected inconsistencies: {inconsistencies}")
            
            # Step 4: Generate corrected response
            corrected_response = self._generate_corrected_response(
                query, current_response, inconsistencies, model
            )
            corrections_made.extend(inconsistencies)
            current_response = corrected_response
            verification_steps.append("Generated corrected response")
        else:
            verification_steps.append("No inconsistencies detected")
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            verification_questions, verification_answers, inconsistencies
        )
        
        return CorrectionResult(
            original_response=response,
            corrected_response=current_response,
            corrections_made=corrections_made,
            confidence_score=confidence_score,
            verification_steps=verification_steps,
            metadata={
                'verification_type': verification_type,
                'num_verification_questions': len(verification_questions),
                'num_inconsistencies': len(inconsistencies)
            }
        )
    
    def _generate_verification_questions(
        self, 
        query: str, 
        response: str, 
        model: JigyasaModel,
        verification_type: str
    ) -> List[str]:
        """Generate verification questions for the response"""
        template = self.verification_templates.get(verification_type, self.verification_templates["general"])
        
        num_questions = 3 if verification_type != "mathematical" else 1
        prompt = template.format(
            query=query, 
            response=response, 
            num_questions=num_questions
        )
        
        # Generate verification questions
        generated = model.generate(
            input_text=prompt,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True
        )
        
        # Extract questions
        questions = self._extract_questions(generated)
        return questions[:self.max_verification_steps]
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract questions from generated text"""
        questions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions or questions ending with ?
            if (re.match(r'^\d+\.', line) or 
                line.endswith('?') or 
                line.startswith('Q:') or
                'question' in line.lower()):
                
                # Clean up the question
                question = re.sub(r'^\d+\.\s*', '', line)
                question = re.sub(r'^Q:\s*', '', question)
                if question and len(question) > 10:
                    questions.append(question)
        
        return questions
    
    def _answer_verification_question(
        self, 
        question: str, 
        original_response: str, 
        model: JigyasaModel
    ) -> str:
        """Answer a verification question"""
        prompt = f"""Original Response: {original_response}

Verification Question: {question}

Answer this verification question based on the original response:

Answer:"""
        
        answer = model.generate(
            input_text=prompt,
            max_new_tokens=100,
            temperature=0.2,
            do_sample=True
        )
        
        # Extract answer
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    
    def _detect_inconsistencies(
        self,
        query: str,
        response: str,
        verification_questions: List[str],
        verification_answers: List[str],
        model: JigyasaModel
    ) -> List[str]:
        """Detect inconsistencies between original response and verification answers"""
        
        # Combine verification Q&As
        verification_context = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in zip(verification_questions, verification_answers)
        ])
        
        prompt = f"""Original Question: {query}
Original Answer: {response}

Verification Questions and Answers:
{verification_context}

Based on the verification questions and answers, identify any inconsistencies, errors, or contradictions in the original answer. List each specific issue:

Inconsistencies:"""
        
        inconsistency_text = model.generate(
            input_text=prompt,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False
        )
        
        # Extract inconsistencies
        inconsistencies = self._extract_inconsistencies(inconsistency_text)
        return inconsistencies
    
    def _extract_inconsistencies(self, text: str) -> List[str]:
        """Extract inconsistencies from generated text"""
        inconsistencies = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if (line and 
                not line.lower().startswith('inconsistencies:') and
                not line.lower().startswith('none') and
                not line.lower().startswith('no inconsistencies')):
                
                # Clean up inconsistency description
                inconsistency = re.sub(r'^\d+\.\s*', '', line)
                inconsistency = re.sub(r'^-\s*', '', inconsistency)
                
                if inconsistency and len(inconsistency) > 5:
                    inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _generate_corrected_response(
        self,
        query: str,
        original_response: str,
        inconsistencies: List[str],
        model: JigyasaModel
    ) -> str:
        """Generate a corrected response addressing the inconsistencies"""
        
        inconsistency_list = "\n".join([f"- {inc}" for inc in inconsistencies])
        
        prompt = f"""Original Question: {query}
Original Answer: {original_response}

Issues identified in the original answer:
{inconsistency_list}

Please provide a corrected and improved answer that addresses these issues:

Corrected Answer:"""
        
        corrected = model.generate(
            input_text=prompt,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=True
        )
        
        # Extract corrected answer
        if "Corrected Answer:" in corrected:
            corrected = corrected.split("Corrected Answer:")[-1].strip()
        
        return corrected
    
    def _calculate_confidence(
        self,
        verification_questions: List[str],
        verification_answers: List[str],
        inconsistencies: List[str]
    ) -> float:
        """Calculate confidence score based on verification results"""
        base_confidence = 0.8
        
        # Reduce confidence for each inconsistency
        inconsistency_penalty = len(inconsistencies) * 0.2
        
        # Increase confidence for thorough verification
        verification_bonus = min(len(verification_questions) * 0.05, 0.2)
        
        confidence = base_confidence + verification_bonus - inconsistency_penalty
        return max(0.0, min(1.0, confidence))


class ReverseCOT(SelfCorrectionStrategy):
    """
    Reverse Chain-of-Thought implementation
    Has the model work backwards to verify its reasoning
    """
    
    def correct(
        self, 
        query: str, 
        response: str, 
        model: JigyasaModel,
        **kwargs
    ) -> CorrectionResult:
        """Apply Reverse Chain-of-Thought correction"""
        
        verification_steps = []
        corrections_made = []
        
        # Step 1: Extract the conclusion/answer
        conclusion = self._extract_conclusion(response)
        verification_steps.append(f"Extracted conclusion: {conclusion}")
        
        # Step 2: Generate a problem that would lead to this conclusion
        reverse_problem = self._generate_reverse_problem(conclusion, model)
        verification_steps.append(f"Generated reverse problem: {reverse_problem}")
        
        # Step 3: Compare reverse problem with original
        comparison = self._compare_problems(query, reverse_problem, model)
        verification_steps.append(f"Problem comparison: {comparison}")
        
        # Step 4: Identify discrepancies
        discrepancies = self._identify_discrepancies(comparison, model)
        
        corrected_response = response
        if discrepancies:
            corrections_made.extend(discrepancies)
            # Generate corrected response
            corrected_response = self._generate_corrected_response_rcot(
                query, response, discrepancies, model
            )
            verification_steps.append("Generated corrected response based on reverse COT")
        
        # Calculate confidence
        confidence_score = 0.9 if not discrepancies else 0.5
        
        return CorrectionResult(
            original_response=response,
            corrected_response=corrected_response,
            corrections_made=corrections_made,
            confidence_score=confidence_score,
            verification_steps=verification_steps,
            metadata={
                'reverse_problem': reverse_problem,
                'discrepancies': discrepancies
            }
        )
    
    def _extract_conclusion(self, response: str) -> str:
        """Extract the main conclusion from the response"""
        # Simple heuristic: take the last sentence or look for conclusion indicators
        sentences = response.split('.')
        
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                return sentence
        
        return response[-100:] if len(response) > 100 else response
    
    def _generate_reverse_problem(self, conclusion: str, model: JigyasaModel) -> str:
        """Generate a problem that would lead to the given conclusion"""
        prompt = f"""Given the following conclusion or answer, generate a problem or question that would logically lead to this conclusion:

Conclusion: {conclusion}

What problem or question would result in this conclusion?

Problem:"""
        
        reverse_problem = model.generate(
            input_text=prompt,
            max_new_tokens=150,
            temperature=0.4,
            do_sample=True
        )
        
        if "Problem:" in reverse_problem:
            reverse_problem = reverse_problem.split("Problem:")[-1].strip()
        
        return reverse_problem
    
    def _compare_problems(self, original: str, reverse: str, model: JigyasaModel) -> str:
        """Compare the original and reverse-generated problems"""
        prompt = f"""Compare these two problems and identify their similarities and differences:

Original Problem: {original}
Reverse-Generated Problem: {reverse}

Are they asking about the same thing? What are the key similarities and differences?

Comparison:"""
        
        comparison = model.generate(
            input_text=prompt,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True
        )
        
        if "Comparison:" in comparison:
            comparison = comparison.split("Comparison:")[-1].strip()
        
        return comparison
    
    def _identify_discrepancies(self, comparison: str, model: JigyasaModel) -> List[str]:
        """Identify discrepancies from the comparison"""
        # Simple heuristic: look for negative indicators
        discrepancy_indicators = [
            "different", "not the same", "inconsistent", "contradiction",
            "mismatch", "error", "wrong", "incorrect"
        ]
        
        discrepancies = []
        if any(indicator in comparison.lower() for indicator in discrepancy_indicators):
            discrepancies.append("Reverse-generated problem differs significantly from original")
        
        return discrepancies
    
    def _generate_corrected_response_rcot(
        self,
        query: str,
        original_response: str,
        discrepancies: List[str],
        model: JigyasaModel
    ) -> str:
        """Generate corrected response based on reverse COT findings"""
        discrepancy_text = "; ".join(discrepancies)
        
        prompt = f"""Original Question: {query}
Original Answer: {original_response}

Issues found through reverse reasoning: {discrepancy_text}

Please provide a corrected answer that addresses these reasoning issues:

Corrected Answer:"""
        
        corrected = model.generate(
            input_text=prompt,
            max_new_tokens=250,
            temperature=0.2,
            do_sample=True
        )
        
        if "Corrected Answer:" in corrected:
            corrected = corrected.split("Corrected Answer:")[-1].strip()
        
        return corrected


class SelfRefine(SelfCorrectionStrategy):
    """
    Self-Refine implementation
    Iterative self-improvement through critique and revision
    """
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
    
    def correct(
        self, 
        query: str, 
        response: str, 
        model: JigyasaModel,
        **kwargs
    ) -> CorrectionResult:
        """Apply Self-Refine correction"""
        
        verification_steps = []
        corrections_made = []
        current_response = response
        
        for iteration in range(self.max_iterations):
            verification_steps.append(f"Iteration {iteration + 1}")
            
            # Generate critique
            critique = self._generate_critique(query, current_response, model)
            verification_steps.append(f"Critique: {critique}")
            
            # Check if refinement is needed
            if self._needs_refinement(critique):
                # Generate refined response
                refined_response = self._generate_refinement(
                    query, current_response, critique, model
                )
                
                corrections_made.append(f"Iteration {iteration + 1}: {critique}")
                current_response = refined_response
                verification_steps.append("Generated refined response")
            else:
                verification_steps.append("No refinement needed")
                break
        
        confidence_score = 0.8 - (len(corrections_made) * 0.1)
        confidence_score = max(0.3, confidence_score)
        
        return CorrectionResult(
            original_response=response,
            corrected_response=current_response,
            corrections_made=corrections_made,
            confidence_score=confidence_score,
            verification_steps=verification_steps,
            metadata={'iterations': iteration + 1}
        )
    
    def _generate_critique(self, query: str, response: str, model: JigyasaModel) -> str:
        """Generate a critique of the current response"""
        prompt = f"""Question: {query}
Answer: {response}

Please critique this answer. Consider:
1. Accuracy and correctness
2. Completeness
3. Clarity and organization
4. Any potential improvements

Critique:"""
        
        critique = model.generate(
            input_text=prompt,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True
        )
        
        if "Critique:" in critique:
            critique = critique.split("Critique:")[-1].strip()
        
        return critique
    
    def _needs_refinement(self, critique: str) -> bool:
        """Determine if refinement is needed based on critique"""
        refinement_indicators = [
            "improve", "incorrect", "missing", "unclear", "add", "correct",
            "better", "should", "could", "needs", "lacks", "error"
        ]
        
        return any(indicator in critique.lower() for indicator in refinement_indicators)
    
    def _generate_refinement(
        self, 
        query: str, 
        response: str, 
        critique: str, 
        model: JigyasaModel
    ) -> str:
        """Generate a refined response based on the critique"""
        prompt = f"""Question: {query}
Original Answer: {response}
Critique: {critique}

Based on the critique, please provide an improved and refined answer:

Refined Answer:"""
        
        refined = model.generate(
            input_text=prompt,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=True
        )
        
        if "Refined Answer:" in refined:
            refined = refined.split("Refined Answer:")[-1].strip()
        
        return refined


class SelfCorrectionModule:
    """
    Main self-correction module that coordinates different correction strategies
    """
    
    def __init__(self, model: JigyasaModel):
        self.model = model
        
        # Initialize correction strategies
        self.strategies = {
            'cove': ChainOfVerification(),
            'rcot': ReverseCOT(),
            'refine': SelfRefine()
        }
        
        # Strategy selection rules
        self.strategy_rules = {
            'factual': 'cove',
            'mathematical': 'cove',
            'logical': 'rcot',
            'creative': 'refine',
            'analytical': 'cove',
            'default': 'cove'
        }
    
    def think_before_answer(
        self,
        query: str,
        context: Optional[str] = None,
        strategy: Optional[str] = None,
        query_type: str = 'default'
    ) -> Dict[str, Any]:
        """
        Main method for "thinking before answering"
        Generates initial response and applies self-correction
        """
        # Step 1: Generate initial response
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"
        
        initial_response = self.model.generate(
            input_text=prompt,
            max_new_tokens=250,
            temperature=0.7,
            do_sample=True
        )
        
        if "Answer:" in initial_response:
            initial_response = initial_response.split("Answer:")[-1].strip()
        
        # Step 2: Select correction strategy
        if strategy is None:
            strategy = self.strategy_rules.get(query_type, 'cove')
        
        # Step 3: Apply self-correction
        correction_result = self.apply_correction(
            query, initial_response, strategy
        )
        
        # Step 4: Generate thinking process summary
        thinking_process = self._generate_thinking_summary(
            query, initial_response, correction_result
        )
        
        return {
            'query': query,
            'initial_response': initial_response,
            'final_response': correction_result.corrected_response,
            'thinking_process': thinking_process,
            'corrections_made': correction_result.corrections_made,
            'confidence_score': correction_result.confidence_score,
            'verification_steps': correction_result.verification_steps,
            'strategy_used': strategy,
            'metadata': correction_result.metadata
        }
    
    def apply_correction(
        self,
        query: str,
        response: str,
        strategy: str = 'cove',
        **kwargs
    ) -> CorrectionResult:
        """Apply a specific correction strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
        
        correction_strategy = self.strategies[strategy]
        return correction_strategy.correct(query, response, self.model, **kwargs)
    
    def _generate_thinking_summary(
        self,
        query: str,
        initial_response: str,
        correction_result: CorrectionResult
    ) -> str:
        """Generate a summary of the thinking process"""
        
        thinking_parts = [
            "Initial Reasoning:",
            f"Generated response: {initial_response[:100]}..." if len(initial_response) > 100 else f"Generated response: {initial_response}",
            "",
            "Verification Process:"
        ]
        
        # Add verification steps
        for step in correction_result.verification_steps:
            thinking_parts.append(f"- {step}")
        
        # Add corrections if any
        if correction_result.corrections_made:
            thinking_parts.extend([
                "",
                "Corrections Made:"
            ])
            for correction in correction_result.corrections_made:
                thinking_parts.append(f"- {correction}")
        
        # Add final assessment
        thinking_parts.extend([
            "",
            f"Final Confidence: {correction_result.confidence_score:.2f}",
            f"Response {'refined' if correction_result.corrections_made else 'validated'}"
        ])
        
        return "\n".join(thinking_parts)
    
    def batch_correct(
        self,
        queries_and_responses: List[Tuple[str, str]],
        strategy: str = 'cove'
    ) -> List[CorrectionResult]:
        """Apply correction to multiple query-response pairs"""
        results = []
        
        for query, response in queries_and_responses:
            result = self.apply_correction(query, response, strategy)
            results.append(result)
        
        return results
    
    def evaluate_correction_quality(
        self,
        correction_result: CorrectionResult,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate the quality of a correction"""
        metrics = {}
        
        # Improvement score based on length and content changes
        original_len = len(correction_result.original_response)
        corrected_len = len(correction_result.corrected_response)
        
        metrics['length_change_ratio'] = corrected_len / original_len if original_len > 0 else 1.0
        metrics['num_corrections'] = len(correction_result.corrections_made)
        metrics['confidence_score'] = correction_result.confidence_score
        
        # Content similarity (simple word overlap)
        original_words = set(correction_result.original_response.lower().split())
        corrected_words = set(correction_result.corrected_response.lower().split())
        
        if original_words:
            metrics['content_similarity'] = len(original_words.intersection(corrected_words)) / len(original_words)
        else:
            metrics['content_similarity'] = 0.0
        
        # Ground truth comparison if available
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            if gt_words:
                original_accuracy = len(original_words.intersection(gt_words)) / len(gt_words)
                corrected_accuracy = len(corrected_words.intersection(gt_words)) / len(gt_words)
                
                metrics['original_accuracy'] = original_accuracy
                metrics['corrected_accuracy'] = corrected_accuracy
                metrics['accuracy_improvement'] = corrected_accuracy - original_accuracy
        
        return metrics
    
    def adaptive_strategy_selection(self, query: str, context: Optional[str] = None) -> str:
        """Automatically select the best correction strategy for a query"""
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Mathematical indicators
        if any(indicator in query_lower for indicator in ['calculate', 'solve', 'equation', 'formula', 'math']):
            return 'cove'
        
        # Logical reasoning indicators
        if any(indicator in query_lower for indicator in ['if', 'then', 'because', 'therefore', 'logic']):
            return 'rcot'
        
        # Creative/subjective indicators
        if any(indicator in query_lower for indicator in ['opinion', 'creative', 'imagine', 'design', 'write']):
            return 'refine'
        
        # Default to Chain-of-Verification for factual questions
        return 'cove'