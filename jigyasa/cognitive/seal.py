"""
SEAL (Self-Adapting Language Models) implementation
Enables continuous learning through self-generated training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
# import wandb  # Optional - commented out for demo
import numpy as np
from tqdm import tqdm

from ..config import SEALConfig
from ..core.model import JigyasaModel


@dataclass 
class SelfEditExample:
    """Container for self-edit training examples"""
    context: str
    self_edit: str
    target_task: str
    reward: float
    metadata: Dict[str, Any]


class SelfEditGenerator(nn.Module):
    """
    Generates self-edits for new information
    Core component of SEAL's inner loop
    """
    
    def __init__(self, model: JigyasaModel, config: SEALConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        # Self-edit generation prompts
        self.self_edit_templates = {
            "qa_generation": """Given the following information, generate question-answer pairs that would help a language model learn this information:

Information: {context}

Generate 3-5 question-answer pairs in the following format:
Q: [question]
A: [answer]

Self-edit:""",
            
            "task_specific": """Given the following context and task, generate a self-improvement instruction:

Context: {context}
Task: {task}

Generate a specific instruction that tells the model how to update its knowledge for this task:

Self-edit:""",
            
            "reasoning_chain": """Given the following problem and solution, generate a step-by-step reasoning chain:

Problem: {context}
Solution: {solution}

Create a detailed reasoning chain that breaks down the solution:

Self-edit:""",
            
            "concept_integration": """Given the following new concept, generate instructions for integrating it with existing knowledge:

New Concept: {context}
Related Concepts: {related_concepts}

Generate integration instructions:

Self-edit:"""
        }
    
    def generate_self_edit(
        self,
        context: str,
        task_type: str = "qa_generation",
        additional_context: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a self-edit instruction for the given context
        
        Args:
            context: New information to learn
            task_type: Type of self-edit to generate
            additional_context: Additional context for template filling
        """
        # Select appropriate template
        template = self.self_edit_templates.get(task_type, self.self_edit_templates["qa_generation"])
        
        # Fill template
        if additional_context:
            prompt = template.format(context=context, **additional_context)
        else:
            prompt = template.format(context=context)
        
        # Generate self-edit
        with torch.no_grad():
            self_edit = self.model.generate(
                input_text=prompt,
                max_new_tokens=self.config.max_self_edit_length,
                temperature=self.config.self_edit_temperature,
                do_sample=True,
                top_p=0.9
            )
        
        # Extract self-edit from generated text
        if "Self-edit:" in self_edit:
            self_edit = self_edit.split("Self-edit:")[-1].strip()
        
        return self_edit
    
    def generate_multiple_self_edits(
        self,
        context: str,
        num_edits: int = 3,
        task_types: Optional[List[str]] = None
    ) -> List[str]:
        """Generate multiple diverse self-edits for the same context"""
        if task_types is None:
            task_types = list(self.self_edit_templates.keys())
        
        self_edits = []
        for i in range(num_edits):
            task_type = task_types[i % len(task_types)]
            self_edit = self.generate_self_edit(context, task_type)
            self_edits.append(self_edit)
        
        return self_edits
    
    def evaluate_self_edit_quality(self, self_edit: str, context: str) -> float:
        """
        Evaluate the quality of a generated self-edit
        Returns a score between 0 and 1
        """
        # Simple heuristic-based evaluation
        # In practice, this could use a trained evaluation model
        
        score = 0.0
        
        # Length check
        if 50 <= len(self_edit) <= 500:
            score += 0.2
        
        # Contains structured information
        if any(marker in self_edit.lower() for marker in ["q:", "a:", "step", "instruction", "learn"]):
            score += 0.2
        
        # References original context
        context_words = set(context.lower().split())
        edit_words = set(self_edit.lower().split())
        overlap = len(context_words.intersection(edit_words)) / len(context_words)
        score += min(overlap * 0.3, 0.3)
        
        # Coherence check (simple)
        sentences = self_edit.split('.')
        if len(sentences) >= 2:
            score += 0.2
        
        # Actionability check
        if any(action in self_edit.lower() for action in ["update", "learn", "remember", "integrate"]):
            score += 0.1
        
        return min(score, 1.0)


class AdaptationEngine(nn.Module):
    """
    Handles the actual parameter updates using LoRA
    Inner loop of SEAL algorithm
    """
    
    def __init__(self, model: JigyasaModel, config: SEALConfig):
        super().__init__()
        self.base_model = model
        self.config = config
        
        # Setup LoRA configuration
        # Check if model is GPT2 and adjust target modules accordingly
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            # GPT-2 uses different module names
            target_modules = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
        else:
            target_modules = config.target_modules
            
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Create LoRA-enhanced model
        self.lora_model = get_peft_model(model, self.lora_config)
        
        # Optimizer for adaptation
        self.optimizer = torch.optim.AdamW(
            self.lora_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Track adaptation history
        self.adaptation_history = []
        
    def adapt(
        self,
        self_edit: str,
        context: str,
        target_task: Optional[str] = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform parameter adaptation based on self-edit
        
        Args:
            self_edit: Self-generated training instruction
            context: Original context/information
            target_task: Specific task to adapt for
            num_steps: Number of adaptation steps
        """
        if num_steps is None:
            num_steps = self.config.inner_loop_steps
        
        # Create training data from self-edit
        training_data = self._create_training_data(self_edit, context, target_task)
        
        # Perform adaptation steps
        total_loss = 0.0
        self.lora_model.train()
        
        for step in range(num_steps):
            for batch in training_data:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.lora_model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lora_model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / (num_steps * len(training_data))
        
        # Store adaptation record
        adaptation_record = {
            'self_edit': self_edit,
            'context': context,
            'target_task': target_task,
            'num_steps': num_steps,
            'final_loss': avg_loss,
            'timestamp': torch.tensor(0)  # Would use actual timestamp
        }
        self.adaptation_history.append(adaptation_record)
        
        return {
            'loss': avg_loss,
            'num_steps': num_steps,
            'adapted_parameters': sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        }
    
    def _create_training_data(
        self,
        self_edit: str,
        context: str,
        target_task: Optional[str] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert self-edit into training data
        """
        training_examples = []
        
        # Parse self-edit to extract training examples
        if "Q:" in self_edit and "A:" in self_edit:
            # Question-answer format
            qa_pairs = self._parse_qa_pairs(self_edit)
            for q, a in qa_pairs:
                input_text = f"Question: {q}\nAnswer:"
                target_text = f"{a}"
                training_examples.append((input_text, target_text))
        
        elif "Step" in self_edit:
            # Step-by-step reasoning format
            steps = self._parse_reasoning_steps(self_edit)
            for i, step in enumerate(steps[:-1]):
                input_text = f"Context: {context}\nStep {i+1}:"
                target_text = step
                training_examples.append((input_text, target_text))
        
        else:
            # General instruction format
            input_text = f"Context: {context}\nInstruction: Apply the following learning:"
            target_text = self_edit
            training_examples.append((input_text, target_text))
        
        # Convert to model inputs
        training_data = []
        for input_text, target_text in training_examples:
            # Tokenize
            full_text = input_text + " " + target_text
            tokenized = self.base_model.tokenizer.batch_encode([full_text], return_tensors="pt")
            
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            # Create labels (same as input_ids, with -100 for input portion)
            input_length = len(self.base_model.tokenizer.encode(input_text, add_special_tokens=False))
            labels = input_ids.clone()
            labels[:, :input_length] = -100  # Ignore loss on input portion
            
            training_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        return training_data
    
    def _parse_qa_pairs(self, self_edit: str) -> List[Tuple[str, str]]:
        """Parse Q: A: format from self-edit"""
        qa_pairs = []
        lines = self_edit.split('\n')
        
        current_q = None
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:') and current_q:
                current_a = line[2:].strip()
                qa_pairs.append((current_q, current_a))
                current_q = None
        
        return qa_pairs
    
    def _parse_reasoning_steps(self, self_edit: str) -> List[str]:
        """Parse step-by-step reasoning from self-edit"""
        steps = []
        lines = self_edit.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Step') or line.startswith('1.') or line.startswith('-'):
                # Extract step content
                if ':' in line:
                    step_content = line.split(':', 1)[1].strip()
                else:
                    step_content = line
                steps.append(step_content)
        
        return steps
    
    def save_adapters(self, save_path: str):
        """Save LoRA adapters"""
        self.lora_model.save_pretrained(save_path)
    
    def load_adapters(self, load_path: str):
        """Load LoRA adapters"""
        self.lora_model.load_adapter(load_path)
    
    def reset_adapters(self):
        """Reset LoRA adapters to initial state"""
        # Reinitialize LoRA parameters
        for name, module in self.lora_model.named_modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


class SEALTrainer:
    """
    Main SEAL training coordinator
    Implements the full two-loop SEAL algorithm
    """
    
    def __init__(self, model: JigyasaModel, config: SEALConfig):
        self.model = model
        self.config = config
        
        # Initialize components
        self.self_edit_generator = SelfEditGenerator(model, config)
        self.adaptation_engine = AdaptationEngine(model, config)
        
        # Outer loop components
        self.policy_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate * 0.1  # Lower LR for policy updates
        )
        
        # Training history
        self.training_history = []
        self.current_episode = 0
        
    def train_episode(
        self,
        new_contexts: List[str],
        evaluation_tasks: List[Dict[str, Any]],
        episode_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train one SEAL episode
        
        Args:
            new_contexts: New information to learn
            evaluation_tasks: Tasks to evaluate adaptation effectiveness
            episode_id: Episode identifier
        """
        if episode_id is None:
            episode_id = self.current_episode
            self.current_episode += 1
        
        episode_metrics = {
            'episode_id': episode_id,
            'total_reward': 0.0,
            'adaptation_loss': 0.0,
            'num_adaptations': 0,
            'self_edit_quality': 0.0
        }
        
        # Inner loop: Generate self-edits and adapt
        adaptations = []
        
        for context in new_contexts:
            # Generate self-edit (inner loop)
            self_edit = self.self_edit_generator.generate_self_edit(context)
            
            # Evaluate self-edit quality
            edit_quality = self.self_edit_generator.evaluate_self_edit_quality(self_edit, context)
            episode_metrics['self_edit_quality'] += edit_quality
            
            # Perform adaptation
            adaptation_result = self.adaptation_engine.adapt(self_edit, context)
            adaptations.append({
                'context': context,
                'self_edit': self_edit,
                'adaptation_result': adaptation_result,
                'edit_quality': edit_quality
            })
            
            episode_metrics['adaptation_loss'] += adaptation_result['loss']
            episode_metrics['num_adaptations'] += 1
        
        # Evaluate adapted model on tasks (generate rewards)
        total_reward = 0.0
        for task in evaluation_tasks:
            reward = self._evaluate_task(task)
            total_reward += reward
        
        episode_metrics['total_reward'] = total_reward
        
        # Outer loop: Update policy based on rewards
        if len(adaptations) > 0:
            self._update_policy(adaptations, total_reward)
        
        # Normalize metrics
        if episode_metrics['num_adaptations'] > 0:
            episode_metrics['adaptation_loss'] /= episode_metrics['num_adaptations']
            episode_metrics['self_edit_quality'] /= episode_metrics['num_adaptations']
        
        # Store training history
        self.training_history.append(episode_metrics)
        
        # Log to wandb if available
        # if wandb.run is not None:
        #     wandb.log(episode_metrics)
        
        return episode_metrics
    
    def _evaluate_task(self, task: Dict[str, Any]) -> float:
        """
        Evaluate the adapted model on a specific task
        Returns reward signal (0-1)
        """
        task_type = task.get('type', 'qa')
        
        if task_type == 'qa':
            return self._evaluate_qa_task(task)
        elif task_type == 'reasoning':
            return self._evaluate_reasoning_task(task)
        elif task_type == 'generation':
            return self._evaluate_generation_task(task)
        else:
            return 0.0
    
    def _evaluate_qa_task(self, task: Dict[str, Any]) -> float:
        """Evaluate question-answering task"""
        question = task['question']
        expected_answer = task['answer']
        
        # Generate answer using adapted model
        prompt = f"Question: {question}\nAnswer:"
        generated_answer = self.adaptation_engine.lora_model.generate(
            input_text=prompt,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False
        )
        
        # Extract answer
        if "Answer:" in generated_answer:
            generated_answer = generated_answer.split("Answer:")[-1].strip()
        
        # Simple similarity-based reward
        # In practice, this could use more sophisticated evaluation
        similarity = self._calculate_text_similarity(generated_answer, expected_answer)
        return similarity
    
    def _evaluate_reasoning_task(self, task: Dict[str, Any]) -> float:
        """Evaluate reasoning task"""
        problem = task['problem']
        expected_solution = task['solution']
        
        # Generate solution
        prompt = f"Problem: {problem}\nSolution:"
        generated_solution = self.adaptation_engine.lora_model.generate(
            input_text=prompt,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True
        )
        
        # Extract solution
        if "Solution:" in generated_solution:
            generated_solution = generated_solution.split("Solution:")[-1].strip()
        
        # Evaluate reasoning quality
        reward = 0.0
        
        # Check for key concepts
        expected_concepts = set(expected_solution.lower().split())
        generated_concepts = set(generated_solution.lower().split())
        concept_overlap = len(expected_concepts.intersection(generated_concepts)) / len(expected_concepts)
        reward += concept_overlap * 0.5
        
        # Check for reasoning structure
        if any(indicator in generated_solution.lower() for indicator in ["because", "therefore", "since", "thus"]):
            reward += 0.3
        
        # Check length appropriateness
        if 50 <= len(generated_solution) <= 300:
            reward += 0.2
        
        return min(reward, 1.0)
    
    def _evaluate_generation_task(self, task: Dict[str, Any]) -> float:
        """Evaluate text generation task"""
        prompt = task['prompt']
        criteria = task.get('criteria', {})
        
        # Generate text
        generated_text = self.adaptation_engine.lora_model.generate(
            input_text=prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        # Evaluate based on criteria
        reward = 0.0
        
        for criterion, weight in criteria.items():
            if criterion == 'length':
                target_length = weight
                actual_length = len(generated_text.split())
                length_score = 1.0 - abs(actual_length - target_length) / target_length
                reward += max(length_score, 0) * 0.3
            
            elif criterion == 'keywords':
                required_keywords = weight
                found_keywords = sum(1 for kw in required_keywords if kw.lower() in generated_text.lower())
                reward += (found_keywords / len(required_keywords)) * 0.4
            
            elif criterion == 'coherence':
                # Simple coherence check
                sentences = generated_text.split('.')
                if len(sentences) >= 2:
                    reward += 0.3
        
        return min(reward, 1.0)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        elif len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def _update_policy(self, adaptations: List[Dict], total_reward: float):
        """
        Update the policy model based on adaptation results and rewards
        Outer loop of SEAL algorithm
        """
        # Simple policy gradient update
        # In practice, this could use more sophisticated RL algorithms
        
        policy_loss = 0.0
        
        for adaptation in adaptations:
            self_edit = adaptation['self_edit']
            context = adaptation['context']
            edit_quality = adaptation['edit_quality']
            
            # Reward shaped by edit quality and task performance
            shaped_reward = total_reward * edit_quality
            
            # Create training example for policy update
            prompt = f"Context: {context}\nGenerate a self-edit:"
            target = self_edit
            
            # Compute log probability of generated self-edit
            # This would require more sophisticated implementation in practice
            log_prob = self._compute_log_prob(prompt, target)
            
            # Policy gradient loss
            policy_loss += -log_prob * shaped_reward
        
        # Update policy parameters
        if policy_loss != 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.policy_optimizer.step()
    
    def _compute_log_prob(self, prompt: str, target: str) -> torch.Tensor:
        """Compute log probability of target given prompt"""
        # Simplified implementation
        # In practice, this would compute actual log probabilities
        full_text = prompt + " " + target
        tokenized = self.model.tokenizer.batch_encode([full_text], return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits
            
            # Compute log probabilities (simplified)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.mean()
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'adaptation_engine_state_dict': self.adaptation_engine.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'training_history': self.training_history,
            'current_episode': self.current_episode,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.adaptation_engine.load_state_dict(checkpoint['adaptation_engine_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.current_episode = checkpoint['current_episode']
    
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics over time"""
        metrics = {
            'episode_rewards': [],
            'adaptation_losses': [],
            'self_edit_qualities': []
        }
        
        for episode in self.training_history:
            metrics['episode_rewards'].append(episode['total_reward'])
            metrics['adaptation_losses'].append(episode['adaptation_loss'])
            metrics['self_edit_qualities'].append(episode['self_edit_quality'])
        
        return metrics