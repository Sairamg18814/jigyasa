"""
ProRL (Prolonged Reinforcement Learning) implementation
Extended RL training for advanced reasoning capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import math
import random
from dataclasses import dataclass
from tqdm import tqdm
# import wandb  # Optional - commented out for demo
import numpy as np
from transformers import get_linear_schedule_with_warmup

from ..config import ProRLConfig
from ..core.model import JigyasaModel


@dataclass
class VerifiableTask:
    """Container for tasks with verifiable solutions"""
    problem: str
    solution: str
    domain: str
    difficulty: float
    verification_fn: Optional[callable] = None
    metadata: Dict[str, Any] = None


class RewardModel(nn.Module):
    """
    Reward model for evaluating solution quality
    Trained on verifiable tasks to provide dense reward signals
    """
    
    def __init__(self, base_model: JigyasaModel, hidden_size: int = 768):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Reward between 0 and 1
        )
        
        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reward and value for given input
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask
        """
        # Get hidden states from base model
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = outputs['hidden_states']  # (batch_size, seq_len, hidden_size)
        
        # Pool hidden states (use last token representation)
        if attention_mask is not None:
            # Get last valid token position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            pooled_hidden = hidden_states[:, -1, :]  # (batch_size, hidden_size)
        
        # Compute reward and value
        reward = self.reward_head(pooled_hidden).squeeze(-1)  # (batch_size,)
        value = self.value_head(pooled_hidden).squeeze(-1)   # (batch_size,)
        
        return {
            'reward': reward,
            'value': value,
            'hidden_states': pooled_hidden
        }
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation) advantages
        
        Args:
            rewards: (batch_size,) or (seq_len,) rewards
            values: (batch_size,) or (seq_len,) value estimates
            gamma: Discount factor
            lam: GAE lambda parameter
        """
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)
            values = values.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = rewards.shape
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0
        
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0  # Terminal value
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae_lambda = delta + gamma * lam * last_gae_lambda
        
        # Compute returns
        returns = advantages + values
        
        if squeeze_output:
            advantages = advantages.squeeze(0)
            returns = returns.squeeze(0)
        
        return advantages, returns


class VerifiableTaskDataset:
    """
    Dataset of verifiable tasks for ProRL training
    Covers math, coding, logic, and STEM domains
    """
    
    def __init__(self, config: ProRLConfig):
        self.config = config
        self.tasks = []
        self._load_tasks()
    
    def _load_tasks(self):
        """Load or generate verifiable tasks"""
        # Math tasks
        self._add_math_tasks()
        
        # Coding tasks
        self._add_coding_tasks()
        
        # Logic tasks
        self._add_logic_tasks()
        
        # STEM tasks
        self._add_stem_tasks()
        
        # Shuffle tasks
        random.shuffle(self.tasks)
    
    def _add_math_tasks(self):
        """Add mathematical reasoning tasks"""
        # Arithmetic
        for _ in range(1000):
            a, b = random.randint(10, 999), random.randint(10, 999)
            op = random.choice(['+', '-', '*'])
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:
                result = a * b
            
            problem = f"Calculate: {a} {op} {b}"
            solution = str(result)
            
            self.tasks.append(VerifiableTask(
                problem=problem,
                solution=solution,
                domain="arithmetic",
                difficulty=0.2,
                verification_fn=lambda p, s: self._verify_arithmetic(p, s)
            ))
        
        # Algebra
        for _ in range(500):
            x = random.randint(1, 20)
            a = random.randint(2, 10)
            b = random.randint(1, 50)
            
            problem = f"Solve for x: {a}x + {b} = {a*x + b}"
            solution = str(x)
            
            self.tasks.append(VerifiableTask(
                problem=problem,
                solution=solution,
                domain="algebra",
                difficulty=0.4,
                verification_fn=lambda p, s: self._verify_algebra(p, s)
            ))
        
        # Geometry
        for _ in range(300):
            r = random.randint(1, 20)
            area = math.pi * r * r
            
            problem = f"Find the area of a circle with radius {r}. Use π ≈ 3.14159."
            solution = f"{area:.2f}"
            
            self.tasks.append(VerifiableTask(
                problem=problem,
                solution=solution,
                domain="geometry",
                difficulty=0.3,
                verification_fn=lambda p, s: self._verify_geometry(p, s)
            ))
    
    def _add_coding_tasks(self):
        """Add coding tasks"""
        # Simple function implementations
        coding_problems = [
            {
                "problem": "Write a function to check if a number is prime.",
                "solution": """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
                "difficulty": 0.5
            },
            {
                "problem": "Write a function to reverse a string.",
                "solution": """def reverse_string(s):
    return s[::-1]""",
                "difficulty": 0.2
            },
            {
                "problem": "Write a function to find the factorial of a number.",
                "solution": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
                "difficulty": 0.3
            }
        ]
        
        for prob_data in coding_problems:
            self.tasks.append(VerifiableTask(
                problem=prob_data["problem"],
                solution=prob_data["solution"],
                domain="coding",
                difficulty=prob_data["difficulty"],
                verification_fn=lambda p, s: self._verify_code(p, s)
            ))
    
    def _add_logic_tasks(self):
        """Add logical reasoning tasks"""
        # Syllogisms
        logic_problems = [
            {
                "problem": "All cats are mammals. All mammals are animals. Therefore, what can we conclude about cats?",
                "solution": "All cats are animals.",
                "difficulty": 0.3
            },
            {
                "problem": "If it rains, then the ground gets wet. The ground is wet. Can we conclude that it rained?",
                "solution": "No, we cannot conclude that it rained. The ground could be wet for other reasons.",
                "difficulty": 0.4
            }
        ]
        
        for prob_data in logic_problems:
            self.tasks.append(VerifiableTask(
                problem=prob_data["problem"],
                solution=prob_data["solution"],
                domain="logic",
                difficulty=prob_data["difficulty"],
                verification_fn=lambda p, s: self._verify_logic(p, s)
            ))
    
    def _add_stem_tasks(self):
        """Add STEM tasks"""
        # Physics
        physics_problems = [
            {
                "problem": "A ball is dropped from a height of 20 meters. How long does it take to reach the ground? (Use g = 9.8 m/s²)",
                "solution": "Using h = 0.5 * g * t², we get: 20 = 0.5 * 9.8 * t². Solving: t = √(40/9.8) ≈ 2.02 seconds.",
                "difficulty": 0.6
            }
        ]
        
        for prob_data in physics_problems:
            self.tasks.append(VerifiableTask(
                problem=prob_data["problem"],
                solution=prob_data["solution"],
                domain="physics",
                difficulty=prob_data["difficulty"],
                verification_fn=lambda p, s: self._verify_physics(p, s)
            ))
    
    def _verify_arithmetic(self, problem: str, solution: str) -> float:
        """Verify arithmetic solution"""
        try:
            # Extract numbers and operation from problem
            import re
            numbers = re.findall(r'\d+', problem)
            if '+' in problem:
                expected = int(numbers[0]) + int(numbers[1])
            elif '-' in problem:
                expected = int(numbers[0]) - int(numbers[1])
            elif '*' in problem:
                expected = int(numbers[0]) * int(numbers[1])
            else:
                return 0.0
            
            # Check if solution matches
            try:
                given = int(solution.strip())
                return 1.0 if given == expected else 0.0
            except:
                return 0.0
        except:
            return 0.0
    
    def _verify_algebra(self, problem: str, solution: str) -> float:
        """Verify algebra solution (simplified)"""
        # For now, simple check - in practice would be more sophisticated
        try:
            given = int(solution.strip())
            return 1.0 if 1 <= given <= 50 else 0.0  # Reasonable range
        except:
            return 0.0
    
    def _verify_geometry(self, problem: str, solution: str) -> float:
        """Verify geometry solution"""
        try:
            # Extract radius and check area calculation
            import re
            radius_match = re.search(r'radius (\d+)', problem)
            if radius_match:
                r = int(radius_match.group(1))
                expected_area = math.pi * r * r
                given_area = float(solution.strip())
                error = abs(given_area - expected_area) / expected_area
                return max(0.0, 1.0 - error * 2)  # Allow some tolerance
        except:
            return 0.0
        return 0.0
    
    def _verify_code(self, problem: str, solution: str) -> float:
        """Verify code solution (simplified)"""
        # Simple heuristic checks
        score = 0.0
        
        if "def " in solution:
            score += 0.3
        if "return" in solution:
            score += 0.3
        if len(solution.strip()) > 20:
            score += 0.2
        if ":" in solution:
            score += 0.2
        
        return min(score, 1.0)
    
    def _verify_logic(self, problem: str, solution: str) -> float:
        """Verify logic solution (simplified)"""
        # Check for logical reasoning indicators
        score = 0.0
        solution_lower = solution.lower()
        
        if any(word in solution_lower for word in ["therefore", "conclude", "because", "since"]):
            score += 0.4
        if len(solution.strip()) > 10:
            score += 0.3
        if "." in solution:
            score += 0.3
        
        return min(score, 1.0)
    
    def _verify_physics(self, problem: str, solution: str) -> float:
        """Verify physics solution (simplified)"""
        # Check for physics reasoning
        score = 0.0
        solution_lower = solution.lower()
        
        if any(word in solution_lower for word in ["formula", "equation", "using", "="]):
            score += 0.4
        if any(unit in solution_lower for unit in ["seconds", "meters", "m/s"]):
            score += 0.3
        if len(solution.strip()) > 20:
            score += 0.3
        
        return min(score, 1.0)
    
    def get_batch(self, batch_size: int) -> List[VerifiableTask]:
        """Get a batch of tasks"""
        return random.sample(self.tasks, min(batch_size, len(self.tasks)))
    
    def get_domain_tasks(self, domain: str, count: int) -> List[VerifiableTask]:
        """Get tasks from specific domain"""
        domain_tasks = [task for task in self.tasks if task.domain == domain]
        return random.sample(domain_tasks, min(count, len(domain_tasks)))


class ProRLTrainer:
    """
    Main ProRL training coordinator
    Implements prolonged reinforcement learning with KL control and reference resetting
    """
    
    def __init__(self, model: JigyasaModel, config: ProRLConfig):
        self.model = model
        self.config = config
        
        # Create reference model (frozen copy)
        self.reference_model = self._create_reference_model()
        
        # Initialize reward model
        self.reward_model = RewardModel(model)
        
        # Dataset
        self.dataset = VerifiableTaskDataset(config)
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.reward_optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=config.learning_rate * 0.5
        )
        
        # Learning rate schedulers
        self.policy_scheduler = get_linear_schedule_with_warmup(
            self.policy_optimizer,
            num_warmup_steps=100,
            num_training_steps=config.num_episodes
        )
        
        # Training state
        self.current_episode = 0
        self.training_history = []
        self.reference_reset_counter = 0
        
    def _create_reference_model(self) -> JigyasaModel:
        """Create frozen reference model"""
        # Get the base model if it's PEFT-wrapped
        base_model = getattr(self.model, 'base_model', self.model)
        if hasattr(base_model, 'model'):
            base_model = base_model.model
        
        # Create a copy of the model
        reference_model = type(base_model)(base_model.config)
        reference_model.load_state_dict(base_model.state_dict())
        
        # Freeze parameters
        for param in reference_model.parameters():
            param.requires_grad = False
        
        reference_model.eval()
        return reference_model
    
    def train(self, num_episodes: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Main training loop for ProRL
        
        Args:
            num_episodes: Number of episodes to train (uses config if None)
        """
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        print(f"Starting ProRL training for {num_episodes} episodes...")
        
        for episode in tqdm(range(num_episodes), desc="ProRL Training"):
            episode_metrics = self.train_episode()
            self.training_history.append(episode_metrics)
            
            # Check for reference policy reset
            if (episode + 1) % self.config.reset_frequency == 0:
                self._reset_reference_policy()
                self.reference_reset_counter += 1
            
            # Log progress
            if episode % 100 == 0:
                self._log_progress(episode, episode_metrics)
            
            self.current_episode += 1
        
        return self._get_training_metrics()
    
    def train_episode(self) -> Dict[str, float]:
        """Train one ProRL episode"""
        episode_metrics = {
            'episode': self.current_episode,
            'total_reward': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'kl_divergence': 0.0,
            'entropy': 0.0,
            'success_rate': 0.0
        }
        
        # Get batch of tasks
        tasks = self.dataset.get_batch(self.config.batch_size)
        
        # Generate solutions and collect trajectories
        trajectories = []
        total_reward = 0.0
        successful_solutions = 0
        
        for task in tasks:
            trajectory = self._generate_trajectory(task)
            trajectories.append(trajectory)
            
            # Evaluate solution
            if task.verification_fn:
                reward = task.verification_fn(task.problem, trajectory['solution'])
            else:
                reward = self._evaluate_solution_heuristic(task, trajectory['solution'])
            
            trajectory['reward'] = reward
            total_reward += reward
            
            if reward > 0.8:  # Consider successful if reward > 0.8
                successful_solutions += 1
        
        episode_metrics['total_reward'] = total_reward / len(tasks)
        episode_metrics['success_rate'] = successful_solutions / len(tasks)
        
        # Compute advantages and update policy
        policy_loss, value_loss, kl_div, entropy = self._update_policy(trajectories)
        
        episode_metrics['policy_loss'] = policy_loss
        episode_metrics['value_loss'] = value_loss
        episode_metrics['kl_divergence'] = kl_div
        episode_metrics['entropy'] = entropy
        
        return episode_metrics
    
    def _generate_trajectory(self, task: VerifiableTask) -> Dict[str, Any]:
        """Generate solution trajectory for a task"""
        # Create prompt
        prompt = f"Problem: {task.problem}\nSolution:"
        
        # Tokenize
        tokenized = self.model.tokenizer.batch_encode([prompt], return_tensors="pt")
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        # Generate solution
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.model.tokenizer.pad_token_id
            )
        
        # Extract solution
        solution = self.model.tokenizer.decode(
            generated[0][input_ids.size(1):], 
            skip_special_tokens=True
        ).strip()
        
        # Get model outputs for the full sequence
        full_outputs = self.model(generated, attention_mask=None)
        logits = full_outputs['logits']
        
        # Get value estimate from reward model
        reward_outputs = self.reward_model(generated)
        value = reward_outputs['value']
        
        return {
            'task': task,
            'prompt': prompt,
            'solution': solution,
            'input_ids': generated,
            'logits': logits,
            'value': value,
            'log_probs': F.log_softmax(logits, dim=-1)
        }
    
    def _evaluate_solution_heuristic(self, task: VerifiableTask, solution: str) -> float:
        """Heuristic evaluation when verification function not available"""
        score = 0.0
        
        # Length check
        if 20 <= len(solution) <= 500:
            score += 0.2
        
        # Domain-specific checks
        if task.domain == "math":
            if any(char in solution for char in "0123456789+-*/="):
                score += 0.3
            if any(word in solution.lower() for word in ["calculate", "solve", "answer"]):
                score += 0.2
        
        elif task.domain == "coding":
            if any(keyword in solution for keyword in ["def", "return", "if", "for", "while"]):
                score += 0.4
            if ":" in solution:
                score += 0.2
        
        elif task.domain == "logic":
            if any(word in solution.lower() for word in ["therefore", "because", "since", "thus"]):
                score += 0.3
            if len(solution.split('.')) >= 2:
                score += 0.2
        
        # Coherence check
        if solution.strip() and not solution.strip().startswith(("error", "Error", "I don't")):
            score += 0.3
        
        return min(score, 1.0)
    
    def _update_policy(self, trajectories: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        """Update policy using PPO-style objective with KL control"""
        
        # Collect data from trajectories
        all_logits = []
        all_values = []
        all_rewards = []
        all_input_ids = []
        
        for traj in trajectories:
            all_logits.append(traj['logits'])
            all_values.append(traj['value'])
            all_rewards.append(traj['reward'])
            all_input_ids.append(traj['input_ids'])
        
        # Stack tensors
        batch_logits = torch.cat(all_logits, dim=0)
        batch_values = torch.cat(all_values, dim=0) if all_values[0].dim() > 0 else torch.stack(all_values)
        batch_rewards = torch.tensor(all_rewards, device=batch_logits.device)
        batch_input_ids = torch.cat(all_input_ids, dim=0)
        
        # Compute advantages
        advantages, returns = self.reward_model.compute_advantages(
            batch_rewards.unsqueeze(-1), 
            batch_values.unsqueeze(-1)
        )
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get reference model log probabilities for KL penalty
        with torch.no_grad():
            ref_outputs = self.reference_model(batch_input_ids)
            ref_log_probs = F.log_softmax(ref_outputs['logits'], dim=-1)
        
        # Current model log probabilities
        current_log_probs = F.log_softmax(batch_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(current_log_probs, ref_log_probs, reduction='batchmean', log_target=True)
        
        # Policy loss with KL penalty
        policy_loss = -advantages.mean() + self.config.kl_penalty * kl_div
        
        # Value loss
        value_loss = F.mse_loss(batch_values, returns)
        
        # Entropy for exploration
        entropy = -(current_log_probs * current_log_probs.exp()).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        
        # Update reward model
        reward_loss = value_loss  # Same as value loss for now
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
        self.reward_optimizer.step()
        
        return (
            policy_loss.item(),
            value_loss.item(), 
            kl_div.item(),
            entropy.item()
        )
    
    def _reset_reference_policy(self):
        """Reset reference policy to current policy"""
        print(f"Resetting reference policy at episode {self.current_episode}")
        
        # Copy current model weights to reference model
        self.reference_model.load_state_dict(self.model.state_dict())
        
        # Freeze reference model parameters
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.reference_model.eval()
    
    def _log_progress(self, episode: int, metrics: Dict[str, float]):
        """Log training progress"""
        print(f"Episode {episode}:")
        print(f"  Reward: {metrics['total_reward']:.3f}")
        print(f"  Success Rate: {metrics['success_rate']:.3f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.3f}")
        print(f"  KL Divergence: {metrics['kl_divergence']:.3f}")
        
        # Log to wandb if available
        # if wandb.run is not None:
        #     wandb.log({
        #         'episode': episode,
        #         'prorl_reward': metrics['total_reward'],
        #         'prorl_success_rate': metrics['success_rate'],
        #         'prorl_policy_loss': metrics['policy_loss'],
        #         'prorl_kl_divergence': metrics['kl_divergence']
        #     })
    
    def _get_training_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics over time"""
        metrics = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'policy_losses': [],
            'kl_divergences': []
        }
        
        for episode_data in self.training_history:
            metrics['episodes'].append(episode_data['episode'])
            metrics['rewards'].append(episode_data['total_reward'])
            metrics['success_rates'].append(episode_data['success_rate'])
            metrics['policy_losses'].append(episode_data['policy_loss'])
            metrics['kl_divergences'].append(episode_data['kl_divergence'])
        
        return metrics
    
    def evaluate_on_domain(self, domain: str, num_tasks: int = 100) -> Dict[str, float]:
        """Evaluate model on specific domain"""
        tasks = self.dataset.get_domain_tasks(domain, num_tasks)
        
        total_reward = 0.0
        successful_solutions = 0
        
        self.model.eval()
        with torch.no_grad():
            for task in tqdm(tasks, desc=f"Evaluating on {domain}"):
                trajectory = self._generate_trajectory(task)
                
                if task.verification_fn:
                    reward = task.verification_fn(task.problem, trajectory['solution'])
                else:
                    reward = self._evaluate_solution_heuristic(task, trajectory['solution'])
                
                total_reward += reward
                if reward > 0.8:
                    successful_solutions += 1
        
        self.model.train()
        
        return {
            'domain': domain,
            'average_reward': total_reward / len(tasks),
            'success_rate': successful_solutions / len(tasks),
            'num_tasks': len(tasks)
        }
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'reference_model_state_dict': self.reference_model.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'reward_optimizer_state_dict': self.reward_optimizer.state_dict(),
            'training_history': self.training_history,
            'current_episode': self.current_episode,
            'reference_reset_counter': self.reference_reset_counter,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.reference_model.load_state_dict(checkpoint['reference_model_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.current_episode = checkpoint['current_episode']
        self.reference_reset_counter = checkpoint['reference_reset_counter']