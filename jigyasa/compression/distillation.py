"""
Knowledge Distillation for Model Compression
Implements teacher-student training for creating smaller, efficient models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm
# import wandb  # Optional - commented out for demo
import numpy as np
from transformers import get_linear_schedule_with_warmup

from ..core.model import JigyasaModel, create_jigyasa_model
from ..config import CompressionConfig


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for student loss
    learning_rate: float = 5e-5
    num_epochs: int = 5
    batch_size: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500


class TeacherStudentPair:
    """
    Manages teacher and student models for distillation
    """
    
    def __init__(
        self,
        teacher_model: JigyasaModel,
        student_config: Dict[str, Any] = None,
        compression_ratio: float = 0.25
    ):
        self.teacher_model = teacher_model
        self.compression_ratio = compression_ratio
        
        # Create student model configuration
        if student_config is None:
            student_config = self._create_student_config()
        
        # Create student model
        self.student_model = create_jigyasa_model(**student_config)
        
        # Model information
        self.teacher_params = sum(p.numel() for p in teacher_model.parameters())
        self.student_params = sum(p.numel() for p in self.student_model.parameters())
        
        logging.info(f"Teacher model parameters: {self.teacher_params:,}")
        logging.info(f"Student model parameters: {self.student_params:,}")
        logging.info(f"Compression ratio: {self.student_params / self.teacher_params:.3f}")
    
    def _create_student_config(self) -> Dict[str, Any]:
        """Create student model configuration based on teacher"""
        teacher_config = self.teacher_model.config
        
        # Calculate compressed dimensions
        student_config = {
            'vocab_size': teacher_config.vocab_size,
            'd_model': int(teacher_config.d_model * (self.compression_ratio ** 0.5)),
            'n_heads': max(4, int(teacher_config.n_heads * self.compression_ratio)),
            'n_layers': max(6, int(teacher_config.n_layers * self.compression_ratio)),
            'd_ff': int(teacher_config.d_ff * self.compression_ratio),
            'max_seq_length': teacher_config.max_seq_length,
            'dropout': teacher_config.dropout,
        }
        
        # Ensure dimensions are compatible
        student_config['d_model'] = (student_config['d_model'] // student_config['n_heads']) * student_config['n_heads']
        
        return student_config
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get detailed compression statistics"""
        return {
            'teacher_parameters': self.teacher_params,
            'student_parameters': self.student_params,
            'compression_ratio': self.student_params / self.teacher_params,
            'parameter_reduction': 1 - (self.student_params / self.teacher_params),
            'teacher_config': {
                'd_model': getattr(self.teacher_model.config, 'd_model', 768),
                'n_heads': self.teacher_model.config.n_heads,
                'n_layers': self.teacher_model.config.n_layers,
                'd_ff': self.teacher_model.config.d_ff,
            },
            'student_config': {
                'd_model': getattr(self.student_model.config, 'd_model', 768),
                'n_heads': self.student_model.config.n_heads,
                'n_layers': self.student_model.config.n_layers,
                'd_ff': self.student_model.config.d_ff,
            }
        }


class DistillationLoss(nn.Module):
    """
    Combined loss function for knowledge distillation
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta    # Weight for ground truth loss
        
        assert abs(alpha + beta - 1.0) < 1e-6, "Alpha and beta should sum to 1.0"
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels (optional)
        """
        # Distillation loss (KL divergence between teacher and student)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Ground truth loss (if labels provided)
        if labels is not None:
            student_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            student_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + self.beta * student_loss
        
        return {
            'total_loss': total_loss,
            'distillation_loss': distillation_loss,
            'student_loss': student_loss
        }


class FeatureMatchingLoss(nn.Module):
    """
    Feature-level distillation loss for intermediate representations
    """
    
    def __init__(self, feature_weight: float = 0.1):
        super().__init__()
        self.feature_weight = feature_weight
    
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feature matching loss between teacher and student intermediate layers
        """
        if len(student_features) != len(teacher_features):
            # If different number of layers, select corresponding layers
            teacher_indices = np.linspace(0, len(teacher_features) - 1, len(student_features), dtype=int)
            teacher_features = [teacher_features[i] for i in teacher_indices]
        
        feature_loss = 0.0
        
        for student_feat, teacher_feat in zip(student_features, teacher_features):
            # Ensure same dimensions (project if necessary)
            if student_feat.size(-1) != teacher_feat.size(-1):
                # Simple linear projection (could be learnable)
                teacher_feat = F.linear(
                    teacher_feat,
                    torch.randn(student_feat.size(-1), teacher_feat.size(-1), device=student_feat.device) * 0.02
                )
            
            # MSE loss between features
            feature_loss += F.mse_loss(student_feat, teacher_feat.detach())
        
        return feature_loss * self.feature_weight


class KnowledgeDistillationTrainer:
    """
    Main trainer for knowledge distillation
    """
    
    def __init__(
        self,
        teacher_student_pair: TeacherStudentPair,
        config: DistillationConfig = None
    ):
        self.teacher_model = teacher_student_pair.teacher_model
        self.student_model = teacher_student_pair.student_model
        self.pair = teacher_student_pair
        
        self.config = config or DistillationConfig()
        
        # Loss functions
        self.distillation_loss = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            beta=self.config.beta
        )
        self.feature_loss = FeatureMatchingLoss()
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.global_step = 0
        self.training_history = []
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        save_dir: str = "./distilled_model"
    ) -> Dict[str, Any]:
        """
        Train student model using knowledge distillation
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Validation data loader (optional)
            save_dir: Directory to save the distilled model
        """
        logging.info("Starting knowledge distillation training...")
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.student_model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_distillation_loss = 0.0
            epoch_student_loss = 0.0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.student_model.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.student_model.device)
                labels = batch.get('labels', None)
                if labels is not None:
                    labels = labels.to(self.student_model.device)
                
                # Forward pass through teacher (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs['logits']
                
                # Forward pass through student
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs['logits']
                
                # Compute distillation loss
                loss_dict = self.distillation_loss(
                    student_logits, teacher_logits, labels
                )
                
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                scheduler.step()
                
                # Update statistics
                epoch_loss += total_loss.item()
                epoch_distillation_loss += loss_dict['distillation_loss'].item()
                epoch_student_loss += loss_dict['student_loss'].item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'distill': f"{loss_dict['distillation_loss'].item():.4f}",
                    'student': f"{loss_dict['student_loss'].item():.4f}"
                })
                
                self.global_step += 1
                
                # Evaluation
                if (self.global_step % self.config.eval_steps == 0 and 
                    eval_dataloader is not None):
                    eval_results = self.evaluate(eval_dataloader)
                    logging.info(f"Evaluation at step {self.global_step}: {eval_results}")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(save_dir, self.global_step)
            
            # Epoch statistics
            num_batches = len(train_dataloader)
            epoch_stats = {
                'epoch': epoch + 1,
                'avg_loss': epoch_loss / num_batches,
                'avg_distillation_loss': epoch_distillation_loss / num_batches,
                'avg_student_loss': epoch_student_loss / num_batches,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            self.training_history.append(epoch_stats)
            
            logging.info(f"Epoch {epoch + 1} completed: {epoch_stats}")
            
            # Log to wandb if available
            # if wandb.run is not None:
            #     wandb.log(epoch_stats)
        
        # Save final model
        self.save_model(save_dir)
        
        return {
            'training_history': self.training_history,
            'final_model_path': save_dir,
            'compression_stats': self.pair.get_compression_statistics()
        }
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate student model on validation set"""
        self.student_model.eval()
        
        total_loss = 0.0
        total_distillation_loss = 0.0
        total_student_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.student_model.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.student_model.device)
                labels = batch.get('labels', None)
                if labels is not None:
                    labels = labels.to(self.student_model.device)
                
                # Teacher predictions
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Student predictions
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute losses
                loss_dict = self.distillation_loss(
                    student_outputs['logits'],
                    teacher_outputs['logits'],
                    labels
                )
                
                total_loss += loss_dict['total_loss'].item()
                total_distillation_loss += loss_dict['distillation_loss'].item()
                total_student_loss += loss_dict['student_loss'].item()
                num_batches += 1
        
        self.student_model.train()
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_distillation_loss': total_distillation_loss / num_batches,
            'eval_student_loss': total_student_loss / num_batches
        }
    
    def save_checkpoint(self, save_dir: str, step: int):
        """Save training checkpoint"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint-{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint at step {step}")
    
    def save_model(self, save_dir: str):
        """Save the final distilled model"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save student model
        self.student_model.save_pretrained(save_dir)
        
        # Save compression metadata
        compression_info = {
            'compression_stats': self.pair.get_compression_statistics(),
            'distillation_config': {
                'temperature': self.config.temperature,
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'num_epochs': self.config.num_epochs
            },
            'training_history': self.training_history
        }
        
        with open(os.path.join(save_dir, 'compression_info.json'), 'w') as f:
            json.dump(compression_info, f, indent=2)
        
        logging.info(f"Saved distilled model to {save_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['step']
        self.training_history = checkpoint['training_history']
        
        logging.info(f"Loaded checkpoint from step {self.global_step}")
    
    def compare_models(self, test_dataloader, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare teacher and student model performance
        
        Args:
            test_dataloader: Test data loader
            metrics: List of metrics to compute
        """
        if metrics is None:
            metrics = ['perplexity', 'accuracy']
        
        results = {
            'teacher': {},
            'student': {},
            'performance_retention': {}
        }
        
        # Evaluate both models
        for model_name, model in [('teacher', self.teacher_model), ('student', self.student_model)]:
            model.eval()
            
            total_loss = 0.0
            total_tokens = 0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].to(model.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(model.device)
                    labels = batch.get('labels', None)
                    if labels is not None:
                        labels = labels.to(model.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    if 'perplexity' in metrics and outputs.get('loss') is not None:
                        total_loss += outputs['loss'].item() * input_ids.size(0)
                        total_tokens += input_ids.size(0)
                    
                    if 'accuracy' in metrics and labels is not None:
                        predictions = outputs['logits'].argmax(dim=-1)
                        mask = labels != -100
                        correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                        total_predictions += mask.sum().item()
            
            # Calculate metrics
            if 'perplexity' in metrics and total_tokens > 0:
                avg_loss = total_loss / total_tokens
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                results[model_name]['perplexity'] = perplexity
            
            if 'accuracy' in metrics and total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                results[model_name]['accuracy'] = accuracy
        
        # Calculate performance retention
        for metric in metrics:
            if metric in results['teacher'] and metric in results['student']:
                if metric == 'perplexity':
                    # Lower is better for perplexity
                    retention = results['teacher'][metric] / results['student'][metric]
                else:
                    # Higher is better for accuracy
                    retention = results['student'][metric] / results['teacher'][metric]
                
                results['performance_retention'][metric] = retention
        
        return results


# Utility functions for distillation
def create_teacher_student_pair(
    teacher_model: JigyasaModel,
    compression_ratio: float = 0.25,
    student_config: Optional[Dict[str, Any]] = None
) -> TeacherStudentPair:
    """
    Convenience function to create teacher-student pair
    """
    return TeacherStudentPair(
        teacher_model=teacher_model,
        student_config=student_config,
        compression_ratio=compression_ratio
    )


def distill_model(
    teacher_model: JigyasaModel,
    train_dataloader,
    compression_ratio: float = 0.25,
    config: Optional[DistillationConfig] = None,
    save_dir: str = "./distilled_model"
) -> JigyasaModel:
    """
    High-level function to distill a model
    
    Args:
        teacher_model: Pre-trained teacher model
        train_dataloader: Training data
        compression_ratio: Target compression ratio
        config: Distillation configuration
        save_dir: Directory to save distilled model
    
    Returns:
        Distilled student model
    """
    # Create teacher-student pair
    pair = create_teacher_student_pair(teacher_model, compression_ratio)
    
    # Create trainer
    trainer = KnowledgeDistillationTrainer(pair, config)
    
    # Train
    results = trainer.train(train_dataloader, save_dir=save_dir)
    
    logging.info(f"Distillation completed. Results: {results}")
    
    return trainer.student_model