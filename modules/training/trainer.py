"""
Training Loop for T5 Classification with Sentiment Gate

Simplified trainer for classification (no decoder).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Optional, Dict
import json

from ..models.t5_sentiment_gate import T5ForSentimentClassification
from .config import TrainingConfig

logger = logging.getLogger(__name__)


class SentimentGateTrainer:
    """
    Trainer for T5 classification with sentiment gate.
    
    Simpler than seq2seq trainer - just classification loss.
    """
    
    def __init__(
        self,
        model: T5ForSentimentClassification,
        tokenizer: T5Tokenizer,
        config: TrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Will be set during training
        self.scheduler = None
        self.global_step = 0
        self.best_val_acc = 0.0
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with separate learning rates.
        
        Gate gets higher LR for faster learning.
        """
        # Collect parameters
        encoder_params = set(self.model.encoder.parameters())
        gate_params = set(self.model.sentiment_gate.parameters())
        classifier_params = set(self.model.classifier.parameters())
        
        param_groups = [
            {
                'params': list(encoder_params),
                'lr': self.config.encoder_lr,
                'name': 'encoder'
            },
            {
                'params': list(gate_params),
                'lr': self.config.gate_lr,
                'name': 'gate'
            },
            {
                'params': list(classifier_params),
                'lr': self.config.decoder_lr,  # Reuse decoder_lr for classifier
                'name': 'classifier'
            }
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
        
        logger.info(f"Optimizer: encoder_lr={self.config.encoder_lr}, "
                   f"gate_lr={self.config.gate_lr}, "
                   f"classifier_lr={self.config.decoder_lr}")
        
        return optimizer
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict:
        """
        Main training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        
        Returns:
            training_stats: Dict with loss history, best accuracy, etc.
        """
        # Calculate total steps
        if self.config.max_steps:
            total_steps = self.config.max_steps
            num_epochs = (total_steps // len(train_loader)) + 1
        else:
            num_epochs = self.config.num_epochs
            total_steps = len(train_loader) * num_epochs
        
        # Create scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Starting training: {num_epochs} epochs, {total_steps} total steps")
        
        # Training stats
        stats = {
            'train_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss = self._train_epoch(train_loader)
            stats['train_loss'].append(train_loss)
            
            # Evaluate
            val_acc = self.evaluate(val_loader)
            stats['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log RL status if enabled
            if self.config.use_rl and epoch + 1 >= self.config.rl_start_epoch:
                logger.info(f"  RL Mode: ACTIVE (weight={self.config.rl_weight})")
            elif self.config.use_rl:
                logger.info(f"  RL Mode: Waiting (starts at epoch {self.config.rl_start_epoch})")
            
            # Save best model
            if val_acc > stats['best_val_acc']:
                stats['best_val_acc'] = val_acc
                stats['best_epoch'] = epoch + 1
                self._save_checkpoint('best_model')
                logger.info(f"✓ New best model saved! Accuracy: {val_acc:.4f}")
            
            # Check if max_steps reached
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                logger.info(f"Reached max_steps={self.config.max_steps}, stopping training")
                break
        
        # Save final model
        self._save_checkpoint('final_model')
        
        # Save training stats
        stats_path = Path(self.config.output_dir) / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\nTraining complete!")
        logger.info(f"Best validation accuracy: {stats['best_val_acc']:.4f} (epoch {stats['best_epoch']})")
        
        return stats
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_rl_loss = 0.0
        total_rewards = 0.0
        num_rl_batches = 0
        
        # Determine if we should use RL this epoch
        # RL starts after rl_start_epoch * steps_per_epoch
        steps_per_epoch = len(train_loader)
        rl_start_step = self.config.rl_start_epoch * steps_per_epoch
        use_rl = self.config.use_rl and (self.global_step >= rl_start_step)
        
        pbar = tqdm(train_loader, desc=f"Training ({'RL' if use_rl else 'Supervised'})")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                use_gate=self.config.use_gate,
                use_rl=use_rl,
                rl_weight=self.config.rl_weight
            )
            
            loss = outputs['loss']
            
            # Track RL metrics if using RL
            if use_rl and 'rl_loss' in outputs:
                total_rl_loss += outputs['rl_loss'].item()
                total_rewards += outputs['rewards'].mean().item()
                num_rl_batches += 1
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            postfix = {'loss': f'{loss.item():.4f}'}
            if use_rl and num_rl_batches > 0:
                postfix['rl_loss'] = f'{total_rl_loss/num_rl_batches:.4f}'
                postfix['avg_reward'] = f'{total_rewards/num_rl_batches:.2f}'
            pbar.set_postfix(postfix)
            
            # Check if max_steps reached
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        avg_loss = total_loss / len(pbar)
        
        # Log RL stats if used
        if use_rl and num_rl_batches > 0:
            logger.info(f"  RL Stats - Avg RL Loss: {total_rl_loss/num_rl_batches:.4f}, "
                       f"Avg Reward: {total_rewards/num_rl_batches:.2f}")
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate on validation set.
        
        Returns:
            accuracy: Validation accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Get predictions
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)
            
            # Compare to labels
            labels = batch['labels']
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        accuracy = correct / total
        return accuracy
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint in HuggingFace format."""
        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict (PyTorch format)
        torch.save(self.model.state_dict(), save_dir / 'pytorch_model.bin')
        
        # Save model config (HuggingFace format)
        self.model.config.save_pretrained(save_dir)
        
        # Save custom model info
        model_info = {
            'model_type': 'T5ForSentimentClassification',
            'num_labels': self.model.num_labels,
            'use_gate': True
        }
        with open(save_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        logger.info(f"Checkpoint saved to {save_dir}")
