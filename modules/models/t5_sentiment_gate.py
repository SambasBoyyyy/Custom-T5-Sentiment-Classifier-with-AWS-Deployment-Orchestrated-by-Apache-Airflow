"""
T5 Encoder with Sentiment Gate + Classification Head

This is a more effective approach for classification:
1. Use T5 encoder only (no decoder)
2. Learnable gate for token importance
3. Weighted pooling based on gate scores
4. Direct classification head

This is faster, simpler, and more interpretable than the decoder approach.
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SentimentGate(nn.Module):
    """
    Learnable gate that scores token importance for sentiment classification.
    
    Architecture:
        Input: encoder_hidden_states [batch, seq_len, hidden_size]
        → Linear(hidden_size, 1)
        → Sigmoid
        Output: importance_scores [batch, seq_len, 1]
    """
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.gate_projection = nn.Linear(hidden_size, 1)
        
        # Initialize with neutral bias (sigmoid(0) = 0.5)
        nn.init.constant_(self.gate_projection.bias, 0.0)
        nn.init.normal_(self.gate_projection.weight, mean=0.0, std=0.01)
    
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for each token.
        
        Args:
            encoder_hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            importance_scores: [batch_size, seq_len, 1] in range [0, 1]
        """
        scores = self.gate_projection(encoder_hidden_states)
        importance = torch.sigmoid(scores)
        return importance


class T5ForSentimentClassification(nn.Module):
    """
    T5 Encoder + Sentiment Gate + Classification Head
    
    Architecture:
        Input → T5 Encoder → Gate → Weighted Pooling → Classifier → Logits
    
    This is more effective than using the decoder because:
    1. Faster (no decoder)
    2. Gate directly affects classification
    3. More interpretable
    4. Better for classification tasks
    
    Usage:
        model = T5ForSentimentClassification.from_pretrained('t5-small')
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        logits = outputs['logits']  # [batch, 2]
    """
    
    def __init__(self, config: T5Config, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # T5 Encoder only
        self.encoder = T5EncoderModel(config)
        
        # Sentiment gate
        self.sentiment_gate = SentimentGate(hidden_size=config.d_model)
        
        # Classification head
        self.classifier = nn.Linear(config.d_model, num_labels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Flag to enable/disable gate
        self.use_gate = True
        
        logger.info(f"Initialized T5ForSentimentClassification with hidden_size={config.d_model}, num_labels={num_labels}")
    
    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int = 2):
        """Load from pretrained T5 model"""
        encoder = T5EncoderModel.from_pretrained(model_name)
        config = encoder.config
        
        model = cls(config, num_labels)
        model.encoder = encoder
        
        return model
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_gate: Optional[bool] = None,
        return_gate_scores: bool = False,
        use_rl: bool = False,
        rl_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional RL training.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] - class labels (0 or 1)
            use_gate: If True, use gate; if False, use mean pooling
            return_gate_scores: If True, return gate scores for visualization
            use_rl: If True, add RL policy gradient loss
            rl_weight: Weight for RL loss component
        
        Returns:
            {
                'loss': scalar (if labels provided),
                'logits': [batch, num_labels],
                'gate_scores': [batch, seq_len] (if return_gate_scores=True),
                'rl_loss': scalar (if use_rl=True),
                'rewards': [batch] (if use_rl=True)
            }
        """
        apply_gate = use_gate if use_gate is not None else self.use_gate
        
        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Compute gate scores
        gate_scores = self.sentiment_gate(hidden_states)  # [batch, seq_len, 1]
        
        # Pooling
        if apply_gate:
            # Weighted pooling using gate scores
            # Mask out padding tokens
            if attention_mask is not None:
                gate_scores = gate_scores * attention_mask.unsqueeze(-1)
            
            # Weighted sum
            weighted_hidden = (hidden_states * gate_scores).sum(dim=1)  # [batch, hidden_size]
            
            # Normalize by sum of gate scores
            gate_sum = gate_scores.sum(dim=1) + 1e-9  # [batch, 1]
            pooled = weighted_hidden / gate_sum  # [batch, hidden_size]
        else:
            # Mean pooling (for ablation)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                pooled = sum_hidden / (sum_mask + 1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        
        # Dropout
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled)  # [batch, num_labels]
        
        # Compute loss if labels provided
        loss = None
        rl_loss = None
        rewards = None
        
        if labels is not None:
            # Standard cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits, labels)
            
            if use_rl and apply_gate:
                # RL: Policy gradient + reward-weighted regression
                
                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                
                # Compute rewards: +1 for correct, -1 for incorrect
                correct = (preds == labels).float()
                rewards = correct * 2.0 - 1.0  # [batch]
                
                # Policy gradient loss (REINFORCE)
                # Gate scores are our policy probabilities
                gate_probs = gate_scores.squeeze(-1)  # [batch, seq_len]
                
                # Mask out padding
                if attention_mask is not None:
                    gate_probs = gate_probs * attention_mask
                
                # Log probabilities
                log_probs = torch.log(gate_probs + 1e-9)  # [batch, seq_len]
                
                # Policy gradient: maximize reward by adjusting gate
                # Higher rewards → reinforce current gate pattern
                # Lower rewards → discourage current gate pattern
                pg_loss = -(log_probs * rewards.unsqueeze(-1)).sum(dim=1).mean()
                
                # Reward-weighted regression: scale CE loss by reward
                # Good predictions get lower loss weight (already doing well)
                # Bad predictions get higher loss weight (need improvement)
                reward_weights = 1.0 - correct * 0.5  # 0.5 for correct, 1.0 for incorrect
                weighted_ce_loss = (ce_loss * reward_weights.mean())
                
                # Combined RL loss
                rl_loss = pg_loss
                
                # Total loss: supervised + RL
                loss = weighted_ce_loss + rl_weight * rl_loss
            else:
                loss = ce_loss
        
        # Prepare output
        output = {
            'logits': logits,
        }
        
        if loss is not None:
            output['loss'] = loss
        
        if rl_loss is not None:
            output['rl_loss'] = rl_loss
            output['rewards'] = rewards
        
        if return_gate_scores:
            output['gate_scores'] = gate_scores.squeeze(-1)  # [batch, seq_len]
        
        return output
    
    def get_gate_scores(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        Get gate scores for visualization.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            gate_scores: [batch, seq_len]
        """
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = encoder_outputs.last_hidden_state
            gate_scores = self.sentiment_gate(hidden_states)
            return gate_scores.squeeze(-1)
