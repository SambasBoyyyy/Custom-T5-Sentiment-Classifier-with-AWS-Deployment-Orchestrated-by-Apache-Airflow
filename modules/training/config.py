"""
Training Configuration for T5 Sentiment Gate

Hyperparameters optimized for SST-2 classification with sentiment gate.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for T5 with sentiment gate.
    
    Key settings:
    - Gate learning rate is 3x higher than encoder/decoder
    - Gradient clipping for stability
    - Small batch size for memory efficiency
    """
    
    # Model
    model_name: str = 't5-small'
    use_gate: bool = True
    
    # Data
    max_length: int = 128
    batch_size: int = 16
    num_workers: int = 0
    
    # Training
    num_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides num_epochs
    
    # Learning rates (gate gets 5x higher LR)
    encoder_lr: float = 1e-4
    decoder_lr: float = 1e-4
    gate_lr: float = 3e-4  # 3x higher for faster gate learning
    
    # Optimization
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0  # Important for gate stability
    warmup_steps: int = 500
    
    # Logging & Checkpointing
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2
    
    # Device
    device: str = 'cuda'  # Will auto-detect if CUDA available
    fp16: bool = True  # Mixed precision training
    
    # Output
    output_dir: str = './t5-sentiment-gate'
    
    # RL Settings
    use_rl: bool = False  # Enable RL training
    rl_start_epoch: int = 2  # Start RL after this many epochs of supervised training
    rl_weight: float = 0.1  # Weight for RL loss (0.1 = 10% RL, 90% supervised)
    reward_scale: float = 1.0  # Scale factor for rewards
    
    def __post_init__(self):
        """Validate and adjust config after initialization."""
        import torch
        
        # Auto-detect device
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            self.fp16 = False  # Disable FP16 on CPU
            print("CUDA not available, using CPU")
        
        # Disable FP16 if on CPU
        if self.device == 'cpu':
            self.fp16 = False
        
        # Validate RL settings
        if self.use_rl and self.rl_start_epoch >= self.num_epochs:
            print(f"Warning: rl_start_epoch ({self.rl_start_epoch}) >= num_epochs ({self.num_epochs})")
            print("RL will never activate. Consider reducing rl_start_epoch or increasing num_epochs.")
