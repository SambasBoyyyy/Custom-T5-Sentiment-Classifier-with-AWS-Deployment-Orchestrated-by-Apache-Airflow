"""
Train T5 with Sentiment Gate on SST-2 (Classification Head)

Simplified training with classification head instead of decoder.

Usage:
    # Quick test
    python train.py --max_steps 10 --output_dir ./test_run
    
    # Full training
    python train.py --epochs 3 --batch_size 32 --output_dir ./t5-classification
"""

import argparse
import logging
import torch
from transformers import T5Tokenizer

from modules.models.t5_sentiment_gate import T5ForSentimentClassification
from modules.data.sst2_dataset import prepare_sst2_data, create_dataloaders
from modules.training.config import TrainingConfig
from modules.training.trainer import SentimentGateTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train T5 Classification with Sentiment Gate")
    
    # Model
    parser.add_argument('--model_name', type=str, default='t5-small',
                       help='Base T5 model name')
    parser.add_argument('--no-gate', action='store_true',
                       help='Disable sentiment gate (for ablation)')
    
    # Data
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Max sequence length')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Max training steps (overrides epochs)')
    parser.add_argument('--encoder_lr', type=float, default=1e-4,
                       help='Encoder learning rate')
    parser.add_argument('--gate_lr', type=float, default=3e-4,
                       help='Gate learning rate')
    
    # RL Settings
    parser.add_argument('--use-rl', action='store_true',
                       help='Enable RL training (policy gradient)')
    parser.add_argument('--rl-start-epoch', type=int, default=2,
                       help='Start RL after this many epochs')
    parser.add_argument('--rl-weight', type=float, default=0.1,
                       help='Weight for RL loss (0.1 = 10%% RL)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./t5-classification',
                       help='Output directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("T5 CLASSIFICATION WITH SENTIMENT GATE")
    logger.info("="*70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Gate: {'DISABLED' if args.no_gate else 'ENABLED'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    if args.max_steps:
        logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*70)
    
    # Load tokenizer
    logger.info("\n1. Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, legacy=False)
    
    # Prepare data
    logger.info("\n2. Preparing SST-2 dataset...")
    dataset_dict = prepare_sst2_data()
    
    # Create dataloaders
    logger.info("\n3. Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Load model
    logger.info("\n4. Loading model...")
    model = T5ForSentimentClassification.from_pretrained(args.model_name, num_labels=2)
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        use_gate=not args.no_gate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        encoder_lr=args.encoder_lr,
        gate_lr=args.gate_lr,
        output_dir=args.output_dir,
        device=args.device,
        use_rl=args.use_rl,
        rl_start_epoch=args.rl_start_epoch,
        rl_weight=args.rl_weight
    )
    
    # Create trainer
    logger.info("\n5. Initializing trainer...")
    trainer = SentimentGateTrainer(model, tokenizer, config)
    
    # Train
    logger.info("\n6. Starting training...")
    stats = trainer.train(train_loader, val_loader)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Best validation accuracy: {stats['best_val_acc']:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
