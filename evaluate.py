"""
Evaluate T5 Classification with Sentiment Gate

Simplified evaluation for classification head.

Usage:
    # Standard evaluation
    python evaluate.py --model_path ./t5-classification/best_model
    
    # With ablation study
    python evaluate.py --model_path ./t5-classification/best_model --ablation
    
    # With gate visualization
    python evaluate.py --model_path ./t5-classification/best_model --visualize
"""

import argparse
import logging
import torch
import json
from transformers import T5Tokenizer
from pathlib import Path

from modules.models.t5_sentiment_gate import T5ForSentimentClassification
from modules.data.sst2_dataset import prepare_sst2_data, create_dataloaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, use_gate=True):
    """Evaluate model on dataloader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                use_gate=use_gate
            )
            
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            
            correct += (preds == labels).sum().item()
            total += len(labels)
    
    accuracy = correct / total
    return accuracy


def visualize_gate_scores(model, tokenizer, dataset_dict, device, num_samples=10):
    """Visualize gate scores for sample texts"""
    model.eval()
    
    print("\n" + "="*70)
    print("GATE SCORE VISUALIZATION")
    print("="*70)
    
    label_map = {0: "negative", 1: "positive"}
    
    for i, example in enumerate(dataset_dict['validation'][:num_samples]):
        text = example['sentence']
        true_label = label_map[example['label']]
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get gate scores and prediction
        with torch.no_grad():
            gate_scores = model.get_gate_scores(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            pred_idx = torch.argmax(outputs['logits'], dim=-1).item()
            pred_label = label_map[pred_idx]
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        scores = gate_scores[0].cpu().tolist()
        
        # Filter out padding
        valid_length = inputs['attention_mask'][0].sum().item()
        tokens = tokens[:valid_length]
        scores = scores[:valid_length]
        
        print(f"\nInput: \"{text}\"")
        print("\nGate scores:")
        for token, score in zip(tokens, scores):
            token_str = token.replace('▁', '').replace('</s>', '')
            if not token_str:
                continue
            
            marker = ""
            if score > 0.7:
                marker = " ← KEY!"
            elif score < 0.3:
                marker = " ← ignored"
            
            print(f"  {token_str:15s} {score:.2f}{marker}")
        
        status = "✓" if pred_label == true_label else "✗"
        print(f"\nPrediction: {pred_label} {status}")
        print(f"True label: {true_label}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate T5 Classification")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study (with/without gate)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize gate scores on sample texts')
    parser.add_argument('--num_viz_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*70)
    logger.info("T5 CLASSIFICATION EVALUATION")
    logger.info("="*70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {device}")
    logger.info("="*70)
    
    # Load tokenizer
    logger.info("\n1. Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_path, legacy=False)
    
    # Load model
    logger.info("\n2. Loading model...")
    model_path = Path(args.model_path)
    
    # Load T5 config
    from transformers import T5Config
    t5_config = T5Config.from_pretrained(model_path)
    
    # Load model info
    with open(model_path / 'model_info.json', 'r') as f:
        model_info = json.load(f)
    
    # Create model
    model = T5ForSentimentClassification(t5_config, num_labels=model_info['num_labels'])
    
    # Load state dict
    state_dict = torch.load(model_path / 'pytorch_model.bin', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    
    logger.info(f"Loaded model: {model_info['model_type']} with {model_info['num_labels']} labels")
    
    # Prepare data
    logger.info("\n3. Loading SST-2 dataset...")
    dataset_dict = prepare_sst2_data()
    
    # Create dataloaders
    logger.info("\n4. Creating dataloaders...")
    _, val_loader = create_dataloaders(
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=128
    )
    
    # Standard evaluation
    logger.info("\n5. Running evaluation...")
    accuracy = evaluate_model(model, val_loader, device, use_gate=True)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70)
    
    # Ablation study
    if args.ablation:
        logger.info("\n6. Running ablation study...")
        acc_with_gate = accuracy
        acc_without_gate = evaluate_model(model, val_loader, device, use_gate=False)
        
        improvement = acc_with_gate - acc_without_gate
        
        print("\n" + "="*70)
        print("ABLATION STUDY")
        print("="*70)
        print(f"Without gate: {acc_without_gate:.4f} ({acc_without_gate*100:.2f}%)")
        print(f"With gate:    {acc_with_gate:.4f} ({acc_with_gate*100:.2f}%)")
        print(f"Improvement:  +{improvement:.4f} ({improvement*100:.2f}%)")
        print("="*70)
    
    # Gate visualization
    if args.visualize:
        logger.info("\n7. Visualizing gate scores...")
        visualize_gate_scores(model, tokenizer, dataset_dict, device, args.num_viz_samples)


if __name__ == '__main__':
    main()
