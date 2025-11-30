"""
Evaluation and Visualization for T5 with Sentiment Gate

Provides:
1. Accuracy evaluation
2. Gate score visualization
3. Ablation study support
"""

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
from pathlib import Path

from ..models.t5_sentiment_gate import T5WithSentimentGate

logger = logging.getLogger(__name__)


class SentimentGateEvaluator:
    """
    Evaluator for T5 with sentiment gate.
    
    Features:
    - Accuracy computation
    - Gate score visualization
    - Ablation studies (with/without gate)
    - Sample prediction analysis
    
    Usage:
        evaluator = SentimentGateEvaluator(model, tokenizer, device='cuda')
        accuracy = evaluator.evaluate(val_loader, pos_id, neg_id)
        evaluator.visualize_gate_scores(sample_texts)
    """
    
    def __init__(
        self,
        model: T5WithSentimentGate,
        tokenizer: T5Tokenizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        pos_token_id: int,
        neg_token_id: int,
        use_gate: bool = True
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: Validation dataloader
            pos_token_id: Token ID for <positive>
            neg_token_id: Token ID for <negative>
            use_gate: If True, use gate; if False, disable for ablation
        
        Returns:
            results: Dict with accuracy, predictions, etc.
        """
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        labels_list = []
        
        for batch in tqdm(dataloader, desc=f"Evaluating (gate={'ON' if use_gate else 'OFF'})"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Generate predictions
            outputs = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=2,  # start + 1 token
                num_beams=1,
                do_sample=False,
                use_gate=use_gate
            )
            
            # Extract predictions
            preds = outputs[:, 1]  # Skip start token
            labels = batch['labels'][:, 0]
            
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            predictions.extend(preds.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())
        
        accuracy = correct / total
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'use_gate': use_gate,
            'predictions': predictions,
            'labels': labels_list
        }
        
        logger.info(f"Accuracy (gate={'ON' if use_gate else 'OFF'}): {accuracy:.4f} ({correct}/{total})")
        
        return results
    
    @torch.no_grad()
    def visualize_gate_scores(
        self,
        texts: List[str],
        true_labels: Optional[List[int]] = None,
        max_examples: int = 10
    ) -> List[Dict]:
        """
        Visualize gate scores for sample texts.
        
        Args:
            texts: List of input sentences
            true_labels: Optional true labels (0=negative, 1=positive)
            max_examples: Maximum number of examples to visualize
        
        Returns:
            visualizations: List of dicts with text, tokens, gate scores, prediction
        
        Example output:
            {
                'text': 'The movie was not very good',
                'tokens': ['The', 'movie', 'was', 'not', 'very', 'good'],
                'gate_scores': [0.12, 0.45, 0.08, 0.92, 0.78, 0.67],
                'prediction': 'negative',
                'true_label': 'negative',
                'correct': True
            }
        """
        self.model.eval()
        visualizations = []
        
        for idx, text in enumerate(texts[:max_examples]):
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get encoder outputs
            encoder_outputs = self.model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            hidden_states = encoder_outputs.last_hidden_state
            
            # Get gate scores
            gate_scores = self.model.get_gate_scores(hidden_states)
            gate_scores = gate_scores[0].cpu().tolist()  # First example
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Filter out padding
            valid_length = inputs['attention_mask'][0].sum().item()
            tokens = tokens[:valid_length]
            gate_scores = gate_scores[:valid_length]
            
            # Get prediction
            decoder_input_ids = torch.full(
                (1, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
            
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=2,
                num_beams=1,
                do_sample=False
            )
            
            pred_token_id = outputs[0, 1].item()
            pred_token = self.tokenizer.decode([pred_token_id])
            
            # Build visualization
            viz = {
                'text': text,
                'tokens': tokens,
                'gate_scores': gate_scores,
                'prediction': pred_token
            }
            
            if true_labels is not None and idx < len(true_labels):
                label_map = {0: 'negative', 1: 'positive'}
                true_label = label_map[true_labels[idx]]
                viz['true_label'] = true_label
                viz['correct'] = (pred_token == true_label)
            
            visualizations.append(viz)
        
        return visualizations
    
    def print_gate_visualization(self, visualizations: List[Dict]):
        """
        Pretty print gate visualizations.
        
        Example output:
            Input: "The movie was not very good"
            
            Gate scores:
              The:    0.12  ← ignored
              movie:  0.45
              was:    0.08  ← ignored
              not:    0.92  ← KEY!
              very:   0.78  ← KEY!
              good:   0.67
            
            Prediction: negative ✓
        """
        print("\n" + "="*70)
        print("GATE SCORE VISUALIZATION")
        print("="*70)
        
        for viz in visualizations:
            print(f"\nInput: \"{viz['text']}\"")
            print("\nGate scores:")
            
            for token, score in zip(viz['tokens'], viz['gate_scores']):
                # Clean token
                token_str = token.replace('▁', '').replace('</s>', '')
                if not token_str:
                    continue
                
                # Format score
                marker = ""
                if score > 0.7:
                    marker = " ← KEY!"
                elif score < 0.3:
                    marker = " ← ignored"
                
                print(f"  {token_str:15s} {score:.2f}{marker}")
            
            # Prediction
            status = ""
            if 'correct' in viz:
                status = " ✓" if viz['correct'] else " ✗"
            
            print(f"\nPrediction: {viz['prediction']}{status}")
            
            if 'true_label' in viz:
                print(f"True label: {viz['true_label']}")
            
            print("-" * 70)
    
    def ablation_study(
        self,
        dataloader: DataLoader,
        pos_token_id: int,
        neg_token_id: int
    ) -> Dict:
        """
        Run ablation study: compare with/without gate.
        
        Returns:
            results: Dict with accuracies for both settings
        """
        logger.info("\n" + "="*70)
        logger.info("ABLATION STUDY")
        logger.info("="*70)
        
        # Evaluate with gate
        results_with_gate = self.evaluate(
            dataloader, pos_token_id, neg_token_id, use_gate=True
        )
        
        # Evaluate without gate
        results_without_gate = self.evaluate(
            dataloader, pos_token_id, neg_token_id, use_gate=False
        )
        
        # Summary
        improvement = results_with_gate['accuracy'] - results_without_gate['accuracy']
        
        summary = {
            'without_gate': results_without_gate['accuracy'],
            'with_gate': results_with_gate['accuracy'],
            'improvement': improvement,
            'improvement_pct': improvement * 100
        }
        
        logger.info(f"\nResults:")
        logger.info(f"  Without gate: {summary['without_gate']:.4f}")
        logger.info(f"  With gate:    {summary['with_gate']:.4f}")
        logger.info(f"  Improvement:  +{summary['improvement']:.4f} ({summary['improvement_pct']:.2f}%)")
        
        return summary
