"""
SST-2 Dataset Preprocessing for Classification Head

Simplified preprocessing for classification (no decoder needed):
- Input: tokenized sentence
- Label: class index (0 or 1)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SST2ClassificationDataset(Dataset):
    """
    SST-2 dataset for classification (no decoder).
    
    Format:
        Input: "The movie was not very good"
        Label: 0 (negative) or 1 (positive)
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: T5Tokenizer,
        max_length: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            {
                'input_ids': [seq_len],
                'attention_mask': [seq_len],
                'labels': scalar (0 or 1)
            }
        """
        example = self.data[idx]
        
        # Tokenize input sentence
        encoding = self.tokenizer(
            example['sentence'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Label is just the class index
        label = torch.tensor(example['label'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }


def prepare_sst2_data() -> Dict:
    """
    Load SST-2 dataset.
    
    Returns:
        dataset: Dict with 'train' and 'validation' splits
    """
    logger.info("Loading SST-2 dataset...")
    dataset = load_dataset('glue', 'sst2')
    
    # Filter out test set (has -1 labels)
    dataset_dict = {
        'train': [{'sentence': ex['sentence'], 'label': ex['label']} 
                  for ex in dataset['train']],
        'validation': [{'sentence': ex['sentence'], 'label': ex['label']} 
                       for ex in dataset['validation']],
    }
    
    logger.info(f"Loaded {len(dataset_dict['train'])} train examples, "
                f"{len(dataset_dict['validation'])} validation examples")
    
    return dataset_dict


def create_dataloaders(
    dataset_dict: Dict,
    tokenizer: T5Tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_dict: Dict with 'train' and 'validation' data
        tokenizer: T5 tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Create datasets
    train_dataset = SST2ClassificationDataset(
        data=dataset_dict['train'],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = SST2ClassificationDataset(
        data=dataset_dict['validation'],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches")
    
    return train_loader, val_loader
