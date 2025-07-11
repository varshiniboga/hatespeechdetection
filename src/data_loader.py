"""
Data loading and preprocessing utilities for hate speech detection.
"""

import pandas as pd
import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast
from typing import Dict, List, Any


class HateSpeechDataLoader:
    """Class to handle data loading and preprocessing for hate speech detection."""
    
    def __init__(self, dataset_name: str = "ucberkeley-dlab/measuring-hate-speech"):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset to load from Hugging Face
        """
        self.dataset_name = dataset_name
        self.target_labels = [
            'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize',
            'violence', 'genocide', 'attack_defend', 'hatespeech'
        ]
        self.tokenizer = None
        
    def load_dataset(self, sample_size: float = 0.1, train_split: float = 0.8, seed: int = 42):
        """
        Load and split the dataset.
        
        Args:
            sample_size: Fraction of the full dataset to use
            train_split: Fraction of sampled data to use for training
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        print("Loading dataset from Hugging Face...")
        full_dataset = load_dataset(self.dataset_name)
        
        # Sample dataset
        sampled_dataset = full_dataset['train'].train_test_split(
            train_size=sample_size, seed=seed
        )['train']
        
        # Split into train and test
        train_test_split = sampled_dataset.train_test_split(
            train_size=train_split, seed=seed
        )
        
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        print(f"Full dataset size: {len(full_dataset['train'])}")
        print(f"Sampled dataset size: {len(sampled_dataset)}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def initialize_tokenizer(self, model_name: str = 'roberta-base'):
        """Initialize the tokenizer."""
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess function for tokenizing text and preparing labels.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Tokenized batch with labels
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call initialize_tokenizer() first.")
            
        # Tokenize the text
        tokenized_batch = self.tokenizer(
            examples['text'], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        
        # Create a 'labels' tensor from the target columns
        labels = [examples[label] for label in self.target_labels]
        
        # Transpose and convert to tensor
        tokenized_batch['labels'] = torch.tensor(
            [list(row) for row in zip(*labels)], 
            dtype=torch.float
        )
        
        return tokenized_batch
    
    def get_dataframes(self, train_dataset, test_dataset) -> tuple:
        """Convert datasets to pandas DataFrames for analysis."""
        train_df = pd.DataFrame(train_dataset)
        test_df = pd.DataFrame(test_dataset)
        return train_df, test_df
