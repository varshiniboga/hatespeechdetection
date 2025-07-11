#!/usr/bin/env python3
"""
Prediction script for hate speech detection model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from src.data_loader import HateSpeechDataLoader
from src.config import TrainingConfig, PredictionConfig
from src.utils import create_results_dataframe


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained hate speech detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_file', type=str, help='Output CSV file for results')
    parser.add_argument('--sample_size', type=float, default=0.01, help='Dataset sample size for prediction')
    parser.add_argument('--max_examples', type=int, default=20, help='Maximum examples to show')
    
    args = parser.parse_args()
    
    # Initialize configurations
    train_config = TrainingConfig()
    pred_config = PredictionConfig()
    pred_config.model_path = args.model_path
    pred_config.max_examples_to_show = args.max_examples
    pred_config.output_file = args.output_file
    
    print("Loading model and tokenizer...")
    
    # Load model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(pred_config.model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    print("Loading test data...")
    
    # Load data for prediction
    data_loader = HateSpeechDataLoader(train_config.dataset_name)
    data_loader.tokenizer = tokenizer
    
    # Load dataset (using smaller sample for prediction demo)
    train_dataset, test_dataset = data_loader.load_dataset(
        sample_size=args.sample_size,
        train_split=0.8,
        seed=42
    )
    
    # Preprocess test data
    temp_split = {'test': test_dataset}
    processed_dataset = temp_split['test'].map(
        data_loader.preprocess_function,
        batched=True,
        remove_columns=temp_split['test'].column_names
    )
    
    print("Making predictions...")
    
    # Make predictions
    model.eval()
    predictions = []
    actual_labels = []
    
    for i in range(min(pred_config.max_examples_to_show, len(processed_dataset))):
        inputs = {
            'input_ids': processed_dataset[i]['input_ids'].unsqueeze(0),
            'attention_mask': processed_dataset[i]['attention_mask'].unsqueeze(0)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions.append(outputs.logits.squeeze().numpy())
            actual_labels.append(processed_dataset[i]['labels'].numpy())
    
    # Create results DataFrame
    results_df = create_results_dataframe(
        processed_dataset, 
        test_dataset, 
        predictions, 
        actual_labels,
        train_config.target_labels,
        pred_config.max_examples_to_show
    )
    
    # Display results
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    print("\nPrediction Results:")
    print(results_df)
    
    # Save results if output file specified
    if pred_config.output_file:
        results_df.to_csv(pred_config.output_file, index=False)
        print(f"\nResults saved to: {pred_config.output_file}")
    
    print("Prediction completed successfully!")


if __name__ == "__main__":
    main()
