#!/usr/bin/env python3
"""
Evaluation script for hate speech detection model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_loader import HateSpeechDataLoader
from src.model import RegressionTrainer
from src.config import TrainingConfig


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained hate speech detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--sample_size', type=float, default=0.05, help='Dataset sample size for evaluation')
    parser.add_argument('--output_file', type=str, help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = TrainingConfig()
    
    print("Loading model and tokenizer...")
    
    # Load model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    print("Loading evaluation data...")
    
    # Load data
    data_loader = HateSpeechDataLoader(config.dataset_name)
    data_loader.tokenizer = tokenizer
    
    # Load dataset
    train_dataset, test_dataset = data_loader.load_dataset(
        sample_size=args.sample_size,
        train_split=0.8,
        seed=42
    )
    
    # Preprocess data
    temp_split = {'test': test_dataset}
    eval_dataset = temp_split['test'].map(
        data_loader.preprocess_function,
        batched=True,
        remove_columns=temp_split['test'].column_names
    )
    
    print("Evaluating model...")
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )
    
    # Make predictions
    predictions = trainer.predict(eval_dataset)
    predicted_values = predictions.predictions
    actual_labels = predictions.label_ids
    
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Overall MSE and MAE
    overall_mse = mean_squared_error(actual_labels, predicted_values)
    overall_mae = mean_absolute_error(actual_labels, predicted_values)
    
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    
    # Per-label metrics
    print("\nPer-label Metrics:")
    print("-" * 30)
    
    mse_per_label = mean_squared_error(actual_labels, predicted_values, multioutput='raw_values')
    mae_per_label = mean_absolute_error(actual_labels, predicted_values, multioutput='raw_values')
    
    results_data = []
    for i, label in enumerate(config.target_labels):
        mse_val = mse_per_label[i]
        mae_val = mae_per_label[i]
        print(f"{label:12} - MSE: {mse_val:.4f}, MAE: {mae_val:.4f}")
        results_data.append({
            'label': label,
            'mse': mse_val,
            'mae': mae_val
        })
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save detailed results if requested
    if args.output_file:
        results_df.to_csv(args.output_file, index=False)
        print(f"\nDetailed results saved to: {args.output_file}")
    
    # Sample predictions
    print("\nSample Predictions:")
    print("-" * 50)
    
    num_samples = min(5, len(eval_dataset))
    for i in range(num_samples):
        print(f"\nExample {i+1}:")
        print(f"Text: {test_dataset[i]['text'][:100]}...")
        print("Predictions vs Actual:")
        for j, label in enumerate(config.target_labels):
            pred_val = predicted_values[i][j]
            actual_val = actual_labels[i][j]
            print(f"  {label:12}: {pred_val:.3f} (actual: {actual_val:.3f})")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
