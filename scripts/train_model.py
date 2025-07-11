#!/usr/bin/env python3
"""
Main training script for hate speech detection model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from transformers import TrainingArguments
from src.data_loader import HateSpeechDataLoader
from src.model import HateSpeechModel, RegressionTrainer
from src.config import TrainingConfig
from src.utils import plot_label_distributions, save_model_and_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train hate speech detection model')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum training steps')
    parser.add_argument('--sample_size', type=float, default=0.1, help='Dataset sample size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = TrainingConfig()
    config.output_dir = args.output_dir
    config.max_steps = args.max_steps
    config.sample_size = args.sample_size
    config.per_device_train_batch_size = args.batch_size
    config.per_device_eval_batch_size = args.batch_size
    
    print("Starting hate speech detection training...")
    print(f"Configuration: {config}")
    
    # Initialize data loader
    data_loader = HateSpeechDataLoader(config.dataset_name)
    
    # Load and split dataset
    train_dataset, test_dataset = data_loader.load_dataset(
        sample_size=config.sample_size,
        train_split=config.train_split,
        seed=config.seed
    )
    
    # Convert to DataFrames for analysis
    train_df, test_df = data_loader.get_dataframes(train_dataset, test_dataset)
    
    # Plot label distributions
    print("Plotting label distributions...")
    plot_label_distributions(train_df, config.target_labels)
    
    # Initialize tokenizer and preprocess data
    print("Preprocessing data...")
    data_loader.initialize_tokenizer(config.model_name)
    
    # Create a temporary dataset split for preprocessing
    temp_split = {'train': train_dataset, 'test': test_dataset}
    processed_dataset = {}
    
    for split_name, dataset in temp_split.items():
        processed_dataset[split_name] = dataset.map(
            data_loader.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    train_dataset_processed = processed_dataset['train']
    eval_dataset_processed = processed_dataset['test']
    
    # Initialize model
    print("Loading model...")
    hate_speech_model = HateSpeechModel(config.model_name, config.num_labels)
    model = hate_speech_model.load_model()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=config.logging_dir,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        report_to=config.report_to,
        fp16=config.use_fp16,
        load_best_model_at_end=config.load_best_model_at_end
    )
    
    # Initialize trainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        compute_metrics=hate_speech_model.compute_metrics,
        tokenizer=data_loader.tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    save_model_and_results(trainer, config.output_dir)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
