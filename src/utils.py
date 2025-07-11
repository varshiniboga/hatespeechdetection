"""
Utility functions for hate speech detection.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any


def plot_label_distributions(df: pd.DataFrame, target_labels: List[str], figsize: tuple = (30, 12)):
    """
    Plot the distribution of target labels.
    
    Args:
        df: DataFrame containing the data
        target_labels: List of target label column names
        figsize: Figure size for the plot
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()
    
    for i, label in enumerate(target_labels):
        sns.countplot(x=label, data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {label}')
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()


def create_results_dataframe(eval_dataset, test_dataset, predicted_values, actual_labels, 
                           target_labels: List[str], num_examples: int = 20) -> pd.DataFrame:
    """
    Create a DataFrame with predictions and actual values for analysis.
    
    Args:
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset (original)
        predicted_values: Model predictions
        actual_labels: Actual labels
        target_labels: List of target label names
        num_examples: Number of examples to include
        
    Returns:
        DataFrame with results
    """
    rows = []
    
    for i in range(min(num_examples, len(eval_dataset))):
        row = {"Text": test_dataset[i]['text']}
        for j, label in enumerate(target_labels):
            row[f"Actual_{label}"] = actual_labels[i][j]
            row[f"Predicted_{label}"] = predicted_values[i][j]
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_model_and_results(trainer, save_path: str, results_df: pd.DataFrame = None):
    """
    Save the trained model and optionally the results.
    
    Args:
        trainer: Trained model trainer
        save_path: Path to save the model
        results_df: Optional results DataFrame to save
    """
    # Save the trained model
    trainer.save_model(save_path)
    print(f"Model saved to: {save_path}")
    
    # Save results if provided
    if results_df is not None:
        results_path = f"{save_path}/results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")


def print_dataset_info(full_size: int, sampled_size: int, train_size: int, test_size: int):
    """Print dataset size information."""
    print(f"Full dataset size: {full_size}")
    print(f"Sampled dataset size: {sampled_size}")
    print(f"Train dataset size: {train_size}")
    print(f"Test dataset size: {test_size}")
