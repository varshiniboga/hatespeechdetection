"""
Model definitions and custom trainer for hate speech detection.
"""

import torch
from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List


class RegressionTrainer(Trainer):
    """Custom Trainer for Multi-output Regression using MSE Loss."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override the compute_loss method to use MSELoss instead of default classification loss.
        
        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return outputs along with loss
            
        Returns:
            Loss value or tuple of (loss, outputs)
        """
        # Extract the 'labels' tensor from inputs dictionary and remove it from inputs
        labels = inputs.pop("labels")
        
        # Forward pass: compute model outputs (logits)
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, num_labels)
        
        # Define Mean Squared Error loss (suitable for regression tasks)
        loss_fct = torch.nn.MSELoss()
        
        # Compute loss between predicted logits and true labels
        loss = loss_fct(logits, labels.float())  # Ensure labels are float type for regression
        
        # Return the loss and outputs if return_outputs=True (needed for eval/prediction)
        return (loss, outputs) if return_outputs else loss


class HateSpeechModel:
    """Wrapper class for the hate speech detection model."""
    
    def __init__(self, model_name: str = 'roberta-base', num_labels: int = 10):
        """
        Initialize the model.
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of target labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.target_labels = [
            'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize',
            'violence', 'genocide', 'attack_defend', 'hatespeech'
        ]
        
    def load_model(self):
        """Load the pre-trained model."""
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        return self.model
    
    def compute_metrics(self, p: EvalPrediction):
        """
        Compute evaluation metrics.
        
        Args:
            p: EvalPrediction object containing predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        # Extract the predictions; handle case where predictions come as a tuple
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        
        # Compute Mean Squared Error (MSE) for each target label
        mse = mean_squared_error(p.label_ids, preds, multioutput='raw_values')
        
        # Compute Mean Absolute Error (MAE) for each target label
        mae = mean_absolute_error(p.label_ids, preds, multioutput='raw_values')
        
        # Create a dictionary mapping label names to their MSE scores
        metrics = {f'mse_{label}': value for label, value in zip(self.target_labels, mse)}
        
        # Add MAE scores to the dictionary using the same label names
        metrics.update({f'mae_{label}': value for label, value in zip(self.target_labels, mae)})
        
        # Return a combined dictionary of metrics (MSE + MAE per label)
        return metrics
