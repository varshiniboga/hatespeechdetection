"""
Unit tests for model functionality.
"""

import unittest
import torch
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import HateSpeechModel, RegressionTrainer
from transformers import EvalPrediction
import numpy as np


class TestHateSpeechModel(unittest.TestCase):
    """Test cases for HateSpeechModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = HateSpeechModel()
    
    def test_initialization(self):
        """Test proper initialization of model."""
        self.assertEqual(self.model.model_name, 'roberta-base')
        self.assertEqual(self.model.num_labels, 10)
        self.assertIsNone(self.model.model)
        self.assertEqual(len(self.model.target_labels), 10)
    
    def test_target_labels(self):
        """Test that all expected target labels are present."""
        expected_labels = [
            'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize',
            'violence', 'genocide', 'attack_defend', 'hatespeech'
        ]
        self.assertEqual(self.model.target_labels, expected_labels)
    
    @patch('src.model.RobertaForSequenceClassification')
    def test_load_model(self, mock_roberta):
        """Test model loading."""
        mock_model = Mock()
        mock_roberta.from_pretrained.return_value = mock_model
        
        result = self.model.load_model()
        
        mock_roberta.from_pretrained.assert_called_once_with(
            'roberta-base',
            num_labels=10,
            problem_type="multi_label_classification"
        )
        self.assertEqual(result, mock_model)
        self.assertEqual(self.model.model, mock_model)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Create mock predictions
        predictions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        labels = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
        
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        
        metrics = self.model.compute_metrics(eval_pred)
        
        # Check that metrics are computed for all labels
        for label in self.model.target_labels:
            self.assertIn(f'mse_{label}', metrics)
            self.assertIn(f'mae_{label}', metrics)
        
        # Check that metrics are numeric
        for key, value in metrics.items():
            self.assertIsInstance(value, (int, float, np.number))


class TestRegressionTrainer(unittest.TestCase):
    """Test cases for RegressionTrainer class."""
    
    def test_compute_loss(self):
        """Test custom loss computation."""
        # Create a mock trainer
        trainer = RegressionTrainer()
        
        # Create mock model
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        mock_model.return_value = mock_outputs
        
        # Create mock inputs
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'labels': torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
        }
        
        # Test loss computation
        loss = trainer.compute_loss(mock_model, inputs)
        
        # Check that loss is a tensor
        self.assertIsInstance(loss, torch.Tensor)
        
        # Check that labels were removed from inputs
        self.assertNotIn('labels', inputs)
        
        # Test with return_outputs=True
        inputs['labels'] = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
        loss, outputs = trainer.compute_loss(mock_model, inputs, return_outputs=True)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(outputs, mock_outputs)


if __name__ == '__main__':
    unittest.main()
