"""
Unit tests for data loader functionality.
"""

import unittest
import torch
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import HateSpeechDataLoader


class TestHateSpeechDataLoader(unittest.TestCase):
    """Test cases for HateSpeechDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = HateSpeechDataLoader()
    
    def test_initialization(self):
        """Test proper initialization of data loader."""
        self.assertEqual(self.data_loader.dataset_name, "ucberkeley-dlab/measuring-hate-speech")
        self.assertEqual(len(self.data_loader.target_labels), 10)
        self.assertIsNone(self.data_loader.tokenizer)
    
    def test_target_labels(self):
        """Test that all expected target labels are present."""
        expected_labels = [
            'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize',
            'violence', 'genocide', 'attack_defend', 'hatespeech'
        ]
        self.assertEqual(self.data_loader.target_labels, expected_labels)
    
    @patch('src.data_loader.RobertaTokenizerFast')
    def test_initialize_tokenizer(self, mock_tokenizer):
        """Test tokenizer initialization."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        self.data_loader.initialize_tokenizer('roberta-base')
        
        mock_tokenizer.from_pretrained.assert_called_once_with('roberta-base')
        self.assertIsNotNone(self.data_loader.tokenizer)
    
    def test_preprocess_function_without_tokenizer(self):
        """Test that preprocess function raises error without tokenizer."""
        examples = {'text': ['test text']}
        
        with self.assertRaises(ValueError):
            self.data_loader.preprocess_function(examples)
    
    def test_preprocess_function_with_tokenizer(self):
        """Test preprocess function with mock tokenizer."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3]],
            'attention_mask': [[1, 1, 1]]
        }
        self.data_loader.tokenizer = mock_tokenizer
        
        # Create mock examples
        examples = {
            'text': ['test text'],
            'sentiment': [0.5],
            'respect': [0.3],
            'insult': [0.7],
            'humiliate': [0.2],
            'status': [0.4],
            'dehumanize': [0.1],
            'violence': [0.8],
            'genocide': [0.0],
            'attack_defend': [0.6],
            'hatespeech': [0.9]
        }
        
        result = self.data_loader.preprocess_function(examples)
        
        # Check that tokenizer was called
        mock_tokenizer.assert_called_once()
        
        # Check that labels tensor is created
        self.assertIn('labels', result)
        self.assertIsInstance(result['labels'], torch.Tensor)
        self.assertEqual(result['labels'].shape, (1, 10))  # 1 example, 10 labels


if __name__ == '__main__':
    unittest.main()
