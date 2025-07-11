"""
Configuration settings for hate speech detection training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Model settings
    model_name: str = 'roberta-base'
    num_labels: int = 10
    max_length: int = 128
    
    # Data settings
    dataset_name: str = "ucberkeley-dlab/measuring-hate-speech"
    sample_size: float = 0.1
    train_split: float = 0.8
    seed: int = 42
    
    # Training parameters
    output_dir: str = './results'
    logging_dir: str = './logs'
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 100
    warmup_steps: int = 100
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 128
    use_fp16: bool = True
    load_best_model_at_end: bool = True
    report_to: str = "none"
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Target labels
    target_labels: list = None
    
    def __post_init__(self):
        if self.target_labels is None:
            self.target_labels = [
                'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize',
                'violence', 'genocide', 'attack_defend', 'hatespeech'
            ]


@dataclass
class PredictionConfig:
    """Configuration class for prediction parameters."""
    
    model_path: str = "./results"
    max_examples_to_show: int = 20
    output_file: Optional[str] = None
