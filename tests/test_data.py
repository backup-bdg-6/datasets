"""
Tests for the data loading and preprocessing modules.
"""

import os
import sys
import unittest
import tempfile
import json
import yaml
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.data.loaders import DatasetLoader
from src.data.preprocessors import TextPreprocessor

class TestDatasetLoader(unittest.TestCase):
    """Test cases for the dataset loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.yaml")
        
        # Create a simple config
        self.config = {
            "datasets": {
                "openai_humaneval": {
                    "name": "openai/humaneval",
                    "split": "test",
                    "max_samples": 10
                },
                "bigcode_the_stack": {
                    "name": "bigcode/the-stack",
                    "subset": "data",
                    "split": "train",
                    "max_samples": 10,
                    "languages": ["python"]
                }
            },
            "preprocessing": {
                "max_length": 512,
                "tokenizer": {
                    "type": "bpe",
                    "vocab_size": 50000
                }
            }
        }
        
        # Write config to file
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
        
        # Create dataset loader
        self.loader = DatasetLoader(self.config_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the loader initializes correctly."""
        self.assertIsInstance(self.loader, DatasetLoader)
        self.assertEqual(self.loader.config, self.config)
    
    def test_get_dataset_config(self):
        """Test getting dataset configuration."""
        config = self.loader.get_dataset_config("openai_humaneval")
        self.assertEqual(config["name"], "openai/humaneval")
        self.assertEqual(config["split"], "test")
        self.assertEqual(config["max_samples"], 10)
    
    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = self.loader.list_datasets()
        self.assertIn("openai_humaneval", datasets)
        self.assertIn("bigcode_the_stack", datasets)
    
    def test_load_dataset_mock(self):
        """Test loading a dataset with a mock."""
        # Create a mock dataset
        class MockDataset:
            def __init__(self, data):
                self.data = data
            
            def __getitem__(self, idx):
                return self.data[idx]
            
            def __len__(self):
                return len(self.data)
        
        # Mock the load_dataset method
        original_load = self.loader.load_dataset
        self.loader.load_dataset = lambda name: MockDataset([{"code": f"def f{i}(): pass"} for i in range(10)])
        
        # Load the dataset
        dataset = self.loader.load_dataset("openai_humaneval")
        
        # Check that the dataset has the expected items
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset[0]["code"], "def f0(): pass")
        
        # Restore the original method
        self.loader.load_dataset = original_load

class TestTextPreprocessor(unittest.TestCase):
    """Test cases for the text preprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple tokenizer config
        self.tokenizer_config = {
            "type": "bpe",
            "vocab_size": 1000,
            "special_tokens": {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]"
            }
        }
        
        # Create preprocessor
        self.preprocessor = TextPreprocessor(
            max_length=128,
            tokenizer_config=self.tokenizer_config,
            tokenizer_path=self.temp_dir.name
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the preprocessor initializes correctly."""
        self.assertIsInstance(self.preprocessor, TextPreprocessor)
        self.assertEqual(self.preprocessor.max_length, 128)
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "Hello,  world!  \n\n  How are you?"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "Hello, world! How are you?")
    
    def test_train_tokenizer_mock(self):
        """Test training a tokenizer with a mock."""
        # Create a mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                self.special_tokens = {
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                    "bos_token": "[BOS]",
                    "eos_token": "[EOS]"
                }
            
            def train(self, texts):
                pass
            
            def save(self, path):
                with open(os.path.join(path, "tokenizer.json"), "w") as f:
                    json.dump({"vocab_size": self.vocab_size}, f)
            
            def encode(self, text):
                return [1, 2, 3]
            
            def decode(self, ids):
                return "decoded text"
        
        # Mock the create_tokenizer method
        original_create = self.preprocessor.create_tokenizer
        self.preprocessor.create_tokenizer = lambda config: MockTokenizer()
        
        # Train the tokenizer
        texts = ["Hello, world!", "How are you?", "I'm fine, thank you."]
        self.preprocessor.train_tokenizer(texts)
        
        # Check that the tokenizer was saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "tokenizer.json")))
        
        # Restore the original method
        self.preprocessor.create_tokenizer = original_create
    
    def test_preprocess_text_mock(self):
        """Test text preprocessing with a mock tokenizer."""
        # Create a mock tokenizer
        class MockTokenizer:
            def encode(self, text, max_length=None, padding=None, truncation=None):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        
        # Set the mock tokenizer
        self.preprocessor.tokenizer = MockTokenizer()
        
        # Preprocess text
        result = self.preprocessor.preprocess_text("Hello, world!")
        
        # Check the result
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(result["input_ids"], [1, 2, 3])
        self.assertEqual(result["attention_mask"], [1, 1, 1])

if __name__ == "__main__":
    unittest.main()