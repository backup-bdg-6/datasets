"""
Tests for the model architecture and training.
"""

import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.model.architecture import TransformerModel
from src.model.training import train_step, evaluate

class TestModelArchitecture(unittest.TestCase):
    """Test cases for the model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 10000
        self.hidden_size = 768
        self.num_hidden_layers = 2
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.initializer_range = 0.02
        
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range
        )
        
        # Set model to evaluation mode for deterministic behavior
        self.model.eval()
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, TransformerModel)
        
        # Check that the model has the expected number of parameters
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(param_count, 1000000)  # Should have at least 1M parameters
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        batch_size = 2
        seq_length = 10
        
        # Create random input
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, seq_length, self.vocab_size))
    
    def test_generate_method(self):
        """Test the generate method of the model."""
        batch_size = 2
        seq_length = 10
        max_new_tokens = 5
        
        # Create random input
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Run generation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Check output shape
        self.assertEqual(generated_ids.shape, (batch_size, seq_length + max_new_tokens))
        
        # Check that the first seq_length tokens are the same as the input
        self.assertTrue(torch.all(generated_ids[:, :seq_length] == input_ids))
    
    def test_save_and_load(self):
        """Test saving and loading the model."""
        # Create a temporary directory for the test
        os.makedirs("temp", exist_ok=True)
        checkpoint_path = "temp/model_test.pt"
        
        # Save the model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size
        }, checkpoint_path)
        
        # Create a new model
        new_model = TransformerModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range
        )
        
        # Load the model
        checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Set to evaluation mode
        new_model.eval()
        
        # Check that the models have the same parameters
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.all(p1 == p2))
        
        # Clean up
        os.remove(checkpoint_path)
        os.rmdir("temp")

class TestTraining(unittest.TestCase):
    """Test cases for the training process."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 10000
        self.hidden_size = 768
        self.num_hidden_layers = 2
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.initializer_range = 0.02
        
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range
        )
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Create dummy batch
        batch_size = 2
        seq_length = 10
        self.batch = {
            "input_ids": torch.randint(0, self.vocab_size, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "labels": torch.randint(0, self.vocab_size, (batch_size, seq_length))
        }
    
    def test_train_step(self):
        """Test the training step."""
        # Set model to training mode
        self.model.train()
        
        # Run training step
        loss = train_step(self.model, self.batch, self.optimizer)
        
        # Check that loss is a scalar
        self.assertIsInstance(loss, float)
        
        # Check that loss is positive
        self.assertGreater(loss, 0)
    
    def test_evaluate(self):
        """Test the evaluation function."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dummy evaluation data
        eval_data = [self.batch for _ in range(5)]
        
        # Run evaluation
        metrics = evaluate(self.model, eval_data)
        
        # Check that metrics contains loss
        self.assertIn("loss", metrics)
        
        # Check that loss is a scalar
        self.assertIsInstance(metrics["loss"], float)
        
        # Check that loss is positive
        self.assertGreater(metrics["loss"], 0)

if __name__ == "__main__":
    unittest.main()