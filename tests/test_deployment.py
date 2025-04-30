"""
Tests for the deployment modules.
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
from unittest.mock import patch, MagicMock

# Add the project root to the path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.deployment.flask_export import FlaskExporter
from src.deployment.coreml_export import CoreMLExporter
from src.deployment.coreml_utils import prepare_model_for_coreml, create_example_inputs

class TestFlaskExporter(unittest.TestCase):
    """Test cases for the Flask exporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 128)
                self.linear = torch.nn.Linear(128, 1000)
            
            def forward(self, input_ids, attention_mask=None):
                embeddings = self.embedding(input_ids)
                return self.linear(embeddings)
            
            def generate(self, input_ids, attention_mask=None, max_new_tokens=10, **kwargs):
                batch_size, seq_length = input_ids.shape
                return torch.randint(0, 1000, (batch_size, seq_length + max_new_tokens))
        
        self.model = MockModel()
        
        # Create a mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3]
            
            def decode(self, ids):
                return "decoded text"
        
        self.tokenizer = MockTokenizer()
        
        # Create a simple config
        self.config = {
            "model": {
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 512
            },
            "deployment": {
                "flask": {
                    "host": "0.0.0.0",
                    "port": 5000,
                    "batch_size": 8,
                    "max_cache_size": 1000
                }
            }
        }
        
        # Create exporter
        self.exporter = FlaskExporter(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            output_dir=self.temp_dir.name
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the exporter initializes correctly."""
        self.assertIsInstance(self.exporter, FlaskExporter)
        self.assertEqual(self.exporter.model, self.model)
        self.assertEqual(self.exporter.tokenizer, self.tokenizer)
        self.assertEqual(self.exporter.config, self.config)
        self.assertEqual(self.exporter.output_dir, self.temp_dir.name)
    
    @patch("torch.save")
    def test_export_model(self, mock_save):
        """Test exporting the model."""
        # Export the model
        self.exporter.export_model()
        
        # Check that torch.save was called
        mock_save.assert_called_once()
    
    @patch("json.dump")
    def test_export_config(self, mock_dump):
        """Test exporting the configuration."""
        # Export the configuration
        self.exporter.export_config()
        
        # Check that json.dump was called
        mock_dump.assert_called_once()
    
    @patch("torch.save")
    @patch("json.dump")
    def test_export(self, mock_dump, mock_save):
        """Test the full export process."""
        # Export
        result = self.exporter.export()
        
        # Check that torch.save and json.dump were called
        mock_save.assert_called_once()
        mock_dump.assert_called_once()
        
        # Check the result
        self.assertIsInstance(result, dict)
        self.assertIn("model_path", result)
        self.assertIn("config_path", result)
        self.assertIn("tokenizer_path", result)

class TestCoreMLExporter(unittest.TestCase):
    """Test cases for the CoreML exporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 128)
                self.linear = torch.nn.Linear(128, 1000)
            
            def forward(self, input_ids, attention_mask=None):
                embeddings = self.embedding(input_ids)
                return self.linear(embeddings)
            
            def generate(self, input_ids, attention_mask=None, max_new_tokens=10, **kwargs):
                batch_size, seq_length = input_ids.shape
                return torch.randint(0, 1000, (batch_size, seq_length + max_new_tokens))
        
        self.model = MockModel()
        
        # Create a mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3]
            
            def decode(self, ids):
                return "decoded text"
        
        self.tokenizer = MockTokenizer()
        
        # Create a simple config
        self.config = {
            "model": {
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 512
            },
            "deployment": {
                "coreml": {
                    "model_name": "transformer_model",
                    "optimize_for_mobile": True,
                    "quantize": True,
                    "prune": True
                }
            }
        }
        
        # Create exporter
        self.exporter = CoreMLExporter(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            model_name="transformer_model",
            output_dir=self.temp_dir.name
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the exporter initializes correctly."""
        self.assertIsInstance(self.exporter, CoreMLExporter)
        self.assertEqual(self.exporter.model, self.model)
        self.assertEqual(self.exporter.tokenizer, self.tokenizer)
        self.assertEqual(self.exporter.config, self.config)
        self.assertEqual(self.exporter.model_name, "transformer_model")
        self.assertEqual(self.exporter.output_dir, self.temp_dir.name)
    
    def test_prepare_model(self):
        """Test preparing the model for CoreML conversion."""
        # Prepare the model
        prepared_model = self.exporter.prepare_model()
        
        # Check that the model is in evaluation mode
        self.assertFalse(prepared_model.training)
    
    @patch("torch.jit.trace")
    def test_trace_model(self, mock_trace):
        """Test tracing the model."""
        # Mock the trace result
        mock_trace.return_value = MagicMock()
        
        # Trace the model
        input_shapes = {"input_ids": [1, 10], "attention_mask": [1, 10]}
        traced_model = self.exporter.trace_model(self.model, input_shapes)
        
        # Check that torch.jit.trace was called
        mock_trace.assert_called_once()
    
    @patch("torch.jit.trace")
    @patch("coremltools.convert")
    def test_export_mock(self, mock_convert, mock_trace):
        """Test the full export process with mocks."""
        # Mock the trace and convert results
        mock_trace.return_value = MagicMock()
        mock_convert.return_value = MagicMock()
        mock_convert.return_value.save = MagicMock()
        
        # Export
        input_shapes = {"input_ids": [1, 10], "attention_mask": [1, 10]}
        with patch("src.deployment.coreml_export.coremltools", MagicMock()):
            result = self.exporter.export(
                pytorch_model=mock_trace.return_value,
                input_shapes=input_shapes,
                conversion_method="trace"
            )
        
        # Check that the necessary methods were called
        mock_trace.assert_called_once()
        mock_convert.assert_called_once()
        mock_convert.return_value.save.assert_called_once()

class TestCoreMLUtils(unittest.TestCase):
    """Test cases for the CoreML utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 128)
                self.linear = torch.nn.Linear(128, 1000)
            
            def forward(self, input_ids, attention_mask=None):
                embeddings = self.embedding(input_ids)
                return self.linear(embeddings)
        
        self.model = MockModel()
        
        # Create a simple config
        self.config = {
            "optimize_for_mobile": True,
            "quantize": True,
            "prune": True
        }
    
    def test_prepare_model_for_coreml(self):
        """Test preparing a model for CoreML conversion."""
        # Prepare the model
        prepared_model = prepare_model_for_coreml(self.model, self.config)
        
        # Check that the model is in evaluation mode
        self.assertFalse(prepared_model.training)
    
    def test_create_example_inputs(self):
        """Test creating example inputs for tracing/exporting."""
        # Create example inputs
        input_shapes = {"input_ids": [1, 10], "attention_mask": [1, 10]}
        example_inputs = create_example_inputs(input_shapes)
        
        # Check that the inputs have the expected shapes
        self.assertEqual(len(example_inputs), 2)
        self.assertEqual(example_inputs[0].shape, torch.Size([1, 10]))
        self.assertEqual(example_inputs[1].shape, torch.Size([1, 10]))

if __name__ == "__main__":
    unittest.main()