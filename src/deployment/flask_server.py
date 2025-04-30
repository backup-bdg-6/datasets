"""
Flask server for model deployment.
This module provides a production-ready Flask server for deploying the trained model.
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import queue
from typing import Dict, Any, List, Optional, Union, Tuple

import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flask_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import project modules
from src.model.architecture import TransformerModel
from src.utils.tokenization import Tokenizer

class ModelServer:
    """
    Server for model inference with batching and caching capabilities.
    """
    
    def __init__(
        self, 
        model_path: str,
        tokenizer_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
        max_cache_size: int = 1000,
        max_sequence_length: int = 1024
    ):
        """
        Initialize the model server.
        
        Args:
            model_path: Path to the model checkpoint
            tokenizer_path: Path to the tokenizer
            config_path: Path to the configuration file
            device: Device to run the model on
            batch_size: Maximum batch size for inference
            max_cache_size: Maximum number of items to cache
            max_sequence_length: Maximum sequence length for input
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_cache_size = max_cache_size
        self.max_sequence_length = max_sequence_length
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Initialize cache
        self.cache = {}
        
        # Initialize request queue and worker thread
        self.request_queue = queue.Queue()
        self.response_queues = {}
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"Model server initialized on {device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load the model from checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Loaded model
        """
        # Load model configuration
        model_config = self.config['model']
        
        # Initialize model
        model = TransformerModel(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            num_hidden_layers=model_config['num_hidden_layers'],
            num_attention_heads=model_config['num_attention_heads'],
            intermediate_size=model_config['intermediate_size'],
            hidden_dropout_prob=0.0,  # Set to 0 for inference
            attention_probs_dropout_prob=0.0,  # Set to 0 for inference
            max_position_embeddings=model_config['max_position_embeddings'],
            initializer_range=model_config['initializer_range']
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def _worker_loop(self):
        """
        Worker loop for processing batched requests.
        """
        batch = []
        batch_ids = []
        
        while True:
            try:
                # Get request from queue
                request_id, request_data = self.request_queue.get(timeout=0.1)
                batch.append(request_data)
                batch_ids.append(request_id)
                
                # Process batch if it's full or if the queue is empty
                if len(batch) >= self.batch_size or self.request_queue.empty():
                    # Process batch
                    results = self._process_batch(batch)
                    
                    # Send results to response queues
                    for i, result in enumerate(results):
                        self.response_queues[batch_ids[i]].put(result)
                    
                    # Clear batch
                    batch = []
                    batch_ids = []
            
            except queue.Empty:
                # Process any remaining items in the batch
                if batch:
                    results = self._process_batch(batch)
                    
                    for i, result in enumerate(results):
                        self.response_queues[batch_ids[i]].put(result)
                    
                    batch = []
                    batch_ids = []
            
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                
                # Send error to all response queues in the batch
                for request_id in batch_ids:
                    self.response_queues[request_id].put({"error": str(e)})
                
                batch = []
                batch_ids = []
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of requests.
        
        Args:
            batch: List of request data
            
        Returns:
            List of results
        """
        # Extract inputs from batch
        input_texts = [item['text'] for item in batch]
        max_new_tokens = [item.get('max_new_tokens', 50) for item in batch]
        temperature = [item.get('temperature', 1.0) for item in batch]
        
        # Check cache for each input
        results = []
        uncached_indices = []
        uncached_inputs = []
        uncached_max_tokens = []
        uncached_temps = []
        
        for i, text in enumerate(input_texts):
            cache_key = (text, max_new_tokens[i], temperature[i])
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_inputs.append(text)
                uncached_max_tokens.append(max_new_tokens[i])
                uncached_temps.append(temperature[i])
        
        # Process uncached inputs
        if uncached_inputs:
            uncached_results = self._generate_text(
                uncached_inputs, 
                uncached_max_tokens,
                uncached_temps
            )
            
            # Update results and cache
            for i, result in zip(uncached_indices, uncached_results):
                cache_key = (input_texts[i], max_new_tokens[i], temperature[i])
                self.cache[cache_key] = result
                results[i] = result
            
            # Trim cache if it's too large
            if len(self.cache) > self.max_cache_size:
                # Remove oldest items
                keys_to_remove = list(self.cache.keys())[:(len(self.cache) - self.max_cache_size)]
                for key in keys_to_remove:
                    del self.cache[key]
        
        return results
    
    def _generate_text(
        self, 
        texts: List[str], 
        max_new_tokens: List[int],
        temperatures: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Generate text using the model.
        
        Args:
            texts: List of input texts
            max_new_tokens: List of maximum new tokens to generate
            temperatures: List of temperature values for sampling
            
        Returns:
            List of generated texts and metadata
        """
        try:
            # Tokenize inputs
            encoded_inputs = [self.tokenizer.encode(text) for text in texts]
            
            # Truncate if necessary
            encoded_inputs = [
                tokens[:self.max_sequence_length] for tokens in encoded_inputs
            ]
            
            # Create attention masks
            attention_masks = [
                [1] * len(tokens) for tokens in encoded_inputs
            ]
            
            # Pad inputs
            max_length = max(len(tokens) for tokens in encoded_inputs)
            padded_inputs = [
                tokens + [0] * (max_length - len(tokens)) for tokens in encoded_inputs
            ]
            padded_masks = [
                mask + [0] * (max_length - len(mask)) for mask in attention_masks
            ]
            
            # Convert to tensors
            input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(padded_masks, dtype=torch.long, device=self.device)
            
            # Generate text
            with torch.no_grad():
                start_time = time.time()
                
                # Generate tokens
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max(max_new_tokens),
                    temperature=temperatures[0],  # Use the first temperature for now
                    do_sample=any(t > 0 for t in temperatures),
                    pad_token_id=0,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                end_time = time.time()
            
            # Decode outputs
            generated_texts = []
            for i, output in enumerate(outputs):
                # Remove input tokens
                new_tokens = output[len(padded_inputs[i]):]
                
                # Truncate to max_new_tokens
                new_tokens = new_tokens[:max_new_tokens[i]]
                
                # Decode
                generated_text = self.tokenizer.decode(new_tokens.tolist())
                
                # Create result
                result = {
                    "generated_text": generated_text,
                    "input_text": texts[i],
                    "generation_time": end_time - start_time,
                    "num_tokens": len(new_tokens)
                }
                
                generated_texts.append(result)
            
            return generated_texts
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return [{"error": str(e)} for _ in texts]
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request asynchronously.
        
        Args:
            request_data: Request data
            
        Returns:
            Response data
        """
        # Create a response queue
        response_queue = queue.Queue()
        request_id = id(response_queue)
        self.response_queues[request_id] = response_queue
        
        # Add request to queue
        self.request_queue.put((request_id, request_data))
        
        # Wait for response
        response = response_queue.get()
        
        # Clean up
        del self.response_queues[request_id]
        
        return response

# Create Flask app
app = Flask(__name__)
CORS(app)

# Global model server instance
model_server = None

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate text from input.
    """
    try:
        # Get request data
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        # Process request
        result = model_server.process_request(data)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    return jsonify({"status": "healthy"})

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flask server for model deployment")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to the tokenizer")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Maximum batch size for inference")
    parser.add_argument("--max-cache-size", type=int, default=1000,
                        help="Maximum number of items to cache")
    parser.add_argument("--max-sequence-length", type=int, default=1024,
                        help="Maximum sequence length for input")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    global model_server
    
    # Initialize model server
    model_server = ModelServer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
        device=args.device,
        batch_size=args.batch_size,
        max_cache_size=args.max_cache_size,
        max_sequence_length=args.max_sequence_length
    )
    
    # Set up WSGI server
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    # Run Flask app
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()