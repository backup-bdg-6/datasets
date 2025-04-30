"""
Flask export utilities for the AI model training workflow.
This module provides functions to export models for use in Flask applications.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

# Configure logging
logger = logging.getLogger(__name__)

class FlaskModelExporter:
    """
    Class for exporting models for use in Flask applications.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
        model_name: str,
        config: Dict
    ):
        """
        Initialize the Flask model exporter.
        
        Args:
            model: Model to export
            tokenizer: Tokenizer for the model
            output_dir: Directory to save exported model
            model_name: Name of the model
            config: Export configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.model_name = model_name
        self.config = config
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def export(self) -> str:
        """
        Export the model for use in Flask applications.
        
        Returns:
            Path to the exported model directory
        """
        # Create model directory
        model_dir = os.path.join(self.output_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Exporting model to {model_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save Flask-specific configuration
        flask_config = {
            "model_name": self.model_name,
            "batch_size": self.config['flask']['batch_size'],
            "max_concurrent_requests": self.config['flask']['max_concurrent_requests'],
            "timeout": self.config['flask']['timeout'],
            "cache_size": self.config['flask']['cache_size'],
            "async_mode": self.config['flask']['async_mode']
        }
        
        with open(os.path.join(model_dir, "flask_config.json"), 'w') as f:
            json.dump(flask_config, f, indent=2)
        
        # Create example Flask application
        self._create_example_flask_app(model_dir)
        
        logger.info(f"Model exported successfully to {model_dir}")
        
        return model_dir
    
    def _create_example_flask_app(self, model_dir: str) -> None:
        """
        Create an example Flask application for the exported model.
        
        Args:
            model_dir: Directory containing the exported model
        """
        app_file = os.path.join(model_dir, "app.py")
        
        app_code = f'''"""
Flask application for serving the {self.model_name} model.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS
import asyncio
import functools
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Flask configuration
with open(os.path.join(os.path.dirname(__file__), "flask_config.json"), 'r') as f:
    flask_config = json.load(f)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Initialize model and tokenizer
model = None
tokenizer = None

# Initialize response cache
response_cache = {{}}

# Initialize thread pool for async processing
thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=flask_config.get("max_concurrent_requests", 10)
)

def load_model():
    """
    Load the model and tokenizer.
    """
    global model, tokenizer
    
    logger.info("Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        os.path.dirname(__file__),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    logger.info("Model and tokenizer loaded successfully")

def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 50) -> str:
    """
    Generate text based on the prompt.
    
    Args:
        prompt: Input prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        
    Returns:
        Generated text
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

def get_cached_response(prompt: str, params: Dict) -> Optional[Dict]:
    """
    Get cached response for the prompt and parameters.
    
    Args:
        prompt: Input prompt
        params: Generation parameters
        
    Returns:
        Cached response or None if not found
    """
    # Create cache key
    cache_key = f"{{prompt}}|{{json.dumps(params, sort_keys=True)}}"
    
    # Check if response is in cache
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    return None

def add_to_cache(prompt: str, params: Dict, response: Dict) -> None:
    """
    Add response to cache.
    
    Args:
        prompt: Input prompt
        params: Generation parameters
        response: Response to cache
    """
    # Create cache key
    cache_key = f"{{prompt}}|{{json.dumps(params, sort_keys=True)}}"
    
    # Add to cache
    response_cache[cache_key] = response
    
    # Limit cache size
    if len(response_cache) > flask_config.get("cache_size", 100):
        # Remove oldest entry
        oldest_key = next(iter(response_cache))
        del response_cache[oldest_key]

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate text based on the prompt.
    """
    # Get request data
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({{"error": "No prompt provided"}}), 400
    
    # Extract parameters
    prompt = data['prompt']
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    
    # Check if response is in cache
    params = {{"max_length": max_length, "temperature": temperature, "top_p": top_p, "top_k": top_k}}
    cached_response = get_cached_response(prompt, params)
    
    if cached_response:
        return jsonify(cached_response)
    
    # Generate text
    try:
        start_time = time.time()
        generated_text = generate_text(prompt, max_length, temperature, top_p, top_k)
        end_time = time.time()
        
        # Create response
        response = {{
            "prompt": prompt,
            "generated_text": generated_text,
            "parameters": params,
            "generation_time": end_time - start_time
        }}
        
        # Add to cache
        add_to_cache(prompt, params, response)
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error generating text: {{e}}")
        return jsonify({{"error": str(e)}}), 500

@app.route('/generate_async', methods=['POST'])
async def generate_async():
    """
    Generate text asynchronously based on the prompt.
    """
    if not flask_config.get("async_mode", False):
        return jsonify({{"error": "Async mode not enabled"}}), 400
    
    # Get request data
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({{"error": "No prompt provided"}}), 400
    
    # Extract parameters
    prompt = data['prompt']
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    
    # Check if response is in cache
    params = {{"max_length": max_length, "temperature": temperature, "top_p": top_p, "top_k": top_k}}
    cached_response = get_cached_response(prompt, params)
    
    if cached_response:
        return jsonify(cached_response)
    
    # Generate text asynchronously
    try:
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        # Run generation in thread pool
        generated_text = await loop.run_in_executor(
            thread_pool,
            functools.partial(generate_text, prompt, max_length, temperature, top_p, top_k)
        )
        
        end_time = time.time()
        
        # Create response
        response = {{
            "prompt": prompt,
            "generated_text": generated_text,
            "parameters": params,
            "generation_time": end_time - start_time
        }}
        
        # Add to cache
        add_to_cache(prompt, params, response)
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error generating text asynchronously: {{e}}")
        return jsonify({{"error": str(e)}}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    if model is None or tokenizer is None:
        return jsonify({{"status": "not_ready"}}), 503
    
    return jsonify({{"status": "ready"}})

@app.route('/info', methods=['GET'])
def info():
    """
    Model information endpoint.
    """
    if model is None or tokenizer is None:
        return jsonify({{"error": "Model not loaded"}}), 503
    
    return jsonify({{
        "model_name": flask_config.get("model_name", "unknown"),
        "parameters": model.config.to_dict(),
        "tokenizer": {{
            "vocab_size": len(tokenizer),
            "model_max_length": tokenizer.model_max_length
        }}
    }})

@app.before_first_request
def before_first_request():
    """
    Load model before first request.
    """
    load_model()

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Run Flask application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
'''
        
        with open(app_file, 'w') as f:
            f.write(app_code)
        
        # Create requirements file
        requirements_file = os.path.join(model_dir, "requirements.txt")
        
        requirements = '''flask==2.0.1
flask-cors==3.0.10
transformers==4.30.2
torch>=1.13.0
numpy>=1.20.0
'''
        
        with open(requirements_file, 'w') as f:
            f.write(requirements)
        
        # Create README file
        readme_file = os.path.join(model_dir, "README.md")
        
        readme = f'''# {self.model_name} Flask Application

This directory contains a Flask application for serving the {self.model_name} model.

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the Flask application:

```bash
python app.py
```

The application will be available at http://localhost:5000.

## API Endpoints

### Generate Text

```
POST /generate
```

Request body:

```json
{{
  "prompt": "Once upon a time",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}}
```

Response:

```json
{{
  "prompt": "Once upon a time",
  "generated_text": "Once upon a time there was a kingdom...",
  "parameters": {{
    "max_length": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
  }},
  "generation_time": 1.23
}}
```

### Generate Text Asynchronously

```
POST /generate_async
```

Same request and response format as `/generate`, but processed asynchronously.

### Health Check

```
GET /health
```

Response:

```json
{{
  "status": "ready"
}}
```

### Model Information

```
GET /info
```

Response:

```json
{{
  "model_name": "{self.model_name}",
  "parameters": {{
    // Model configuration
  }},
  "tokenizer": {{
    "vocab_size": 50257,
    "model_max_length": 1024
  }}
}}
```

## Configuration

The application is configured using the `flask_config.json` file:

```json
{{
  "model_name": "{self.model_name}",
  "batch_size": {self.config['flask']['batch_size']},
  "max_concurrent_requests": {self.config['flask']['max_concurrent_requests']},
  "timeout": {self.config['flask']['timeout']},
  "cache_size": {self.config['flask']['cache_size']},
  "async_mode": {str(self.config['flask']['async_mode']).lower()}
}}
```

## Deployment

For production deployment, it is recommended to use a WSGI server such as Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
'''
        
        with open(readme_file, 'w') as f:
            f.write(readme)
        
        logger.info(f"Created example Flask application in {model_dir}")


# Example usage
if __name__ == "__main__":
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create exporter
    exporter = FlaskModelExporter(
        model=model,
        tokenizer=tokenizer,
        output_dir="outputs/deployment",
        model_name="gpt2-flask",
        config=config
    )
    
    # Export model
    export_dir = exporter.export()
    
    logger.info(f"Model exported to {export_dir}")