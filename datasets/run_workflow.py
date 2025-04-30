#!/usr/bin/env python
"""
Run the complete AI model training workflow.
This script executes all steps of the workflow in sequence.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
import yaml
import torch
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("workflow.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_notebook(notebook_path, output_path=None):
    """
    Run a Jupyter notebook and save the output.
    
    Args:
        notebook_path: Path to the notebook
        output_path: Path to save the output notebook
    
    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '_output.ipynb')
    
    logger.info(f"Running notebook: {notebook_path}")
    
    try:
        # Execute notebook
        result = subprocess.run(
            [
                "jupyter", "nbconvert", 
                "--to", "notebook", 
                "--execute",
                "--ExecutePreprocessor.timeout=3600",
                "--output", output_path,
                notebook_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Notebook executed successfully: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing notebook {notebook_path}: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def run_workflow(args):
    """
    Run the complete workflow.
    
    Args:
        args: Command-line arguments
    
    Returns:
        True if successful, False otherwise
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting workflow run: {output_dir}")
    
    # Define notebooks to run
    notebooks = [
        "notebooks/01_dataset_preparation.ipynb",
        "notebooks/02_model_training.ipynb",
        "notebooks/03_evaluation.ipynb",
        "notebooks/04_deployment.ipynb"
    ]
    
    # Run each notebook in sequence
    success = True
    start_time = time.time()
    
    for i, notebook in enumerate(notebooks):
        if args.skip_to and i < args.skip_to - 1:
            logger.info(f"Skipping notebook {i+1}: {notebook}")
            continue
        
        if args.stop_after and i > args.stop_after - 1:
            logger.info(f"Stopping after notebook {args.stop_after}")
            break
        
        notebook_start_time = time.time()
        notebook_output = os.path.join(output_dir, os.path.basename(notebook).replace('.ipynb', f'_output.ipynb'))
        
        logger.info(f"Running notebook {i+1}/{len(notebooks)}: {notebook}")
        notebook_success = run_notebook(notebook, notebook_output)
        
        if not notebook_success:
            logger.error(f"Notebook {notebook} failed. Stopping workflow.")
            success = False
            break
        
        notebook_end_time = time.time()
        logger.info(f"Notebook {i+1} completed in {notebook_end_time - notebook_start_time:.2f} seconds")
    
    # Calculate total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    if success:
        logger.info(f"Workflow completed successfully in {total_runtime:.2f} seconds")
    else:
        logger.error(f"Workflow failed after {total_runtime:.2f} seconds")
    
    return success

def load_config(config_path):
    """Load the configuration file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(output_dir):
    """Set up the output directories."""
    logger.info(f"Setting up output directories in {output_dir}")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "data",
        "checkpoints",
        "tokenizer",
        "logs",
        "evaluation",
        "flask_deployment",
        "coreml_models"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    logger.info(f"Created {len(subdirs)} output subdirectories")

def run_programmatic_workflow(args):
    """
    Run the workflow programmatically (without notebooks).
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up output directories
        setup_directories(args.output_dir)
        
        # Import workflow modules
        from src.data.loaders import DatasetLoader
        from src.data.preprocessors import TextPreprocessor
        from src.model.architecture import TransformerModel
        from src.model.training import train
        from src.evaluation.evaluator import ModelEvaluator
        from src.deployment.flask_export import FlaskExporter
        from src.deployment.coreml_export import CoreMLExporter
        
        # Run data preparation
        if not args.skip_data:
            logger.info("Starting data preparation")
            # Implementation of data preparation steps
            # ...
        
        # Run model training
        if not args.skip_train:
            logger.info("Starting model training")
            # Implementation of model training steps
            # ...
        
        # Run model evaluation
        if not args.skip_eval:
            logger.info("Starting model evaluation")
            # Implementation of model evaluation steps
            # ...
        
        # Run model deployment
        if not args.skip_deploy:
            logger.info("Starting model deployment")
            # Implementation of model deployment steps
            # ...
        
        logger.info("Programmatic workflow completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in programmatic workflow: {e}")
        return False

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the AI model training workflow")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save workflow outputs")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--skip-to", type=int, default=None,
                        help="Skip to a specific notebook (1-indexed)")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Stop after a specific notebook (1-indexed)")
    parser.add_argument("--use-notebooks", action="store_true",
                        help="Use Jupyter notebooks for workflow execution")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip the data preparation steps")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip the training steps")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the evaluation steps")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="Skip the deployment steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run workflow
    if args.use_notebooks:
        success = run_workflow(args)
    else:
        success = run_programmatic_workflow(args)
    
    sys.exit(0 if success else 1)