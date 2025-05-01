#!/usr/bin/env python
"""
Run AutoTrain fine-tuning workflow.
This script automates the process of fine-tuning a language model using AutoTrain.
"""

import os
import sys
import logging
import argparse
import yaml
import torch
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autotrain.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_directories(output_dir):
    """Set up the output directories."""
    logger.info(f"Setting up output directories in {output_dir}")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "data",
        "checkpoints",
        "logs",
        "models"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    logger.info(f"Created {len(subdirs)} output subdirectories")
    
    return os.path.join(output_dir, "data")

def prepare_dataset(args, config, data_dir):
    """Prepare the dataset for fine-tuning."""
    logger.info("Preparing dataset for fine-tuning")
    
    from datasets import load_dataset
    from src.data.autotrain_preprocessor import AutoTrainPreprocessor
    
    # Initialize preprocessor
    preprocessor = AutoTrainPreprocessor(config, output_dir=data_dir)
    
    # Load dataset
    if args.dataset_path:
        # Load from local path
        if args.dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=args.dataset_path, split='train')
        elif args.dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=args.dataset_path, split='train')
        elif args.dataset_path.endswith('.parquet'):
            dataset = load_dataset('parquet', data_files=args.dataset_path, split='train')
        else:
            # Try to load as a directory
            dataset = load_dataset(args.dataset_path, split='train')
    else:
        # Load from Hugging Face
        dataset_name = args.dataset_name or config['datasets']['core_datasets'][0]['name']
        dataset = load_dataset(dataset_name, split='train')
    
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    
    # Take a subset if specified
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
        logger.info(f"Selected {args.max_samples} examples from dataset")
    
    # Prepare dataset
    dataset_path = preprocessor.prepare_dataset(
        dataset, 
        format_type=args.dataset_format,
        output_file="train.csv"
    )
    
    logger.info(f"Dataset prepared and saved to {dataset_path}")
    return dataset_path

def run_autotrain(args, config, dataset_path):
    """Run AutoTrain for fine-tuning."""
    logger.info("Starting AutoTrain fine-tuning")
    
    import subprocess
    
    # Set up model parameters
    model_name = args.model_name or "mistralai/Mistral-7B-v0.1"
    project_name = args.project_name or f"finetuned-{model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d')}"
    
    # Set up repository ID
    if args.repo_id:
        repo_id = args.repo_id
    else:
        # Try to get username from Hugging Face CLI
        try:
            result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True)
            username = result.stdout.strip()
            if username:
                repo_id = f"{username}/{project_name}"
            else:
                repo_id = f"local/{project_name}"
        except:
            repo_id = f"local/{project_name}"
    
    logger.info(f"Using model: {model_name}")
    logger.info(f"Project name: {project_name}")
    logger.info(f"Repository ID: {repo_id}")
    
    # Build AutoTrain command
    cmd = [
        "autotrain", "llm",
        "--train",
        f"--project_name", project_name,
        f"--model", model_name,
        f"--data_path", os.path.dirname(dataset_path),
        "--text-column", "prompt",
        f"--learning_rate", str(args.learning_rate),
        f"--train_batch_size", str(args.batch_size),
        f"--num_train_epochs", str(args.epochs),
        "--trainer", "sft",
        "--use_peft",
    ]
    
    # Add quantization options
    if args.quantization == "int8":
        cmd.extend(["--use_int8"])
    elif args.quantization == "int4":
        cmd.extend(["--use_int4"])
    
    # Add precision options
    if args.fp16:
        cmd.extend(["--fp16"])
    
    # Add LoRA parameters
    cmd.extend([
        f"--lora-r", str(args.lora_r),
        f"--lora-alpha", str(args.lora_alpha),
        f"--lora-dropout", str(args.lora_dropout),
    ])
    
    # Add target modules
    if args.target_modules:
        cmd.extend(["--target-modules", args.target_modules])
    
    # Add push to hub if specified
    if args.push_to_hub:
        cmd.extend([
            "--push_to_hub",
            f"--repo_id", repo_id,
        ])
        
        # Add token if provided
        if args.token:
            cmd.extend([f"--token", args.token])
    
    # Run AutoTrain
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("AutoTrain completed successfully")
        logger.info(result.stdout)
    else:
        logger.error("AutoTrain failed")
        logger.error(result.stderr)
        raise RuntimeError("AutoTrain failed")
    
    return project_name, repo_id

def save_model_metadata(args, config, project_name, repo_id, output_dir):
    """Save metadata about the fine-tuned model."""
    logger.info("Saving model metadata")
    
    metadata = {
        "model_name": project_name,
        "base_model": args.model_name,
        "adapter_model": repo_id,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_parameters": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules.split(",") if args.target_modules else [],
            "quantization": args.quantization
        },
        "dataset": {
            "name": args.dataset_name or args.dataset_path,
            "format": args.dataset_format,
            "max_samples": args.max_samples
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "models", f"{project_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved model metadata to {metadata_path}")
    return metadata_path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run AutoTrain fine-tuning workflow")
    
    # General arguments
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save workflow outputs")
    
    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Name of the dataset on Hugging Face")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to local dataset file or directory")
    parser.add_argument("--dataset-format", type=str, default=None,
                        choices=["instruction", "chat", "completion"],
                        help="Format of the dataset")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to use from the dataset")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default=None,
                        help="Name of the base model on Hugging Face")
    parser.add_argument("--project-name", type=str, default=None,
                        help="Name of the fine-tuned model project")
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--quantization", type=str, default="int4",
                        choices=["none", "int8", "int4"],
                        help="Quantization method to use")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--target-modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated list of target modules for LoRA")
    
    # Hugging Face arguments
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push the model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="Repository ID for Hugging Face Hub")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    return parser.parse_args()

def main():
    """Run the AutoTrain workflow."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # Set up directories
        data_dir = setup_directories(args.output_dir)
        
        # Prepare dataset
        dataset_path = prepare_dataset(args, config, data_dir)
        
        # Run AutoTrain
        project_name, repo_id = run_autotrain(args, config, dataset_path)
        
        # Save model metadata
        metadata_path = save_model_metadata(args, config, project_name, repo_id, args.output_dir)
        
        logger.info(f"AutoTrain workflow completed successfully")
        logger.info(f"Model: {repo_id}")
        logger.info(f"Metadata: {metadata_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in AutoTrain workflow: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())