#!/usr/bin/env python
"""
Main script for training AI models using the enhanced training pipeline.
"""

import os
import sys
import logging
import argparse
import yaml
import torch
from typing import Dict, Optional, Union, List

from src.data.loaders import DatasetLoader
from src.data.preprocessors import DataPreprocessor
from src.data.augmentation import TextAugmenter, augment_dataset
from src.model.architecture import create_model_from_config
from src.model.training import Trainer, TrainingArguments
from src.model.distributed_training import train_distributed, DeepSpeedConfig
from src.utils.tokenization import get_tokenizer
from src.utils.hyperparameter_tuning import optimize_hyperparameters
from src.utils.model_evaluation import evaluate_model
from src.model.optimization import optimize_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AI models using the enhanced training pipeline.")
    
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        help="Training stage to run (overrides config)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed training"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--deepspeed", action="store_true",
        help="Enable DeepSpeed"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable mixed precision training"
    )
    parser.add_argument(
        "--hyperparameter_tuning", action="store_true",
        help="Run hyperparameter tuning"
    )
    parser.add_argument(
        "--evaluate_only", action="store_true",
        help="Only evaluate the model, don't train"
    )
    parser.add_argument(
        "--optimize_model", action="store_true",
        help="Optimize the model after training"
    )
    parser.add_argument(
        "--optimization_type", type=str, default="dynamic_quantization",
        choices=["dynamic_quantization", "static_quantization", "int8", "int4", "pruning", "onnx"],
        help="Type of model optimization to apply"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace API token for accessing gated datasets"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def update_config_with_args(config: Dict, args) -> Dict:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create a copy of the config
    updated_config = config.copy()
    
    # Update with command line arguments
    if args.stage:
        updated_config['training']['active_stage'] = args.stage
    
    if args.output_dir:
        updated_config['output_dir'] = args.output_dir
    
    if args.batch_size:
        updated_config['data_processing']['batching']['train_batch_size'] = args.batch_size
        updated_config['data_processing']['batching']['eval_batch_size'] = args.batch_size
    
    if args.epochs:
        # Update epochs for the active stage
        active_stage = updated_config['training']['active_stage']
        for stage in updated_config['training']['stages']:
            if stage['name'] == active_stage:
                stage['epochs'] = args.epochs
    
    if args.learning_rate:
        # Update learning rate for the active stage
        active_stage = updated_config['training']['active_stage']
        for stage in updated_config['training']['stages']:
            if stage['name'] == active_stage:
                stage['learning_rate']['initial'] = args.learning_rate
    
    if args.seed:
        updated_config['seed'] = args.seed
    
    if args.distributed:
        updated_config['distributed_training']['enabled'] = True
    
    if args.deepspeed:
        updated_config['distributed_training']['use_deepspeed'] = True
    
    if args.fp16:
        updated_config['training']['mixed_precision'] = "fp16"
    
    if args.hyperparameter_tuning:
        updated_config['hyperparameter_optimization']['enabled'] = True
    
    if args.optimize_model:
        updated_config['model_optimization']['enabled'] = True
        updated_config['model_optimization']['method'] = args.optimization_type
    
    return updated_config


def main():
    """Main function for training AI models."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save updated configuration
    config_path = os.path.join(config['output_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Get active stage
    active_stage = config['training']['active_stage']
    logger.info(f"Active training stage: {active_stage}")
    
    # Get stage configuration
    stage_config = None
    for stage in config['training']['stages']:
        if stage['name'] == active_stage:
            stage_config = stage
            break
    
    if stage_config is None:
        raise ValueError(f"Training stage {active_stage} not found in configuration")
    
    # Initialize dataset loader with HuggingFace token if provided
    dataset_loader = DatasetLoader(args.config, huggingface_token=args.hf_token)
    
    # Load datasets
    train_datasets = []
    for dataset_config in stage_config['datasets']:
        if dataset_config.get('split') == 'train':
            dataset = dataset_loader.load_dataset(
                dataset_config['name'],
                subset=dataset_config.get('subset'),
                split='train',
                streaming=dataset_config.get('streaming', False),
                max_samples=dataset_config.get('max_samples')
            )
            train_datasets.append(dataset)
    
    eval_datasets = []
    for dataset_config in stage_config['datasets']:
        if dataset_config.get('split') == 'validation':
            dataset = dataset_loader.load_dataset(
                dataset_config['name'],
                subset=dataset_config.get('subset'),
                split='validation',
                streaming=dataset_config.get('streaming', False),
                max_samples=dataset_config.get('max_samples')
            )
            eval_datasets.append(dataset)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Preprocess datasets
    train_datasets = [preprocessor.process_dataset(ds) for ds in train_datasets]
    eval_datasets = [preprocessor.process_dataset(ds) for ds in eval_datasets]
    
    # Apply data augmentation if enabled
    if config['data_processing']['augmentation']['enabled']:
        # Extract augmentation techniques and probabilities
        techniques = [t['name'] for t in config['data_processing']['augmentation']['techniques']]
        probabilities = [t['probability'] for t in config['data_processing']['augmentation']['techniques']]
        
        # Augment training datasets
        for i, dataset in enumerate(train_datasets):
            train_datasets[i] = augment_dataset(
                dataset=dataset,
                text_column='text' if 'text' in dataset.column_names else 'input',
                techniques=techniques,
                probabilities=probabilities,
                n_aug=1,
                keep_original=True
            )
    
    # Get tokenizer
    tokenizer = get_tokenizer(config['tokenizer'])
    
    # Run hyperparameter tuning if enabled
    if config['hyperparameter_optimization']['enabled']:
        logger.info("Running hyperparameter tuning")
        
        # Get hyperparameter optimization configuration
        hpo_config = config['hyperparameter_optimization']
        
        # Run hyperparameter optimization
        best_params = optimize_hyperparameters(
            config_path=args.config,
            search_space=None,  # Use default search space from config
            num_samples=hpo_config['num_samples'],
            num_epochs=hpo_config['num_epochs'],
            search_alg=hpo_config['search_algorithm'],
            scheduler=hpo_config['scheduler'],
            metric=hpo_config['metric'],
            mode=hpo_config['mode']
        )
        
        logger.info(f"Best hyperparameters: {best_params['best_config']}")
        
        # Update configuration with best hyperparameters
        # (This is handled internally by the optimize_hyperparameters function)
        
        # Load updated configuration
        with open(os.path.join(config['output_dir'], 'best_config', 'best_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    
    # Create model
    model = create_model_from_config(config)
    
    # Evaluate only if requested
    if args.evaluate_only:
        logger.info("Evaluating model")
        
        # Get evaluation configuration
        eval_config = config['evaluation']
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_datasets[0] if eval_datasets else None,
            task_type="text_generation",
            metrics=eval_config['metrics'],
            batch_size=config['data_processing']['batching']['eval_batch_size'],
            max_length=eval_config['generation']['max_length'],
            num_beams=eval_config['generation']['num_beams'],
            temperature=eval_config['generation']['temperature'],
            top_p=eval_config['generation']['top_p'],
            top_k=eval_config['generation']['top_k'],
            do_sample=eval_config['generation']['do_sample'],
            output_dir=os.path.join(config['output_dir'], 'evaluation'),
            visualize=True
        )
        
        logger.info(f"Evaluation metrics: {metrics}")
        return
    
    # Train model
    if args.distributed:
        logger.info("Running distributed training")
        
        # Get distributed training configuration
        dist_config = config['distributed_training']
        
        # Create DeepSpeed configuration if needed
        deepspeed_config = None
        if dist_config['use_deepspeed']:
            deepspeed_config = DeepSpeedConfig(
                zero_stage=dist_config['zero_stage'],
                offload_optimizer=dist_config['offload_optimizer'],
                offload_param=dist_config['offload_param'],
                fp16=config['training']['mixed_precision'] == 'fp16',
                bf16=config['training']['mixed_precision'] == 'bf16',
                gradient_accumulation_steps=config['data_processing']['batching']['gradient_accumulation_steps'],
                gradient_clipping=config['training']['gradient_clipping'],
                output_dir=config['output_dir']
            )
        
        # Run distributed training
        results = train_distributed(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_datasets[0] if train_datasets else None,
            eval_dataset=eval_datasets[0] if eval_datasets else None,
            config_path=args.config,
            output_dir=config['output_dir'],
            num_epochs=stage_config['epochs'],
            batch_size=config['data_processing']['batching']['train_batch_size'],
            learning_rate=stage_config['learning_rate']['initial'],
            weight_decay=config['training']['optimizer']['weight_decay'],
            warmup_steps=stage_config['learning_rate']['warmup_steps'],
            gradient_accumulation_steps=config['data_processing']['batching']['gradient_accumulation_steps'],
            gradient_clipping=config['training']['gradient_clipping'],
            fp16=config['training']['mixed_precision'] == 'fp16',
            bf16=config['training']['mixed_precision'] == 'bf16',
            zero_stage=dist_config['zero_stage'],
            offload_optimizer=dist_config['offload_optimizer'],
            offload_param=dist_config['offload_param'],
            use_deepspeed=dist_config['use_deepspeed'],
            local_rank=args.local_rank,
            seed=config['seed'],
            save_steps=config['training']['checkpointing']['save_steps'],
            eval_steps=config['training']['evaluation']['eval_steps'],
            logging_steps=config['logging']['log_steps'],
            use_wandb=config['logging']['use_wandb'],
            wandb_project=config['logging']['wandb_project'],
            wandb_run_name=f"{config['project_name']}_{active_stage}"
        )
    else:
        logger.info("Running single-GPU training")
        
        # Create training arguments
        training_args = TrainingArguments(config, active_stage)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_datasets[0] if train_datasets else None,
            eval_dataset=eval_datasets[0] if eval_datasets else None,
            args=training_args
        )
        
        # Train model
        results = trainer.train()
    
    logger.info(f"Training results: {results}")
    
    # Optimize model if requested
    if args.optimize_model:
        logger.info(f"Optimizing model with {args.optimization_type}")
        
        # Get optimization configuration
        opt_config = config['model_optimization']
        
        # Optimize model
        optimized_model = optimize_model(
            model=model,
            tokenizer=tokenizer,
            optimization_type=args.optimization_type,
            output_dir=os.path.join(config['output_dir'], 'optimized_model'),
            save_model=True,
            benchmark=True
        )
        
        logger.info(f"Model optimization completed")
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()