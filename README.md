# Advanced Machine Learning Training Pipeline

This repository contains a comprehensive, production-grade machine learning training pipeline for creating and training transformer-based models. The pipeline includes components for data loading, preprocessing, augmentation, model architecture, training, evaluation, hyperparameter tuning, and model optimization.

## Features

- **Flexible Data Loading**: Support for various data sources including HuggingFace datasets, local files, and Kaggle datasets
- **Advanced Data Preprocessing**: Text normalization, filtering, and custom preprocessing pipelines
- **Data Augmentation**: Text and code augmentation techniques including synonym replacement, random deletion/insertion, back translation, and more
- **Customizable Model Architecture**: Configurable transformer-based models with various sizes and attention mechanisms
- **Distributed Training**: Support for multi-GPU training with PyTorch DDP and DeepSpeed integration
- **Mixed Precision Training**: FP16 and BF16 support for faster training and reduced memory usage
- **Hyperparameter Optimization**: Integration with Ray Tune for efficient hyperparameter search
- **Model Evaluation**: Comprehensive evaluation metrics for various tasks (text generation, classification, regression)
- **Model Optimization**: Quantization, pruning, and ONNX export for deployment
- **Experiment Tracking**: Integration with Weights & Biases for experiment tracking and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-training-pipeline.git
cd ml-training-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Configure your training in `config/training_config.yaml`
2. Run training:

```bash
python train_model.py --config config/training_config.yaml
```

## Configuration

The pipeline is highly configurable through YAML configuration files. The main configuration file is `config/training_config.yaml`, which includes settings for:

- Model architecture and size
- Tokenizer configuration
- Training stages and datasets
- Optimizer and learning rate schedules
- Data processing and augmentation
- Distributed training
- Hyperparameter optimization
- Model evaluation and optimization

## Directory Structure

```
├── config/                  # Configuration files
│   └── training_config.yaml # Main configuration file
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   │   ├── loaders.py       # Dataset loading utilities
│   │   ├── preprocessors.py # Data preprocessing utilities
│   │   ├── augmentation.py  # Data augmentation utilities
│   │   └── text_preprocessor.py # Text-specific preprocessing
│   ├── model/               # Model architecture and training
│   │   ├── architecture.py  # Model architecture definitions
│   │   ├── training.py      # Training utilities
│   │   ├── distributed_training.py # Distributed training utilities
│   │   ├── optimization.py  # Model optimization utilities
│   │   └── custom_transformer.py # Custom transformer implementation
│   └── utils/               # Utility functions
│       ├── tokenization.py  # Tokenization utilities
│       ├── metrics.py       # Evaluation metrics
│       ├── hyperparameter_tuning.py # Hyperparameter optimization
│       └── model_evaluation.py # Model evaluation utilities
├── train_model.py           # Main training script
└── requirements.txt         # Dependencies
```

## Advanced Usage

### Distributed Training

```bash
# Run with DeepSpeed
python -m torch.distributed.launch --nproc_per_node=4 train_model.py --config config/training_config.yaml --distributed --deepspeed
```

### Hyperparameter Tuning

```bash
python train_model.py --config config/training_config.yaml --hyperparameter_tuning
```

### Model Evaluation

```bash
python train_model.py --config config/training_config.yaml --evaluate_only
```

### Model Optimization

```bash
python train_model.py --config config/training_config.yaml --optimize_model --optimization_type dynamic_quantization
```

## Components

### Data Loading

The `DatasetLoader` class in `src/data/loaders.py` provides utilities for loading datasets from various sources:

- HuggingFace datasets
- Local files (CSV, JSON, text)
- Kaggle datasets

### Data Preprocessing

The `DataPreprocessor` class in `src/data/preprocessors.py` handles data preprocessing:

- Text normalization
- Length filtering
- Custom preprocessing for different data types

### Data Augmentation

The `TextAugmenter` and `CodeAugmenter` classes in `src/data/augmentation.py` provide data augmentation techniques:

- Synonym replacement
- Random deletion/insertion/swapping
- Back translation
- Contextual word embeddings
- Code-specific augmentations

### Model Architecture

The model architecture is defined in `src/model/architecture.py` and includes:

- Transformer blocks with configurable parameters
- Attention mechanisms (causal, rotary embeddings)
- Customizable model sizes

### Training

The `Trainer` class in `src/model/training.py` handles model training:

- Learning rate scheduling
- Mixed precision training
- Gradient accumulation
- Early stopping

### Distributed Training

The `DistributedTrainer` class in `src/model/distributed_training.py` extends the `Trainer` for distributed training:

- PyTorch DDP integration
- DeepSpeed integration with ZeRO optimization
- Multi-node training support

### Hyperparameter Tuning

The `HyperparameterOptimizer` class in `src/utils/hyperparameter_tuning.py` provides hyperparameter optimization:

- Integration with Ray Tune
- Support for various search algorithms (HyperOpt, Bayesian Optimization, Optuna)
- Efficient scheduling with ASHA and PBT

### Model Evaluation

The `ModelEvaluator` class in `src/utils/model_evaluation.py` handles model evaluation:

- Task-specific metrics (text generation, classification, regression)
- Visualization of evaluation results
- Integration with Weights & Biases

### Model Optimization

The `ModelOptimizer` class in `src/model/optimization.py` provides model optimization techniques:

- Dynamic and static quantization
- Model pruning
- ONNX export and optimization
- Benchmarking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)