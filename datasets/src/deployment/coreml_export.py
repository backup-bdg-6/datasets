"""
Core ML export utilities for the AI model training workflow.
This module provides functions to export models to Apple's Core ML format.
"""

import os
import logging
import json
import shutil
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

# Configure logging
logger = logging.getLogger(__name__)

class CoreMLModelExporter:
    """
    Class for exporting models to Apple's Core ML format.
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
        Initialize the Core ML model exporter.
        
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
        Export the model to Core ML format.
        
        Returns:
            Path to the exported model directory
        """
        # Create model directory
        model_dir = os.path.join(self.output_dir, f"{self.model_name}_coreml")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Exporting model to Core ML format in {model_dir}")
        
        # Check if coremltools is installed
        try:
            import coremltools as ct
        except ImportError:
            logger.error("coremltools not installed. Please install it with 'pip install coremltools'")
            return model_dir
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Apply quantization if enabled
        if self.config['coreml']['quantization']['enabled']:
            self._quantize_model()
        
        # Apply pruning if enabled
        if self.config['coreml']['pruning']['enabled']:
            self._prune_model()
        
        # Export model to ONNX format first
        onnx_path = os.path.join(model_dir, f"{self.model_name}.onnx")
        self._export_to_onnx(onnx_path)
        
        # Convert ONNX model to Core ML
        mlmodel_path = os.path.join(model_dir, f"{self.model_name}.mlmodel")
        self._convert_onnx_to_coreml(onnx_path, mlmodel_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_dir)
        
        # Save Core ML-specific configuration
        coreml_config = {
            "model_name": self.model_name,
            "quantization": self.config['coreml']['quantization'],
            "pruning": self.config['coreml']['pruning'],
            "optimization": self.config['coreml']['optimization']
        }
        
        with open(os.path.join(model_dir, "coreml_config.json"), 'w') as f:
            json.dump(coreml_config, f, indent=2)
        
        # Create example Swift code
        self._create_example_swift_code(model_dir)
        
        logger.info(f"Model exported successfully to {model_dir}")
        
        return model_dir
    
    def _quantize_model(self) -> None:
        """
        Quantize the model to reduce size using PyTorch's quantization capabilities.
        """
        precision = self.config['coreml']['quantization']['precision']
        
        logger.info(f"Quantizing model to {precision} precision")
        
        try:
            import torch.quantization
            from torch.quantization import quantize_dynamic
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Define quantization configuration based on precision
            if precision == 'int8':
                # Prepare the model for dynamic quantization
                quantized_model = quantize_dynamic(
                    self.model,  # the original model
                    {torch.nn.Linear},  # a set of layers to dynamically quantize
                    dtype=torch.qint8  # the target dtype for quantized weights
                )
                self.model = quantized_model
                
            elif precision == 'int4':
                # For int4 quantization, we need to use a more advanced approach
                # First, identify all linear layers
                linear_layers = []
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        linear_layers.append(name)
                
                # Apply weight-only int4 quantization to linear layers
                from transformers.utils.quantization_config import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Apply the configuration to the model
                from transformers import AutoModelForCausalLM
                
                # Save the model temporarily
                temp_dir = os.path.join(self.output_dir, "temp_model")
                os.makedirs(temp_dir, exist_ok=True)
                self.model.save_pretrained(temp_dir)
                
                # Reload with quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    temp_dir,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_dir)
                
            elif precision == 'float16':
                # Convert model to float16
                self.model = self.model.half()
            
            logger.info(f"Model successfully quantized to {precision}")
            
        except ImportError as e:
            logger.error(f"Quantization failed due to missing dependencies: {e}")
            logger.warning("Proceeding with unquantized model")
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            logger.warning("Proceeding with unquantized model")
    
    def _prune_model(self) -> None:
        """
        Prune the model to reduce size by removing less important weights.
        """
        method = self.config['coreml']['pruning']['method']
        target_sparsity = self.config['coreml']['pruning']['target_sparsity']
        
        logger.info(f"Pruning model using {method} method with target sparsity {target_sparsity}")
        
        try:
            import torch.nn.utils.prune as prune
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Track pruned parameters
            pruned_params = 0
            total_params = 0
            
            # Apply pruning to linear layers
            for module_name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Skip output layer to maintain accuracy
                    if "output" in module_name or "classifier" in module_name or "lm_head" in module_name:
                        continue
                    
                    # Apply different pruning methods
                    if method == 'magnitude':
                        prune.l1_unstructured(module, name='weight', amount=target_sparsity)
                    elif method == 'random':
                        prune.random_unstructured(module, name='weight', amount=target_sparsity)
                    elif method == 'structured':
                        # Structured pruning removes entire channels/neurons
                        prune.ln_structured(module, name='weight', amount=target_sparsity, n=2, dim=0)
                    
                    # Count pruned parameters
                    mask = module.weight_mask if hasattr(module, 'weight_mask') else torch.ones_like(module.weight)
                    pruned_params += torch.sum(mask == 0).item()
                    total_params += mask.numel()
            
            # Make pruning permanent (removes the mask and updates the original parameters)
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    prune.remove(module, 'weight')
            
            # Calculate actual sparsity
            actual_sparsity = pruned_params / total_params if total_params > 0 else 0
            
            logger.info(f"Model pruned successfully. Actual sparsity: {actual_sparsity:.4f}")
            
        except ImportError as e:
            logger.error(f"Pruning failed due to missing dependencies: {e}")
            logger.warning("Proceeding with unpruned model")
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            logger.warning("Proceeding with unpruned model")
    
    def _export_to_onnx(self, onnx_path: str) -> None:
        """
        Export the model to ONNX format with optimizations for inference.
        
        Args:
            onnx_path: Path to save the ONNX model
        """
        logger.info(f"Exporting model to ONNX format at {onnx_path}")
        
        try:
            from transformers.onnx import FeaturesManager
            from transformers.onnx.convert import export
            from optimum.onnxruntime import ORTOptimizer
            from optimum.onnxruntime.configuration import OptimizationConfig
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Get model type
            model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self.model)
            
            # Export model to ONNX
            onnx_inputs, onnx_outputs = export(
                preprocessor=self.tokenizer,
                model=self.model,
                config=model_onnx_config,
                opset=12,
                output=onnx_path
            )
            
            logger.info("Model exported to ONNX format successfully")
            
            # Optimize the ONNX model
            logger.info("Optimizing ONNX model...")
            
            # Create optimizer
            optimizer = ORTOptimizer.from_pretrained(self.model)
            
            # Define optimization configuration
            optimization_config = OptimizationConfig(
                optimization_level=99,  # Maximum optimization
                enable_gelu_approximation=True,
                enable_layer_norm_optimization=True,
                enable_attention_fusion=True,
                enable_skip_layer_norm_fusion=True,
                enable_embed_layer_norm_fusion=True,
                enable_bias_gelu_fusion=True,
                enable_gelu_fusion=True,
                enable_graph_optimizations=True
            )
            
            # Optimize model
            optimized_model_path = onnx_path.replace(".onnx", "_optimized.onnx")
            optimizer.optimize(
                optimization_config=optimization_config,
                save_dir=os.path.dirname(optimized_model_path),
                file_suffix="_optimized"
            )
            
            # Replace original model with optimized version
            if os.path.exists(optimized_model_path):
                import shutil
                shutil.move(optimized_model_path, onnx_path)
                logger.info("ONNX model optimized successfully")
            
            # Verify the ONNX model
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verified successfully")
            
        except ImportError as e:
            logger.error(f"ONNX export failed due to missing dependencies: {e}")
            logger.warning("Falling back to basic ONNX export")
            
            # Basic ONNX export
            try:
                # Create dummy input
                batch_size = 1
                seq_len = 16
                input_ids = torch.randint(0, self.tokenizer.vocab_size, (batch_size, seq_len))
                attention_mask = torch.ones_like(input_ids)
                
                # Move tensors to the same device as the model
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                # Export to ONNX
                torch.onnx.export(
                    self.model,
                    (input_ids, attention_mask),
                    onnx_path,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                        'logits': {0: 'batch_size', 1: 'sequence_length'}
                    },
                    opset_version=12,
                    do_constant_folding=True,  # Fold constant values for optimization
                    verbose=False
                )
                
                logger.info("Model exported to ONNX format with basic settings")
            except Exception as e:
                logger.error(f"Basic ONNX export failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Error exporting model to ONNX format: {e}")
            raise
    
    def _convert_onnx_to_coreml(self, onnx_path: str, mlmodel_path: str) -> None:
        """
        Convert ONNX model to Core ML format with optimizations for Apple devices.
        
        Args:
            onnx_path: Path to the ONNX model
            mlmodel_path: Path to save the Core ML model
        """
        logger.info(f"Converting ONNX model to Core ML format at {mlmodel_path}")
        
        try:
            import coremltools as ct
            from coremltools.models.neural_network import quantization_utils
            
            # Load ONNX model
            logger.info("Loading ONNX model...")
            onnx_model = ct.converters.onnx.load(onnx_path)
            
            # Set compute precision based on configuration
            compute_precision = self.config['coreml']['optimization']['compute_precision']
            
            # Map compute precision to coremltools constants
            if compute_precision == 'float32':
                ct_precision = ct.precision.FLOAT32
            elif compute_precision == 'float16':
                ct_precision = ct.precision.FLOAT16
            else:
                logger.warning(f"Unsupported compute precision: {compute_precision}. Using float16.")
                ct_precision = ct.precision.FLOAT16
            
            # Set compute units based on target devices
            memory_layout = self.config['coreml']['optimization'].get('memory_layout', 'gpu_family')
            
            if memory_layout == 'gpu_family':
                compute_units = ct.ComputeUnit.ALL
            elif memory_layout == 'cpu_only':
                compute_units = ct.ComputeUnit.CPU_ONLY
            elif memory_layout == 'neural_engine':
                compute_units = ct.ComputeUnit.CPU_AND_NE
            else:
                compute_units = ct.ComputeUnit.ALL
            
            logger.info(f"Converting with precision {compute_precision} and compute units {compute_units}")
            
            # Define input and output descriptions
            input_descriptions = {
                "input_ids": "Token IDs of the input text",
                "attention_mask": "Mask to avoid performing attention on padding token indices"
            }
            
            output_descriptions = {
                "logits": "Prediction scores for each token position"
            }
            
            # Convert to Core ML
            logger.info("Converting ONNX model to Core ML format...")
            mlmodel = ct.convert(
                onnx_model,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS15,  # Target iOS 15+ for best compatibility
                compute_precision=ct_precision,
                compute_units=compute_units,
                skip_model_load=False
            )
            
            # Set metadata
            mlmodel.author = "AI Model Training Workflow"
            mlmodel.license = "MIT"
            mlmodel.version = "1.0"
            mlmodel.short_description = f"Core ML version of {self.model_name}"
            
            # Add detailed description
            mlmodel.description = f"""
            This model is a Core ML version of {self.model_name}, optimized for Apple devices.
            It can be used for text generation, code completion, and other natural language tasks.
            
            Input:
            - input_ids: Token IDs of the input text
            - attention_mask: Mask to avoid performing attention on padding token indices
            
            Output:
            - logits: Prediction scores for each token position
            
            Quantization: {self.config['coreml']['quantization']['precision']}
            Compute Precision: {compute_precision}
            """
            
            # Add input and output descriptions
            for input_name, description in input_descriptions.items():
                if input_name in mlmodel.input_description:
                    mlmodel.input_description[input_name] = description
            
            for output_name, description in output_descriptions.items():
                if output_name in mlmodel.output_description:
                    mlmodel.output_description[output_name] = description
            
            # Apply additional quantization if needed
            if self.config['coreml']['quantization']['enabled'] and self.config['coreml']['quantization']['precision'] == 'int8':
                logger.info("Applying int8 quantization to Core ML model...")
                mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)
            
            # Save model
            logger.info(f"Saving Core ML model to {mlmodel_path}...")
            mlmodel.save(mlmodel_path)
            
            # Verify the model
            logger.info("Verifying Core ML model...")
            verification_model = ct.models.MLModel(mlmodel_path)
            
            # Print model details
            logger.info(f"Core ML model input: {verification_model.input_description}")
            logger.info(f"Core ML model output: {verification_model.output_description}")
            logger.info(f"Core ML model size: {os.path.getsize(mlmodel_path) / (1024 * 1024):.2f} MB")
            
            logger.info("Model converted to Core ML format successfully")
            
        except ImportError as e:
            logger.error(f"Core ML conversion failed due to missing dependencies: {e}")
            logger.warning("coremltools is required for Core ML conversion. Please install it with 'pip install coremltools'.")
            raise
        except Exception as e:
            logger.error(f"Error converting model to Core ML format: {e}")
            raise
    
    def _create_example_swift_code(self, model_dir: str) -> None:
        """
        Create example Swift code for using the Core ML model.
        
        Args:
            model_dir: Directory containing the exported model
        """
        swift_file = os.path.join(model_dir, "ModelExample.swift")
        
        swift_code = f'''//
// ModelExample.swift
// Example code for using the {self.model_name} Core ML model
//

import Foundation
import CoreML
import NaturalLanguage

class TextGenerator {{
    private let model: MLModel
    private let tokenizer: NLTokenizer
    private let vocabulary: [String: Int]
    private let inverseVocabulary: [Int: String]
    private let maxLength: Int
    
    init?() {{
        // Load the Core ML model
        guard let modelURL = Bundle.main.url(forResource: "{self.model_name}", withExtension: "mlmodel") else {{
            print("Error: Model file not found")
            return nil
        }}
        
        do {{
            let compiledModelURL = try MLModel.compileModel(at: modelURL)
            self.model = try MLModel(contentsOf: compiledModelURL)
        }} catch {{
            print("Error loading model: \\(error)")
            return nil
        }}
        
        // Initialize tokenizer
        self.tokenizer = NLTokenizer(using: .word)
        
        // Load vocabulary
        guard let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "json") else {{
            print("Error: Vocabulary file not found")
            return nil
        }}
        
        do {{
            let vocabData = try Data(contentsOf: vocabURL)
            self.vocabulary = try JSONDecoder().decode([String: Int].self, from: vocabData)
            
            // Create inverse vocabulary
            var inverse = [Int: String]()
            for (token, id) in vocabulary {{
                inverse[id] = token
            }}
            self.inverseVocabulary = inverse
        }} catch {{
            print("Error loading vocabulary: \\(error)")
            return nil
        }}
        
        // Set maximum length
        self.maxLength = 100
    }}
    
    func tokenize(text: String) -> [Int] {{
        // Tokenize the input text
        tokenizer.string = text
        let tokens = tokenizer.tokens(for: text.startIndex..<text.endIndex).map {{
            String(text[text.index($0.range.lowerBound, offsetBy: 0)..<text.index($0.range.upperBound, offsetBy: 0)])
        }}
        
        // Convert tokens to IDs
        return tokens.compactMap {{ vocabulary[$0] }}
    }}
    
    func detokenize(ids: [Int]) -> String {{
        // Convert IDs to tokens
        let tokens = ids.compactMap {{ inverseVocabulary[$0] }}
        
        // Join tokens
        return tokens.joined(separator: " ")
    }}
    
    func generate(prompt: String, temperature: Double = 0.7, maxNewTokens: Int = 50) -> String? {{
        // Tokenize prompt
        var inputIds = tokenize(text: prompt)
        
        // Ensure input is not too long
        if inputIds.count > maxLength {{
            inputIds = Array(inputIds.prefix(maxLength))
        }}
        
        // Create attention mask
        let attentionMask = [Int](repeating: 1, count: inputIds.count)
        
        // Generate tokens one by one
        for _ in 0..<maxNewTokens {{
            // Prepare input
            let inputFeatures = {{
                "input_ids": MLMultiArray(inputIds),
                "attention_mask": MLMultiArray(attentionMask)
            }}
            
            // Run inference
            guard let output = try? model.prediction(from: inputFeatures) else {{
                return nil
            }}
            
            // Get logits
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {{
                return nil
            }}
            
            // Get last token logits
            let lastTokenLogits = getLastTokenLogits(logits)
            
            // Apply temperature
            let scaledLogits = applyTemperature(lastTokenLogits, temperature: temperature)
            
            // Sample next token
            guard let nextToken = sampleToken(scaledLogits) else {{
                break
            }}
            
            // Add token to input
            inputIds.append(nextToken)
            
            // Check if end of sequence
            if nextToken == vocabulary["</s>"] ?? -1 {{
                break
            }}
        }}
        
        // Remove prompt from output
        let outputIds = Array(inputIds.dropFirst(tokenize(text: prompt).count))
        
        // Detokenize output
        return detokenize(ids: outputIds)
    }}
    
    private func getLastTokenLogits(_ logits: MLMultiArray) -> [Double] {{
        // Extract logits for the last token
        let shape = logits.shape
        let batchSize = shape[0].intValue
        let seqLen = shape[1].intValue
        let vocabSize = shape[2].intValue
        
        var lastLogits = [Double]()
        
        for i in 0..<vocabSize {{
            let index = (batchSize - 1) * seqLen * vocabSize + (seqLen - 1) * vocabSize + i
            lastLogits.append(logits[index].doubleValue)
        }}
        
        return lastLogits
    }}
    
    private func applyTemperature(_ logits: [Double], temperature: Double) -> [Double] {{
        // Apply temperature to logits
        return logits.map {{ $0 / max(temperature, 1e-7) }}
    }}
    
    private func sampleToken(_ logits: [Double]) -> Int? {{
        // Convert logits to probabilities
        let expLogits = logits.map {{ exp($0) }}
        let sum = expLogits.reduce(0, +)
        let probs = expLogits.map {{ $0 / sum }}
        
        // Sample from distribution
        let rand = Double.random(in: 0..<1)
        var cumProb = 0.0
        
        for (i, prob) in probs.enumerated() {{
            cumProb += prob
            if rand < cumProb {{
                return i
            }}
        }}
        
        return nil
    }}
}}

// Example usage
func exampleUsage() {{
    guard let generator = TextGenerator() else {{
        print("Failed to initialize generator")
        return
    }}
    
    let prompt = "Once upon a time"
    if let generated = generator.generate(prompt: prompt) {{
        print("Prompt: \\(prompt)")
        print("Generated: \\(generated)")
    }} else {{
        print("Failed to generate text")
    }}
}}
'''
        
        with open(swift_file, 'w') as f:
            f.write(swift_code)
        
        # Create README file
        readme_file = os.path.join(model_dir, "README.md")
        
        readme = f'''# {self.model_name} Core ML Model

This directory contains a Core ML version of the {self.model_name} model.

## Files

- `{self.model_name}.mlmodel`: The Core ML model file
- `ModelExample.swift`: Example Swift code for using the model
- `coreml_config.json`: Configuration used for the Core ML conversion

## Integration

To integrate this model into your iOS or macOS application:

1. Add the `.mlmodel` file to your Xcode project
2. Use the example code in `ModelExample.swift` as a starting point

## Requirements

- iOS 14.0+ / macOS 11.0+
- Xcode 12.0+
- Swift 5.3+

## Model Details

- Model Name: {self.model_name}
- Quantization: {self.config['coreml']['quantization']['precision']}
- Compute Precision: {self.config['coreml']['optimization']['compute_precision']}

## Example Usage

```swift
import CoreML

// Initialize the model
guard let generator = TextGenerator() else {{
    print("Failed to initialize generator")
    return
}}

// Generate text
let prompt = "Once upon a time"
if let generated = generator.generate(prompt: prompt) {{
    print("Generated: \\(generated)")
}}
```

## Performance Considerations

- The model performs best on devices with Neural Engine
- For older devices, consider using a smaller model or reducing the generation length
- Use batching for multiple generations to improve throughput
'''
        
        with open(readme_file, 'w') as f:
            f.write(readme)
        
        logger.info(f"Created example Swift code in {model_dir}")


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
    exporter = CoreMLModelExporter(
        model=model,
        tokenizer=tokenizer,
        output_dir="outputs/deployment",
        model_name="gpt2-coreml",
        config=config
    )
    
    # Export model
    export_dir = exporter.export()
    
    logger.info(f"Model exported to {export_dir}")