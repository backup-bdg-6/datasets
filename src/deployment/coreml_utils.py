"""
Utility functions for CoreML model conversion and optimization.
"""

import os
import logging
import torch
import yaml
from typing import Dict, Any, Optional, Tuple, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_model_for_coreml(model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Prepare a PyTorch model for CoreML conversion by applying necessary optimizations.
    
    Args:
        model: PyTorch model to prepare
        config: Configuration dictionary with optimization settings
        
    Returns:
        Prepared PyTorch model
    """
    logger.info("Preparing model for CoreML conversion")
    
    # Set model to evaluation mode
    model.eval()
    
    # Apply optimizations based on config
    if config.get('optimize_for_mobile', True):
        try:
            # Try to use torch.utils.mobile_optimizer if available
            from torch.utils.mobile_optimizer import optimize_for_mobile
            model = optimize_for_mobile(model)
            logger.info("Applied mobile optimizations to the model")
        except ImportError:
            logger.warning("torch.utils.mobile_optimizer not available, skipping mobile optimization")
    
    return model

def create_example_inputs(
    input_shapes: Dict[str, List[int]], 
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, ...]:
    """
    Create example inputs for model tracing/exporting.
    
    Args:
        input_shapes: Dictionary mapping input names to their shapes
        device: Device to create tensors on
        
    Returns:
        Tuple of example input tensors
    """
    example_inputs = []
    
    for name, shape in input_shapes.items():
        # Create random tensor with the specified shape
        example_input = torch.rand(*shape, device=device)
        example_inputs.append(example_input)
    
    return tuple(example_inputs)

def trace_model(
    model: torch.nn.Module, 
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> torch.jit.ScriptModule:
    """
    Trace a PyTorch model using torch.jit.trace.
    
    Args:
        model: PyTorch model to trace
        example_inputs: Example inputs for tracing
        
    Returns:
        Traced model
    """
    logger.info("Tracing model with torch.jit.trace")
    
    try:
        # Ensure model is in eval mode
        model.eval()
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_inputs)
            
        # Optimize the traced model
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        logger.info("Model traced successfully")
        return traced_model
    
    except Exception as e:
        logger.error(f"Error tracing model: {e}")
        raise

def export_model(
    model: torch.nn.Module, 
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> Any:  # Return type is torch.export.ExportedProgram but avoiding import issues
    """
    Export a PyTorch model using torch.export.export.
    
    Args:
        model: PyTorch model to export
        example_inputs: Example inputs for exporting
        
    Returns:
        Exported model
    """
    logger.info("Exporting model with torch.export.export")
    
    try:
        # Ensure torch.export is available (PyTorch 2.0+)
        if not hasattr(torch, 'export'):
            raise ImportError("torch.export is not available. Please upgrade to PyTorch 2.0 or newer.")
        
        # Ensure model is in eval mode
        model.eval()
        
        # Export the model
        with torch.no_grad():
            exported_model = torch.export.export(model, example_inputs)
        
        logger.info("Model exported successfully")
        return exported_model
    
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise

def save_model_metadata(
    output_dir: str, 
    model_name: str, 
    input_shapes: Dict[str, List[int]],
    output_shapes: Optional[Dict[str, List[int]]] = None,
    model_description: Optional[str] = None
) -> None:
    """
    Save model metadata to a YAML file.
    
    Args:
        output_dir: Directory to save metadata
        model_name: Name of the model
        input_shapes: Dictionary mapping input names to their shapes
        output_shapes: Dictionary mapping output names to their shapes
        model_description: Description of the model
    """
    metadata = {
        'model_name': model_name,
        'inputs': {name: {'shape': shape} for name, shape in input_shapes.items()},
        'outputs': {name: {'shape': shape} for name, shape in (output_shapes or {}).items()} if output_shapes else {},
        'description': model_description or f"CoreML model converted from {model_name}"
    }
    
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.yaml")
    
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Model metadata saved to {metadata_path}")

def create_swift_example(
    output_dir: str,
    model_name: str,
    input_shapes: Dict[str, List[int]],
    input_descriptions: Optional[Dict[str, str]] = None
) -> None:
    """
    Create a Swift example file for using the CoreML model.
    
    Args:
        output_dir: Directory to save the Swift example
        model_name: Name of the model
        input_shapes: Dictionary mapping input names to their shapes
        input_descriptions: Dictionary mapping input names to their descriptions
    """
    # Create Swift code template
    swift_code = f"""//
// {model_name}_Example.swift
// Example code for using the {model_name} CoreML model
//

import CoreML
import Foundation
import UIKit

class {model_name}Predictor {{
    private let model: MLModel
    
    init() throws {{
        // Load the model
        guard let modelURL = Bundle.main.url(forResource: "{model_name}", withExtension: "mlmodelc") else {{
            throw NSError(domain: "ModelNotFound", code: -1, userInfo: nil)
        }}
        
        self.model = try MLModel(contentsOf: modelURL)
    }}
    
    func predict("""
    
    # Add input parameters
    input_params = []
    for name, shape in input_shapes.items():
        # Determine Swift type based on shape
        if len(shape) == 4:  # NCHW format for images
            param_type = "CVPixelBuffer"
        else:
            param_type = "MLMultiArray"
        
        input_params.append(f"{name.lower()}: {param_type}")
    
    swift_code += ", ".join(input_params)
    swift_code += """) throws -> MLFeatureProvider {{
        // Create input dictionary
        let inputFeatures = try MLDictionary(dictionary: [
"""
    
    # Add input features
    for name in input_shapes.keys():
        swift_code += f'            "{name}": {name.lower()},\n'
    
    swift_code += """        ])
        
        // Get predictions
        return try model.prediction(from: inputFeatures)
    }}
}}

// Example usage
func example{model_name}Usage() {{
    do {{
        let predictor = try {model_name}Predictor()
        
        // Prepare inputs
"""
    
    # Add example input preparation
    for name, shape in input_shapes.items():
        if len(shape) == 4:  # NCHW format for images
            swift_code += f"""        // Create a sample image
        let image = UIImage(named: "sample_image")!
        let pixelBuffer = image.pixelBuffer(width: {shape[2]}, height: {shape[3]})!
        
"""
        else:
            # Create MLMultiArray
            shape_str = ", ".join([str(dim) for dim in shape])
            swift_code += f"""        // Create a sample input array
        let {name.lower()} = try! MLMultiArray(shape: [{shape_str}], dataType: .float32)
        // Fill with sample data
        for i in 0..<{name.lower()}.count {{
            {name.lower()}[i] = NSNumber(value: Float.random(in: 0...1))
        }}
        
"""
    
    swift_code += """        // Get predictions
        let predictions = try predictor.predict("""
    
    # Add input arguments
    input_args = []
    for name in input_shapes.keys():
        if len(input_shapes[name]) == 4:
            input_args.append(f"{name.lower()}: pixelBuffer")
        else:
            input_args.append(f"{name.lower()}: {name.lower()}")
    
    swift_code += ", ".join(input_args)
    
    swift_code += """)
        
        // Process predictions
        // Example: if the model outputs "output" as MLMultiArray
        if let output = predictions.featureValue(for: "output")?.multiArrayValue {{
            // Process the output
            print("Model output shape: \\(output.shape)")
        }}
    }} catch {{
        print("Error: \\(error)")
    }}
}}

// Extension to convert UIImage to CVPixelBuffer
extension UIImage {{
    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {{
        var pixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width, height,
                                         kCVPixelFormatType_32ARGB,
                                         attrs as CFDictionary,
                                         &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {{
            return nil
        }}
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                               width: width, height: height,
                               bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: CGFloat(height))
        context?.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }}
}}
"""
    
    # Save the Swift example
    swift_path = os.path.join(output_dir, f"{model_name}_Example.swift")
    with open(swift_path, 'w') as f:
        f.write(swift_code)
    
    logger.info(f"Swift example saved to {swift_path}")