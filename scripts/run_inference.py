"""
Script for running inference with the complete dysarthric voice conversion pipeline
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from evaluation.inference import (
    DysarthricVoiceConverter,
    DysarthricVoiceConverterWithMHuBERT,
    create_inference_pipeline,
    load_model_checkpoints
)
from evaluation.metrics import ComprehensiveEvaluator
from training.trainer_utils import load_config

def convert_single_file(pipeline,
                       input_path: str,
                       output_path: str,
                       return_intermediate: bool = False) -> Dict:
    """
    Convert a single dysarthric audio file
    
    Args:
        pipeline: Voice conversion pipeline
        input_path: Path to input dysarthric audio
        output_path: Path to save converted audio
        return_intermediate: Whether to return intermediate outputs
        
    Returns:
        Conversion results dictionary
    """
    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")
    
    # Run conversion
    if hasattr(pipeline, 'convert_speech_full_pipeline'):
        # Full pipeline with mHuBERT
        result = pipeline.convert_speech_full_pipeline(
            dysarthric_audio_path=input_path,
            output_path=output_path,
            return_intermediate=return_intermediate
        )
    else:
        # S2UT + HiFi-GAN pipeline
        result = pipeline.convert_speech(
            dysarthric_audio_path=input_path,
            output_path=output_path,
            return_intermediate=return_intermediate
        )
    
    # Print timing information
    print(f"Conversion completed in {result['processing_time']['total']:.2f}s")
    
    if 'processing_time' in result:
        for component, time_taken in result['processing_time'].items():
            if component != 'total':
                print(f"  {component}: {time_taken:.3f}s")
    
    print(f"Original duration: {result['original_duration']:.2f}s")
    print(f"Converted duration: {result['converted_duration']:.2f}s")
    
    return result

def convert_batch(pipeline,
                 input_dir: str,
                 output_dir: str,
                 audio_extensions: List[str] = [".wav", ".WAV"],
                 batch_size: int = 4) -> List[Dict]:
    """
    Convert multiple dysarthric audio files
    
    Args:
        pipeline: Voice conversion pipeline
        input_dir: Directory containing dysarthric audio files
        output_dir: Directory to save converted audio files
        audio_extensions: List of audio file extensions to process
        batch_size: Batch size for processing
        
    Returns:
        List of conversion results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(input_dir.glob(f"**/*{ext}")))
    
    print(f"Found {len(audio_files)} audio files to convert")
    
    if not audio_files:
        print("No audio files found!")
        return []
    
    # Convert files using pipeline's batch method
    if hasattr(pipeline, 'convert_batch'):
        results = pipeline.convert_batch(
            dysarthric_audio_paths=[str(f) for f in audio_files],
            output_dir=str(output_dir),
            batch_size=batch_size
        )
    else:
        # Manual batch processing
        results = []
        for audio_file in audio_files:
            output_file = output_dir / f"converted_{audio_file.stem}.wav"
            
            try:
                result = convert_single_file(
                    pipeline=pipeline,
                    input_path=str(audio_file),
                    output_path=str(output_file),
                    return_intermediate=False
                )
                result["input_file"] = str(audio_file)
                result["success"] = True
                
            except Exception as e:
                print(f"Error converting {audio_file.name}: {e}")
                result = {
                    "input_file": str(audio_file),
                    "success": False,
                    "error": str(e)
                }
            
            results.append(result)
    
    # Save results
    results_file = output_dir / "conversion_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    
    print(f"\nConversion Summary:")
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if not result.get("success", False):
                print(f"  {Path(result['input_file']).name}: {result.get('error', 'Unknown error')}")
    
    return results

def evaluate_conversions(original_dir: str,
                        converted_dir: str,
                        output_file: str,
                        config: ModelConfig) -> Dict:
    """
    Evaluate converted audio against original
    
    Args:
        original_dir: Directory containing original dysarthric audio
        converted_dir: Directory containing converted audio
        output_file: Path to save evaluation results
        config: Model configuration
        
    Returns:
        Evaluation results dictionary
    """
    print(f"Evaluating conversions...")
    print(f"Original audio: {original_dir}")
    print(f"Converted audio: {converted_dir}")
    
    original_dir = Path(original_dir)
    converted_dir = Path(converted_dir)
    
    # Find matching audio files
    original_files = list(original_dir.glob("**/*.wav"))
    converted_files = []
    
    for orig_file in original_files:
        # Look for corresponding converted file
        converted_file = converted_dir / f"converted_{orig_file.stem}.wav"
        if converted_file.exists():
            converted_files.append(str(converted_file))
        else:
            print(f"Warning: No converted file found for {orig_file.name}")
    
    if not converted_files:
        print("No matching converted files found!")
        return {}
    
    print(f"Evaluating {len(converted_files)} file pairs...")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        sampling_rate=config.audio.sampling_rate,
        device=config.device
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        original_audio_files=[str(f) for f in original_files[:len(converted_files)]],
        converted_audio_files=converted_files
    )
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nEvaluation Results:")
    for metric, value in results.items():
        if "_mean" in metric:
            metric_name = metric.replace("_mean", "")
            std_value = results.get(f"{metric_name}_std", 0.0)
            print(f"  {metric_name}: {value:.4f} ± {std_value:.4f}")
    
    return results

def demonstrate_pipeline(pipeline,
                        demo_audio_path: str,
                        output_dir: str) -> Dict:
    """
    Demonstrate the pipeline with a single audio file and detailed analysis
    
    Args:
        pipeline: Voice conversion pipeline
        demo_audio_path: Path to demonstration audio file
        output_dir: Directory to save demonstration outputs
        
    Returns:
        Demonstration results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running pipeline demonstration...")
    print(f"Demo audio: {demo_audio_path}")
    print(f"Output directory: {output_dir}")
    
    # Run conversion with intermediate outputs
    result = convert_single_file(
        pipeline=pipeline,
        input_path=demo_audio_path,
        output_path=str(output_dir / "converted_demo.wav"),
        return_intermediate=True
    )
    
    # Save intermediate outputs if available
    if "intermediate" in result:
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        
        import torch
        import numpy as np
        
        for name, data in result["intermediate"].items():
            if isinstance(data, np.ndarray):
                np.save(intermediate_dir / f"{name}.npy", data)
            elif isinstance(data, torch.Tensor):
                torch.save(data, intermediate_dir / f"{name}.pt")
    
    # Save detailed results
    with open(output_dir / "demo_results.json", 'w') as f:
        # Remove intermediate data for JSON serialization
        save_result = {k: v for k, v in result.items() if k != "intermediate" and k != "converted_audio"}
        json.dump(save_result, f, indent=2)
    
    print(f"Demonstration completed. Results saved to {output_dir}")
    
    return result

def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description="Run dysarthric voice conversion inference")
    
    # Required arguments
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing model checkpoints")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                       choices=["single", "batch", "evaluate", "demo"],
                       help="Inference mode")
    
    # Input/output arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Input audio file (single mode) or directory (batch mode)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file (single mode) or directory (batch/evaluate mode)")
    
    # Optional arguments
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to model config file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    parser.add_argument("--full_pipeline", action="store_true",
                       help="Use full pipeline including soft encoder (requires fairseq units)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for batch processing")
    parser.add_argument("--return_intermediate", action="store_true",
                       help="Return intermediate outputs (single/demo mode)")
    
    # Evaluation specific arguments
    parser.add_argument("--original_dir", type=str, default=None,
                       help="Directory with original audio for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_path:
        config_dict = load_config(args.config_path)
        config = ModelConfig()
        # Update config with loaded values (simplified)
    else:
        config = ModelConfig()
    
    # Create inference pipeline
    print("Loading models...")
    pipeline = create_inference_pipeline(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config_path,
        device=args.device
    )
    
    # Run inference based on mode
    if args.mode == "single":
        result = convert_single_file(
            pipeline=pipeline,
            input_path=args.input,
            output_path=args.output,
            return_intermediate=args.return_intermediate
        )
        
        print("Conversion completed successfully!")
        
    elif args.mode == "batch":
        results = convert_batch(
            pipeline=pipeline,
            input_dir=args.input,
            output_dir=args.output,
            batch_size=args.batch_size
        )
        
        print("Batch conversion completed!")
        
    elif args.mode == "evaluate":
        if not args.original_dir:
            raise ValueError("--original_dir required for evaluation mode")
        
        results = evaluate_conversions(
            original_dir=args.original_dir,
            converted_dir=args.input,  # Converted files directory
            output_file=args.output,   # Evaluation results file
            config=config
        )
        
        print("Evaluation completed!")
        
    elif args.mode == "demo":
        result = demonstrate_pipeline(
            pipeline=pipeline,
            demo_audio_path=args.input,
            output_dir=args.output
        )
        
        print("Pipeline demonstration completed!")

if __name__ == "__main__":
    main()