"""
Extract discrete units using fairseq speech2unit pipeline with mHuBERT-147
Based on: fairseq/examples/textless_nlp/gslm/speech2unit/clustering/
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from training.trainer_utils import load_config

def run_fairseq_command(command: List[str], description: str = ""):
    """Run fairseq command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_fairseq_installation():
    """Check if fairseq is properly installed"""
    try:
        import fairseq
        fairseq_path = Path("fairseq")
        speech2unit_path = fairseq_path / "examples" / "textless_nlp" / "gslm" / "speech2unit"
        
        if not speech2unit_path.exists():
            print("Error: fairseq speech2unit not found!")
            print("Please run: python fairseq_setup/setup_fairseq.py")
            return False
        
        return True
    except ImportError:
        print("Error: fairseq not installed!")
        print("Please run: python fairseq_setup/setup_fairseq.py")
        return False

def create_fairseq_manifest(audio_files: List[str], 
                           manifest_path: str,
                           audio_root: str = None) -> bool:
    """
    Create fairseq manifest file
    Format:
    <path_of_root_directory_containing_audio_files>
    <relative_path_of_audio_file_1>\t<number_of_frames_1>
    <relative_path_of_audio_file_2>\t<number_of_frames_2>
    """
    print(f"Creating manifest: {manifest_path}")
    
    if audio_root is None:
        audio_root = str(Path(audio_files[0]).parent.parent)
    
    import soundfile as sf
    
    with open(manifest_path, 'w') as f:
        # Write root directory
        f.write(f"{audio_root}\n")
        
        # Write audio files with frame counts
        for audio_file in audio_files:
            try:
                # Load audio to get frame count
                audio, sr = sf.read(audio_file)
                num_frames = len(audio)
                
                # Get relative path
                rel_path = str(Path(audio_file).relative_to(audio_root))
                
                f.write(f"{rel_path}\t{num_frames}\n")
                
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")
                continue
    
    print(f"✓ Created manifest with {len(audio_files)} files")
    return True

def learn_kmeans_clustering(config: ModelConfig,
                           manifest_path: str,
                           output_kmeans_path: str,
                           acoustic_model_path: str,
                           n_clusters: int = 1000,
                           layer: int = 11) -> bool:
    """
    Learn K-means clustering model using fairseq
    
    PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
        --num_clusters $N_CLUSTERS \
        --feature_type $TYPE \
        --checkpoint_path $CKPT_PATH \
        --layer $LAYER \
        --manifest_path $MANIFEST \
        --out_kmeans_model_path $KM_MODEL_PATH
    """
    print("Learning K-means clustering model...")
    
    # Ensure we're in the fairseq directory
    original_dir = os.getcwd()
    fairseq_dir = Path("fairseq")
    
    if not fairseq_dir.exists():
        print("Error: fairseq directory not found!")
        return False
    
    try:
        os.chdir(fairseq_dir)
        
        command = [
            "python", "-m", "examples.textless_nlp.gslm.speech2unit.clustering.cluster_kmeans",
            "--num_clusters", str(n_clusters),
            "--feature_type", "hubert",  # mHuBERT uses hubert feature type
            "--checkpoint_path", acoustic_model_path,
            "--layer", str(layer),
            "--manifest_path", manifest_path,
            "--out_kmeans_model_path", output_kmeans_path
        ]
        
        success = run_fairseq_command(command, "K-means clustering")
        
    finally:
        os.chdir(original_dir)
    
    return success

def quantize_with_kmeans(manifest_path: str,
                        output_quantized_path: str,
                        acoustic_model_path: str,
                        kmeans_model_path: str,
                        layer: int = 11,
                        extension: str = ".wav") -> bool:
    """
    Quantize audio using learned K-means clusters
    
    python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
        --feature_type $TYPE \
        --kmeans_model_path $KM_MODEL_PATH \
        --acoustic_model_path $CKPT_PATH \
        --layer $LAYER \
        --manifest_path $MANIFEST \
        --out_quantized_file_path $OUT_QUANTIZED_FILE \
        --extension ".flac"
    """
    print("Quantizing audio with K-means...")
    
    # Ensure we're in the fairseq directory
    original_dir = os.getcwd()
    fairseq_dir = Path("fairseq")
    
    try:
        os.chdir(fairseq_dir)
        
        command = [
            "python", "-m", "examples.textless_nlp.gslm.speech2unit.clustering.quantize_with_kmeans",
            "--feature_type", "hubert",
            "--kmeans_model_path", kmeans_model_path,
            "--acoustic_model_path", acoustic_model_path,
            "--layer", str(layer),
            "--manifest_path", manifest_path,
            "--out_quantized_file_path", output_quantized_path,
            "--extension", extension
        ]
        
        success = run_fairseq_command(command, "Audio quantization")
        
    finally:
        os.chdir(original_dir)
    
    return success

def collect_audio_files(data_dir: str, 
                       severity_filter: List[str] = None) -> Dict[str, List[str]]:
    """Collect audio files from processed dataset"""
    data_dir = Path(data_dir)
    audio_files = {"healthy": [], "dysarthric": []}
    
    # Collect healthy audio files
    healthy_dir = data_dir / "healthy"
    if healthy_dir.exists():
        for audio_file in healthy_dir.glob("**/*.wav"):
            audio_files["healthy"].append(str(audio_file))
    
    # Collect dysarthric audio files
    dysarthric_dir = data_dir / "dysarthric"
    if dysarthric_dir.exists():
        for severity in ["mild", "moderate", "severe"]:
            if severity_filter and severity not in severity_filter:
                continue
            
            severity_dir = dysarthric_dir / severity
            if severity_dir.exists():
                for audio_file in severity_dir.glob("**/*.wav"):
                    audio_files["dysarthric"].append(str(audio_file))
    
    return audio_files

def main():
    """Main extraction function"""
    parser = argparse.ArgumentParser(description="Extract units using fairseq speech2unit")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to model config file")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed audio data")
    parser.add_argument("--acoustic_model_path", type=str, required=True,
                       help="Path to mHuBERT-147 model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for units and manifests")
    parser.add_argument("--n_clusters", type=int, default=1000,
                       help="Number of K-means clusters")
    parser.add_argument("--layer", type=int, default=11,
                       help="Layer to extract features from")
    parser.add_argument("--severity_filter", type=str, nargs="+", 
                       default=["mild", "moderate"],
                       choices=["mild", "moderate", "severe"],
                       help="Dysarthria severities to include")
    parser.add_argument("--fit_kmeans", action="store_true",
                       help="Fit K-means clustering on healthy speech")
    parser.add_argument("--extract_units", action="store_true",
                       help="Extract discrete units for all data")
    
    args = parser.parse_args()
    
    # Check fairseq installation
    if not check_fairseq_installation():
        return False
    
    # Load configuration
    if args.config_path:
        config_dict = load_config(args.config_path)
        config = ModelConfig()
    else:
        config = ModelConfig()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    # Collect audio files
    print("Collecting audio files...")
    audio_files = collect_audio_files(args.data_dir, args.severity_filter)
    
    print(f"Found {len(audio_files['healthy'])} healthy audio files")
    print(f"Found {len(audio_files['dysarthric'])} dysarthric audio files")
    
    if not audio_files["healthy"] and args.fit_kmeans:
        print("Error: No healthy audio files found for K-means fitting!")
        return False
    
    # Step 1: Fit K-means clustering (if requested)
    kmeans_model_path = output_dir / "kmeans_model.pkl"
    
    if args.fit_kmeans:
        print("\n=== Step 1: Fitting K-means clustering ===")
        
        # Create manifest for healthy speech
        healthy_manifest = manifests_dir / "healthy_manifest.tsv"
        create_fairseq_manifest(
            audio_files["healthy"],
            str(healthy_manifest),
            args.data_dir
        )
        
        # Learn K-means clustering
        success = learn_kmeans_clustering(
            config=config,
            manifest_path=str(healthy_manifest),
            output_kmeans_path=str(kmeans_model_path),
            acoustic_model_path=args.acoustic_model_path,
            n_clusters=args.n_clusters,
            layer=args.layer
        )
        
        if not success:
            print("❌ K-means clustering failed!")
            return False
        
        print(f"✓ K-means model saved to: {kmeans_model_path}")
    
    # Step 2: Extract discrete units (if requested)
    if args.extract_units:
        print("\n=== Step 2: Extracting discrete units ===")
        
        if not kmeans_model_path.exists():
            print(f"Error: K-means model not found at {kmeans_model_path}")
            print("Please run with --fit_kmeans first or provide existing model")
            return False
        
        # Extract units for all data
        all_audio_files = audio_files["healthy"] + audio_files["dysarthric"]
        
        if not all_audio_files:
            print("Error: No audio files found!")
            return False
        
        # Create manifest for all audio
        all_manifest = manifests_dir / "all_audio_manifest.tsv"
        create_fairseq_manifest(
            all_audio_files,
            str(all_manifest),
            args.data_dir
        )
        
        # Quantize audio
        units_output = output_dir / "discrete_units.txt"
        success = quantize_with_kmeans(
            manifest_path=str(all_manifest),
            output_quantized_path=str(units_output),
            acoustic_model_path=args.acoustic_model_path,
            kmeans_model_path=str(kmeans_model_path),
            layer=args.layer,
            extension=".wav"
        )
        
        if not success:
            print("❌ Unit extraction failed!")
            return False
        
        print(f"✓ Discrete units saved to: {units_output}")
    
    print("\n=== Unit extraction completed! ===")
    print("Next steps:")
    print("1. Train soft encoder: python training/train_soft_encoder.py")
    print("2. Train S2UT model: python training/train_s2ut.py")
    print("3. Train HiFi-GAN: python training/train_hifigan.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)