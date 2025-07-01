"""
Create fairseq manifest files from processed dataset
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import soundfile as sf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def create_fairseq_manifest(audio_files: List[str], 
                           manifest_path: str,
                           audio_root: str) -> bool:
    """
    Create fairseq manifest file
    Format:
    <path_of_root_directory_containing_audio_files>
    <relative_path_of_audio_file_1>\t<number_of_frames_1>
    <relative_path_of_audio_file_2>\t<number_of_frames_2>
    """
    print(f"Creating manifest: {manifest_path}")
    
    with open(manifest_path, 'w') as f:
        # Write root directory
        f.write(f"{audio_root}\n")
        
        # Write audio files with frame counts
        valid_files = 0
        for audio_file in audio_files:
            try:
                # Load audio to get frame count
                audio, sr = sf.read(audio_file)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                
                num_frames = len(audio)
                
                # Get relative path
                rel_path = str(Path(audio_file).relative_to(audio_root))
                
                f.write(f"{rel_path}\t{num_frames}\n")
                valid_files += 1
                
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")
                continue
    
    print(f"✓ Created manifest with {valid_files} files")
    return True

def collect_audio_files_from_metadata(metadata_file: str,
                                     severity_filter: List[str] = None) -> Dict[str, List[str]]:
    """Collect audio files from metadata JSON"""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    audio_files = {"healthy": [], "dysarthric": []}
    
    for item in metadata:
        severity = item["severity"]
        audio_path = item["processed_path"]
        
        if not Path(audio_path).exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        if severity == "healthy":
            audio_files["healthy"].append(audio_path)
        elif severity_filter is None or severity in severity_filter:
            audio_files["dysarthric"].append(audio_path)
    
    return audio_files

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create fairseq manifest files")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed audio data")
    parser.add_argument("--metadata_file", type=str, required=True,
                       help="Path to complete metadata JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for manifest files")
    parser.add_argument("--severity_filter", type=str, nargs="+", 
                       default=["mild", "moderate"],
                       choices=["mild", "moderate", "severe"],
                       help="Dysarthria severities to include")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect audio files
    print("Collecting audio files from metadata...")
    audio_files = collect_audio_files_from_metadata(
        args.metadata_file, 
        args.severity_filter
    )
    
    print(f"Found {len(audio_files['healthy'])} healthy audio files")
    print(f"Found {len(audio_files['dysarthric'])} dysarthric audio files")
    
    # Create manifests
    data_dir = Path(args.data_dir)
    
    # 1. Healthy speech manifest (for K-means training)
    if audio_files["healthy"]:
        healthy_manifest = output_dir / "healthy_manifest.tsv"
        create_fairseq_manifest(
            audio_files["healthy"],
            str(healthy_manifest),
            str(data_dir)
        )
        print(f"✓ Created healthy manifest: {healthy_manifest}")
    
    # 2. All audio manifest (for unit extraction)
    all_audio_files = audio_files["healthy"] + audio_files["dysarthric"]
    if all_audio_files:
        all_manifest = output_dir / "all_audio_manifest.tsv"
        create_fairseq_manifest(
            all_audio_files,
            str(all_manifest),
            str(data_dir)
        )
        print(f"✓ Created all audio manifest: {all_manifest}")
    
    # 3. Dysarthric only manifest (for training data preparation)
    if audio_files["dysarthric"]:
        dysarthric_manifest = output_dir / "dysarthric_manifest.tsv"
        create_fairseq_manifest(
            audio_files["dysarthric"],
            str(dysarthric_manifest),
            str(data_dir)
        )
        print(f"✓ Created dysarthric manifest: {dysarthric_manifest}")
    
    print("\nManifest creation completed!")
    print("Use these manifests with fairseq speech2unit pipeline:")
    print(f"- For K-means training: {output_dir}/healthy_manifest.tsv")
    print(f"- For unit extraction: {output_dir}/all_audio_manifest.tsv")

if __name__ == "__main__":
    main()