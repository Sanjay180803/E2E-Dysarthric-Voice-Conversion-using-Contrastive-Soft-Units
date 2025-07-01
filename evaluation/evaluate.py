"""
Comprehensive evaluation script for dysarthric voice conversion
Implements evaluation methodology from the research paper
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import (
    IntelligibilityMetrics, 
    AudioQualityMetrics, 
    ProsodyMetrics,
    ComprehensiveEvaluator
)
from evaluation.inference import create_inference_pipeline
from config.model_config import ModelConfig
import soundfile as sf

class DysarthricVoiceEvaluator:
    """
    Comprehensive evaluator for dysarthric voice conversion system
    
    Evaluation categories:
    1. Objective Intelligibility (PESQ, STOI, Spectral Convergence)
    2. Subjective Intelligibility (MOS, DMOS, ASR WER)
    3. Audio Quality (MCD, F0 RMSE, Rhythm Similarity)
    4. Semantic Preservation (BERT Similarity)
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 test_data_dir: str,
                 reference_healthy_dir: Optional[str] = None,
                 device: str = "cuda"):
        """
        Initialize evaluator
        
        Args:
            checkpoint_dir: Directory with trained model checkpoints
            test_data_dir: Directory with test dysarthric audio
            reference_healthy_dir: Directory with reference healthy audio (optional)
            device: Device for inference
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.test_data_dir = Path(test_data_dir)
        self.reference_healthy_dir = Path(reference_healthy_dir) if reference_healthy_dir else None
        self.device = device
        
        # Initialize metrics
        self.intelligibility_metrics = IntelligibilityMetrics(device=device)
        self.audio_quality_metrics = AudioQualityMetrics(sampling_rate=16000)
        self.prosody_metrics = ProsodyMetrics(sampling_rate=16000)
        self.comprehensive_evaluator = ComprehensiveEvaluator(
            sampling_rate=16000, 
            device=device
        )
        
        # Load inference pipeline
        print("Loading inference pipeline...")
        self.inference_pipeline = create_inference_pipeline(
            checkpoint_dir=str(checkpoint_dir),
            device=device
        )
        print("✓ Inference pipeline loaded")
        
        # Results storage
        self.evaluation_results = {
            "summary": {},
            "per_file": [],
            "per_severity": {},
            "statistical_analysis": {}
        }
    
    def evaluate_single_file(self, 
                           dysarthric_path: str,
                           reference_path: Optional[str] = None,
                           original_text: Optional[str] = None,
                           converted_text: Optional[str] = None) -> Dict:
        """
        Evaluate single dysarthric audio file
        
        Args:
            dysarthric_path: Path to dysarthric audio
            reference_path: Path to reference healthy audio (optional)
            original_text: Original transcript (optional)
            converted_text: Converted audio transcript (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "file_path": str(dysarthric_path),
            "reference_path": str(reference_path) if reference_path else None,
            "conversion_success": False,
            "metrics": {}
        }
        
        try:
            # Convert dysarthric audio
            conversion_result = self.inference_pipeline.convert_speech(
                dysarthric_audio_path=dysarthric_path,
                return_intermediate=True
            )
            
            if "converted_audio" not in conversion_result:
                results["error"] = "Conversion failed"
                return results
            
            results["conversion_success"] = True
            converted_audio = conversion_result["converted_audio"]
            
            # Load original dysarthric audio
            original_audio, sr = sf.read(dysarthric_path)
            if original_audio.ndim > 1:
                original_audio = original_audio.mean(axis=1)
            
            # Evaluate against reference if available
            if reference_path and Path(reference_path).exists():
                reference_audio, ref_sr = sf.read(reference_path)
                if reference_audio.ndim > 1:
                    reference_audio = reference_audio.mean(axis=1)
                
                # Evaluate converted vs reference
                ref_metrics = self.comprehensive_evaluator.evaluate_pair(
                    original_audio=reference_audio,
                    converted_audio=converted_audio,
                    original_text=converted_text,  # Use converted text as target
                    converted_text=converted_text
                )
                results["metrics"]["vs_reference"] = ref_metrics
            
            # Evaluate converted vs original (improvement metrics)
            improvement_metrics = self._compute_improvement_metrics(
                original_audio, converted_audio
            )
            results["metrics"]["improvement"] = improvement_metrics
            
            # Audio quality metrics
            quality_metrics = self._compute_audio_quality_metrics(
                original_audio, converted_audio
            )
            results["metrics"]["quality"] = quality_metrics
            
            # Processing time metrics
            results["metrics"]["processing_time"] = conversion_result.get("processing_time", {})
            
            # If transcripts available, compute intelligibility metrics
            if original_text and converted_text:
                intelligibility_metrics = self._compute_intelligibility_metrics(
                    original_text, converted_text
                )
                results["metrics"]["intelligibility"] = intelligibility_metrics
            
        except Exception as e:
            results["error"] = str(e)
            print(f"Error evaluating {dysarthric_path}: {e}")
        
        return results
    
    def _compute_improvement_metrics(self, 
                                   original_audio: np.ndarray,
                                   converted_audio: np.ndarray) -> Dict:
        """Compute metrics showing improvement from conversion"""
        metrics = {}
        
        try:
            # Spectral convergence (lower is better - closer to reference)
            spec_conv = self.audio_quality_metrics.compute_spectral_convergence(
                original_audio, converted_audio
            )
            metrics["spectral_convergence"] = spec_conv
            
            # F0 consistency 
            original_f0 = self.prosody_metrics.extract_f0(original_audio)
            converted_f0 = self.prosody_metrics.extract_f0(converted_audio)
            
            # Remove NaN values
            original_f0_clean = original_f0[~np.isnan(original_f0)]
            converted_f0_clean = converted_f0[~np.isnan(converted_f0)]
            
            if len(original_f0_clean) > 0 and len(converted_f0_clean) > 0:
                f0_rmse = np.sqrt(np.mean((original_f0_clean.mean() - converted_f0_clean.mean())**2))
                metrics["f0_rmse"] = f0_rmse
                
                # F0 variability (dysarthric speech often has reduced variability)
                original_f0_std = np.std(original_f0_clean)
                converted_f0_std = np.std(converted_f0_clean)
                metrics["f0_variability_improvement"] = converted_f0_std - original_f0_std
            
            # Speech rate metrics
            original_speech_rate = self._estimate_speech_rate(original_audio)
            converted_speech_rate = self._estimate_speech_rate(converted_audio)
            
            metrics["speech_rate_original"] = original_speech_rate
            metrics["speech_rate_converted"] = converted_speech_rate
            metrics["speech_rate_improvement"] = converted_speech_rate - original_speech_rate
            
        except Exception as e:
            print(f"Error computing improvement metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _compute_audio_quality_metrics(self,
                                     original_audio: np.ndarray,
                                     converted_audio: np.ndarray) -> Dict:
        """Compute audio quality metrics"""
        metrics = {}
        
        try:
            # PESQ (if applicable)
            pesq_score = self.audio_quality_metrics.compute_pesq(
                original_audio, converted_audio
            )
            metrics["pesq"] = pesq_score
            
            # STOI approximation
            stoi_score = self.audio_quality_metrics.compute_stoi(
                original_audio, converted_audio
            )
            metrics["stoi"] = stoi_score
            
            # Mel-Cepstral Distortion
            mcd = self.audio_quality_metrics.compute_mel_cepstral_distortion(
                original_audio, converted_audio
            )
            metrics["mcd"] = mcd
            
            # Signal-to-noise ratio estimation
            snr_original = self._estimate_snr(original_audio)
            snr_converted = self._estimate_snr(converted_audio)
            
            metrics["snr_original"] = snr_original
            metrics["snr_converted"] = snr_converted
            metrics["snr_improvement"] = snr_converted - snr_original
            
        except Exception as e:
            print(f"Error computing audio quality metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _compute_intelligibility_metrics(self,
                                       original_text: str,
                                       converted_text: str) -> Dict:
        """Compute intelligibility metrics using transcripts"""
        metrics = {}
        
        try:
            # Word Error Rate
            wer = self.intelligibility_metrics.compute_word_error_rate(
                [original_text], [converted_text]
            )
            metrics["wer"] = wer
            
            # BERT similarity
            bert_sim = self.intelligibility_metrics.compute_bert_similarity(
                [original_text], [converted_text]
            )
            metrics["bert_similarity"] = bert_sim
            
            # Simple word-level metrics
            original_words = set(original_text.lower().split())
            converted_words = set(converted_text.lower().split())
            
            if len(original_words) > 0:
                word_overlap = len(original_words.intersection(converted_words)) / len(original_words)
                metrics["word_overlap"] = word_overlap
            
        except Exception as e:
            print(f"Error computing intelligibility metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _estimate_speech_rate(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Estimate speech rate (syllables per second)"""
        try:
            # Simple energy-based syllable detection
            # Compute short-time energy
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)   # 10ms hop
            
            energy = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy.append(np.sum(frame**2))
            
            energy = np.array(energy)
            
            # Find peaks (approximating syllables)
            if len(energy) > 0:
                threshold = np.mean(energy) + 0.5 * np.std(energy)
                peaks = np.where(energy > threshold)[0]
                
                # Count syllable-like peaks
                syllable_count = len(peaks)
                duration = len(audio) / sr
                
                return syllable_count / duration if duration > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            print(f"Error estimating speech rate: {e}")
            return 0.0
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        try:
            # Simple SNR estimation
            # Assume signal is the entire audio and noise is the quieter parts
            
            # Compute frame energies
            frame_length = int(0.025 * 16000)  # 25ms
            hop_length = int(0.010 * 16000)    # 10ms
            
            energies = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame_energy = np.mean(audio[i:i + frame_length]**2)
                energies.append(frame_energy)
            
            energies = np.array(energies)
            
            if len(energies) > 0:
                # Signal: top 50% of energies
                # Noise: bottom 20% of energies
                signal_energy = np.mean(np.sort(energies)[-len(energies)//2:])
                noise_energy = np.mean(np.sort(energies)[:len(energies)//5])
                
                if noise_energy > 0:
                    snr = 10 * np.log10(signal_energy / noise_energy)
                    return snr
            
            return 0.0
            
        except Exception as e:
            print(f"Error estimating SNR: {e}")
            return 0.0
    
    def evaluate_dataset(self,
                        severity_levels: List[str] = ["mild", "moderate", "severe"],
                        max_files_per_severity: Optional[int] = None,
                        transcripts_file: Optional[str] = None) -> Dict:
        """
        Evaluate entire test dataset
        
        Args:
            severity_levels: Severity levels to evaluate
            max_files_per_severity: Maximum files per severity (for quick testing)
            transcripts_file: Path to transcripts JSON file (optional)
            
        Returns:
            Complete evaluation results
        """
        print("Starting dataset evaluation...")
        
        # Load transcripts if available
        transcripts = {}
        if transcripts_file and Path(transcripts_file).exists():
            with open(transcripts_file, 'r') as f:
                transcripts_data = json.load(f)
                for item in transcripts_data:
                    file_key = Path(item.get("audio_path", "")).stem
                    transcripts[file_key] = item.get("transcript", "")
        
        all_results = []
        
        for severity in severity_levels:
            print(f"\nEvaluating {severity} severity...")
            
            # Find test files for this severity
            severity_dir = self.test_data_dir / severity
            if not severity_dir.exists():
                print(f"Warning: Directory not found: {severity_dir}")
                continue
            
            test_files = list(severity_dir.rglob("*.wav"))
            
            if max_files_per_severity:
                test_files = test_files[:max_files_per_severity]
            
            print(f"Found {len(test_files)} files for {severity} severity")
            
            severity_results = []
            
            for test_file in tqdm(test_files, desc=f"Evaluating {severity}"):
                # Get transcript if available
                file_key = test_file.stem
                original_text = transcripts.get(file_key)
                
                # Find reference healthy audio if available
                reference_path = None
                if self.reference_healthy_dir:
                    # Look for corresponding healthy audio
                    possible_refs = list(self.reference_healthy_dir.rglob(f"*{file_key}*"))
                    if possible_refs:
                        reference_path = possible_refs[0]
                
                # Evaluate file
                result = self.evaluate_single_file(
                    dysarthric_path=str(test_file),
                    reference_path=str(reference_path) if reference_path else None,
                    original_text=original_text,
                    converted_text=None  # Would need ASR to get this
                )
                
                result["severity"] = severity
                severity_results.append(result)
                all_results.append(result)
            
            # Compute severity-level statistics
            self.evaluation_results["per_severity"][severity] = self._compute_severity_statistics(
                severity_results
            )
        
        # Store all results
        self.evaluation_results["per_file"] = all_results
        
        # Compute overall statistics
        self.evaluation_results["summary"] = self._compute_overall_statistics(all_results)
        
        # Statistical analysis
        self.evaluation_results["statistical_analysis"] = self._perform_statistical_analysis(
            all_results
        )
        
        print("\n✓ Dataset evaluation completed")
        return self.evaluation_results
    
    def _compute_severity_statistics(self, results: List[Dict]) -> Dict:
        """Compute statistics for a specific severity level"""
        stats = {
            "total_files": len(results),
            "successful_conversions": sum(1 for r in results if r["conversion_success"]),
            "failed_conversions": sum(1 for r in results if not r["conversion_success"]),
            "metrics": {}
        }
        
        # Collect all metrics
        successful_results = [r for r in results if r["conversion_success"]]
        
        if successful_results:
            # Average processing time
            processing_times = []
            for r in successful_results:
                pt = r.get("metrics", {}).get("processing_time", {})
                if "total" in pt:
                    processing_times.append(pt["total"])
            
            if processing_times:
                stats["metrics"]["avg_processing_time"] = np.mean(processing_times)
                stats["metrics"]["std_processing_time"] = np.std(processing_times)
            
            # Audio quality metrics
            quality_metrics = ["pesq", "stoi", "mcd", "snr_improvement"]
            for metric in quality_metrics:
                values = []
                for r in successful_results:
                    quality = r.get("metrics", {}).get("quality", {})
                    if metric in quality and not isinstance(quality[metric], str):
                        values.append(quality[metric])
                
                if values:
                    stats["metrics"][f"avg_{metric}"] = np.mean(values)
                    stats["metrics"][f"std_{metric}"] = np.std(values)
            
            # Improvement metrics
            improvement_metrics = ["spectral_convergence", "f0_rmse", "speech_rate_improvement"]
            for metric in improvement_metrics:
                values = []
                for r in successful_results:
                    improvement = r.get("metrics", {}).get("improvement", {})
                    if metric in improvement and not isinstance(improvement[metric], str):
                        values.append(improvement[metric])
                
                if values:
                    stats["metrics"][f"avg_{metric}"] = np.mean(values)
                    stats["metrics"][f"std_{metric}"] = np.std(values)
        
        return stats
    
    def _compute_overall_statistics(self, results: List[Dict]) -> Dict:
        """Compute overall dataset statistics"""
        total_files = len(results)
        successful = sum(1 for r in results if r["conversion_success"])
        
        summary = {
            "total_files_evaluated": total_files,
            "successful_conversions": successful,
            "success_rate": successful / total_files if total_files > 0 else 0.0,
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        }
        
        # Overall metrics across all severities
        successful_results = [r for r in results if r["conversion_success"]]
        
        if successful_results:
            # Processing efficiency
            processing_times = []
            for r in successful_results:
                pt = r.get("metrics", {}).get("processing_time", {})
                if "total" in pt:
                    processing_times.append(pt["total"])
            
            if processing_times:
                summary["avg_processing_time_seconds"] = np.mean(processing_times)
                summary["processing_efficiency"] = len(processing_times) / np.sum(processing_times)  # files per second
            
            # Key performance indicators
            pesq_scores = []
            stoi_scores = []
            mcd_scores = []
            
            for r in successful_results:
                quality = r.get("metrics", {}).get("quality", {})
                if "pesq" in quality and not isinstance(quality["pesq"], str):
                    pesq_scores.append(quality["pesq"])
                if "stoi" in quality and not isinstance(quality["stoi"], str):
                    stoi_scores.append(quality["stoi"])
                if "mcd" in quality and not isinstance(quality["mcd"], str):
                    mcd_scores.append(quality["mcd"])
            
            if pesq_scores:
                summary["overall_pesq"] = np.mean(pesq_scores)
            if stoi_scores:
                summary["overall_stoi"] = np.mean(stoi_scores)
            if mcd_scores:
                summary["overall_mcd"] = np.mean(mcd_scores)
        
        return summary
    
    def _perform_statistical_analysis(self, results: List[Dict]) -> Dict:
        """Perform statistical analysis on results"""
        analysis = {}
        
        try:
            # Group by severity
            severity_groups = {}
            for r in results:
                if r["conversion_success"]:
                    severity = r.get("severity", "unknown")
                    if severity not in severity_groups:
                        severity_groups[severity] = []
                    severity_groups[severity].append(r)
            
            # Compare across severities
            if len(severity_groups) > 1:
                # PESQ comparison
                pesq_by_severity = {}
                for severity, group in severity_groups.items():
                    pesq_scores = []
                    for r in group:
                        quality = r.get("metrics", {}).get("quality", {})
                        if "pesq" in quality and not isinstance(quality["pesq"], str):
                            pesq_scores.append(quality["pesq"])
                    if pesq_scores:
                        pesq_by_severity[severity] = pesq_scores
                
                analysis["pesq_by_severity"] = {
                    severity: {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "min": np.min(scores),
                        "max": np.max(scores),
                        "count": len(scores)
                    }
                    for severity, scores in pesq_by_severity.items()
                }
            
            # Performance trends
            analysis["performance_trends"] = {
                "conversion_success_by_severity": {
                    severity: len([r for r in group if r["conversion_success"]]) / len(group)
                    for severity, group in severity_groups.items()
                },
                "processing_time_by_severity": {
                    severity: np.mean([
                        r.get("metrics", {}).get("processing_time", {}).get("total", 0)
                        for r in group if r["conversion_success"]
                    ])
                    for severity, group in severity_groups.items()
                }
            }
            
        except Exception as e:
            analysis["error"] = str(e)
            print(f"Error in statistical analysis: {e}")
        
        return analysis
    
    def generate_report(self, output_dir: str, include_plots: bool = True) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            output_dir: Directory to save report
            include_plots: Whether to generate plots
            
        Returns:
            Path to generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Generate summary report
        report_file = output_dir / "evaluation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Dysarthric Voice Conversion Evaluation Report\n\n")
            
            # Summary section
            summary = self.evaluation_results.get("summary", {})
            f.write("## Summary\n\n")
            f.write(f"- **Total Files Evaluated**: {summary.get('total_files_evaluated', 0)}\n")
            f.write(f"- **Successful Conversions**: {summary.get('successful_conversions', 0)}\n")
            f.write(f"- **Success Rate**: {summary.get('success_rate', 0):.2%}\n")
            f.write(f"- **Average Processing Time**: {summary.get('avg_processing_time_seconds', 0):.2f}s\n")
            
            if "overall_pesq" in summary:
                f.write(f"- **Overall PESQ Score**: {summary['overall_pesq']:.3f}\n")
            if "overall_stoi" in summary:
                f.write(f"- **Overall STOI Score**: {summary['overall_stoi']:.3f}\n")
            if "overall_mcd" in summary:
                f.write(f"- **Overall MCD**: {summary['overall_mcd']:.3f}\n")
            
            # Per-severity results
            f.write("\n## Results by Severity\n\n")
            per_severity = self.evaluation_results.get("per_severity", {})
            
            for severity, stats in per_severity.items():
                f.write(f"### {severity.title()} Severity\n\n")
                f.write(f"- Files: {stats.get('total_files', 0)}\n")
                f.write(f"- Success Rate: {stats.get('successful_conversions', 0)}/{stats.get('total_files', 0)}\n")
                
                metrics = stats.get("metrics", {})
                if "avg_pesq" in metrics:
                    f.write(f"- Average PESQ: {metrics['avg_pesq']:.3f} ± {metrics.get('std_pesq', 0):.3f}\n")
                if "avg_stoi" in metrics:
                    f.write(f"- Average STOI: {metrics['avg_stoi']:.3f} ± {metrics.get('std_stoi', 0):.3f}\n")
                if "avg_mcd" in metrics:
                    f.write(f"- Average MCD: {metrics['avg_mcd']:.3f} ± {metrics.get('std_mcd', 0):.3f}\n")
                
                f.write("\n")
            
            # Statistical analysis
            f.write("## Statistical Analysis\n\n")
            stat_analysis = self.evaluation_results.get("statistical_analysis", {})
            
            if "pesq_by_severity" in stat_analysis:
                f.write("### PESQ Scores by Severity\n\n")
                pesq_stats = stat_analysis["pesq_by_severity"]
                for severity, stats in pesq_stats.items():
                    f.write(f"- **{severity.title()}**: {stats['mean']:.3f} ± {stats['std']:.3f} "
                           f"(range: {stats['min']:.3f}-{stats['max']:.3f}, n={stats['count']})\n")
                f.write("\n")
        
        # Generate plots if requested
        if include_plots:
            self._generate_evaluation_plots(output_dir)
        
        print(f"✓ Evaluation report generated: {report_file}")
        return str(report_file)
    
    def _generate_evaluation_plots(self, output_dir: Path):
        """Generate evaluation plots"""
        try:
            plt.style.use('seaborn-v0_8')
            
            # Extract data for plotting
            successful_results = [
                r for r in self.evaluation_results.get("per_file", []) 
                if r["conversion_success"]
            ]
            
            if not successful_results:
                print("No successful results for plotting")
                return
            
            # PESQ scores by severity
            pesq_data = []
            severity_data = []
            
            for r in successful_results:
                quality = r.get("metrics", {}).get("quality", {})
                if "pesq" in quality and not isinstance(quality["pesq"], str):
                    pesq_data.append(quality["pesq"])
                    severity_data.append(r.get("severity", "unknown"))
            
            if pesq_data:
                plt.figure(figsize=(10, 6))
                df = pd.DataFrame({"PESQ": pesq_data, "Severity": severity_data})
                sns.boxplot(data=df, x="Severity", y="PESQ")
                plt.title("PESQ Scores by Dysarthria Severity")
                plt.ylabel("PESQ Score")
                plt.tight_layout()
                plt.savefig(output_dir / "pesq_by_severity.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Processing time distribution
            processing_times = []
            for r in successful_results:
                pt = r.get("metrics", {}).get("processing_time", {})
                if "total" in pt:
                    processing_times.append(pt["total"])
            
            if processing_times:
                plt.figure(figsize=(10, 6))
                plt.hist(processing_times, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel("Processing Time (seconds)")
                plt.ylabel("Number of Files")
                plt.title("Processing Time Distribution")
                plt.axvline(np.mean(processing_times), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(processing_times):.2f}s')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / "processing_time_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print("✓ Evaluation plots generated")
            
        except Exception as e:
            print(f"Error generating plots: {e}")

def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description="Evaluate dysarthric voice conversion system")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory with trained model checkpoints")
    parser.add_argument("--test_data_dir", type=str, required=True,
                       help="Directory with test dysarthric audio files")
    parser.add_argument("--reference_healthy_dir", type=str, default=None,
                       help="Directory with reference healthy audio files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device for inference")
    parser.add_argument("--severity_levels", type=str, nargs="+", 
                       default=["mild", "moderate", "severe"],
                       help="Severity levels to evaluate")
    parser.add_argument("--max_files_per_severity", type=int, default=None,
                       help="Maximum files per severity (for quick testing)")
    parser.add_argument("--transcripts_file", type=str, default=None,
                       help="Path to transcripts JSON file")
    parser.add_argument("--include_plots", action="store_true",
                       help="Generate evaluation plots")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DysarthricVoiceEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        test_data_dir=args.test_data_dir,
        reference_healthy_dir=args.reference_healthy_dir,
        device=args.device
    )
    
    # Run evaluation
    print("Starting comprehensive evaluation...")
    results = evaluator.evaluate_dataset(
        severity_levels=args.severity_levels,
        max_files_per_severity=args.max_files_per_severity,
        transcripts_file=args.transcripts_file
    )
    
    # Generate report
    report_path = evaluator.generate_report(
        output_dir=args.output_dir,
        include_plots=args.include_plots
    )
    
    # Print summary
    summary = results.get("summary", {})
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total files evaluated: {summary.get('total_files_evaluated', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.2%}")
    
    if "overall_pesq" in summary:
        print(f"Overall PESQ: {summary['overall_pesq']:.3f}")
    if "overall_stoi" in summary:
        print(f"Overall STOI: {summary['overall_stoi']:.3f}")
    if "overall_mcd" in summary:
        print(f"Overall MCD: {summary['overall_mcd']:.3f}")
    
    print(f"\nDetailed report: {report_path}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()