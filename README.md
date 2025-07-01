# End-to-End Dysarthric Voice Conversion Using Fairseq Speech2Unit Pipeline

A complete implementation of the dysarthric voice conversion system that leverages fairseq's proven speech2unit pipeline with mHuBERT-147 for unit extraction, combined with our novel soft unit modeling and contrastive learning approach.

## Overview

This system converts dysarthric speech to more intelligible healthy speech using a **textless approach** with soft units and contrastive learning. Instead of implementing mHuBERT from scratch, we leverage fairseq's established speech2unit pipeline.

### Pipeline Architecture

1. **Unit Extraction (Fairseq)**: Uses fairseq's speech2unit with mHuBERT-147 to extract discrete units
2. **Soft Content Encoder**: Converts discrete units to continuous soft unit distributions  
3. **Speech-to-Unit Translation (S2UT)**: Transforms dysarthric speech to healthy soft units using contrastive learning
4. **Soft-Unit HiFi-GAN**: Synthesizes high-quality waveforms from soft units



## Installation

### 1. Basic Setup
```bash
git clone <repository-url>
cd dysarthric_voice_conversion
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fairseq Setup
Follow the instructions in fairseq-installation-guide.txt

### 3. Download mHuBERT-147


## Data Preparation

### 1. Organize Raw Data
```
data_raw/
├── speakers/
│   ├── healthy/
│   │   ├── speaker_001/
│   │   │   ├── utterance_001.wav
│   │   │   └── ...
│   │   └── ...
│   └── dysarthric/
│       ├── mild/
│       ├── moderate/
│       └── severe/
```

### 2. Preprocess Dataset
```bash
python scripts/prepare_data.py \
    --raw_data_dir data_raw \
    --output_dir data_processed
```

## Training Pipeline

### Step 1: Extract Discrete Units (Fairseq)

Use fairseq's speech2unit pipeline with mHuBERT-147:

```bash
python scripts/extract_units_fairseq.py \
    --data_dir data_processed \
    --acoustic_model_path models/pretrained/mhubert_base_vp_en_es_fr_it3.pt \
    --output_dir data_processed/units \
    --fit_kmeans \
    --extract_units \
    --n_clusters 1000 \
    --layer 11
```

**This runs the fairseq commands:**
```bash
# 1. Learn K-means clustering model
PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
    --num_clusters 1000 \
    --feature_type hubert \
    --checkpoint_path models/pretrained/mhubert_base_vp_en_es_fr_it3.pt \
    --layer 11 \
    --manifest_path data_processed/manifests/healthy_manifest.tsv \
    --out_kmeans_model_path data_processed/units/kmeans_model.pkl

# 2. Quantize using the learned clusters  
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path data_processed/units/kmeans_model.pkl \
    --acoustic_model_path models/pretrained/mhubert_base_vp_en_es_fr_it3.pt \
    --layer 11 \
    --manifest_path data_processed/manifests/all_audio_manifest.tsv \
    --out_quantized_file_path data_processed/units/discrete_units.txt \
    --extension ".wav"
```

**Outputs:** 
- `data_processed/units/kmeans_model.pkl`
- `data_processed/units/discrete_units.txt`
- Fairseq manifest files

### Step 2: Train Soft Encoder

Convert discrete units to continuous soft distributions:

```bash
python training/train_soft_encoder.py
```

**Outputs:** `checkpoints/soft_encoder/soft_encoder_latest.pt`

### Step 3: Train S2UT Model (Core Component)

Recommended: Before proceeding with the implementation below, we suggest following the instructions in https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/direct_s2st_discrete_units.md, as our approach is inspired by their methodology.

Train the dysarthric-to-healthy conversion model:

```bash
python training/train_s2ut.py
```

**Key Features:**
- Cross-entropy loss for unit prediction
- Contrastive loss (λ=0.1) for robustness across severities  
- SpecAugment for regularization
- Transformer architecture (12 encoder, 6 decoder layers)

 
**Outputs:** `checkpoints/s2ut/s2ut_latest.pt`

### Step 4: Train HiFi-GAN Vocoder


Recommended: Before proceeding with the implementation below, we suggest following the instructions in https://github.com/bshall/hifigan, as our approach is inspired by their methodology.

Synthesize waveforms from soft units:

```bash
python training/train_hifigan.py
```

**Outputs:** `checkpoints/hifigan/hifigan_latest.pt`

## Inference

### Single File Conversion
```bash
python scripts/run_inference.py \
    --mode single \
    --checkpoint_dir checkpoints \
    --input dysarthric_audio.wav \
    --output converted_audio.wav \
    --device cuda
```

### Batch Processing
```bash
python scripts/run_inference.py \
    --mode batch \
    --checkpoint_dir checkpoints \
    --input /path/to/dysarthric_audio_directory \
    --output /path/to/output_directory \
    --device cuda
```

### Python API
```python
from evaluation.inference import create_inference_pipeline

# Load pipeline
pipeline = create_inference_pipeline(
    checkpoint_dir="checkpoints",
    device="cuda"
)

# Convert speech
result = pipeline.convert_speech(
    dysarthric_audio_path="input.wav",
    output_path="output.wav"
)

print(f"Conversion time: {result['processing_time']['total']:.2f}s")
```


## Troubleshooting

### Common Issues
1. **Fairseq Installation**: Run `python fairseq_setup/setup_fairseq.py`
2. **Missing mHuBERT-147**: Download model manually to `models/pretrained/`
3. **Unit File Not Found**: Run `extract_units_fairseq.py` first
4. **Memory Issues**: Reduce batch sizes in config files



## Acknowledgments

This implementation builds upon:
- **[Fairseq Speech2Unit](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit)** by Meta AI
- **[mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147)** multilingual speech representations
- **[HiFi-GAN](https://github.com/jik876/hifi-gan)** high-fidelity neural vocoder
