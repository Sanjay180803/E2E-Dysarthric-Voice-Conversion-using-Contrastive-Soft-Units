"""
Fairseq setup and utilities for dysarthric voice conversion
"""

from .setup_fairseq import create_fairseq_manifest, collect_audio_files_from_metadata
from .create_manifests import main as create_manifests_main

__all__ = [
    "create_fairseq_manifest",
    "collect_audio_files_from_metadata", 
    "create_manifests_main",

]