#!/usr/bin/env python3
"""
Download and prepare Hindi emotion-labeled audio datasets for fine-tuning.

This script helps download and organize Hindi TTS datasets with emotion labels.
"""

import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
import os
import json


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file from URL with progress bar."""
    print(f"Downloading {url} to {output_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print(f"\nDownloaded: {output_path}")


def setup_indic_tts_structure(output_dir: Path):
    """
    Setup directory structure for IndicTTS dataset.
    
    IndicTTS: https://www.iitm.ac.in/donlab/tts/
    """
    print("Setting up IndicTTS directory structure...")
    
    # Create emotion-based directory structure
    emotions = ["happy", "sad", "angry", "neutral", "excited", "calm", "surprised", "fearful"]
    
    for emotion in emotions:
        (output_dir / f"emotion_{emotion}").mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {output_dir}")
    print("\nTo use IndicTTS:")
    print("1. Download from: https://www.iitm.ac.in/donlab/tts/")
    print("2. Extract audio files")
    print("3. Organize by emotion labels")
    print("4. Place in corresponding emotion_* folders")


def setup_ai4bharat_indic_tts(output_dir: Path):
    """
    Setup for AI4Bharat IndicTTS dataset.
    
    GitHub: https://github.com/ai4bharat/IndicTTS
    """
    print("Setting up AI4Bharat IndicTTS structure...")
    
    # Create structure
    setup_indic_tts_structure(output_dir)
    
    print("\nTo use AI4Bharat IndicTTS:")
    print("1. Clone: git clone https://github.com/ai4bharat/IndicTTS")
    print("2. Follow their setup instructions")
    print("3. Extract Hindi audio samples")
    print("4. Label with emotions and organize into emotion_* folders")


def create_sample_metadata(output_dir: Path):
    """Create sample metadata file for emotion labeling."""
    metadata = {
        "emotions": {
            "happy": {
                "description": "Joyful, upbeat speech",
                "examples": ["खुशी", "आनंद", "प्रसन्न"]
            },
            "sad": {
                "description": "Melancholic, subdued speech",
                "examples": ["दुख", "उदासी", "निराशा"]
            },
            "angry": {
                "description": "Aggressive, intense speech",
                "examples": ["क्रोध", "गुस्सा", "रोष"]
            },
            "neutral": {
                "description": "Balanced, natural speech",
                "examples": ["सामान्य", "तटस्थ"]
            },
            "excited": {
                "description": "Very energetic, enthusiastic",
                "examples": ["उत्साहित", "उत्तेजित"]
            },
            "calm": {
                "description": "Peaceful, relaxed",
                "examples": ["शांत", "सुकून"]
            },
            "surprised": {
                "description": "Shocked, amazed",
                "examples": ["आश्चर्य", "चकित"]
            },
            "fearful": {
                "description": "Anxious, worried",
                "examples": ["भय", "डर", "चिंता"]
            }
        },
        "data_format": {
            "audio": "WAV format, 16kHz or 24kHz sample rate",
            "text": "Devanagari script (UTF-8)",
            "naming": "text_emotion.wav or use metadata.json"
        },
        "organization": {
            "structure": "data/hindi_emotions/emotion_*/audio_files.wav",
            "metadata": "Optional: metadata.json with text and emotion labels"
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Created metadata file: {metadata_path}")


def create_labeling_script(output_dir: Path):
    """Create a helper script for labeling audio files."""
    script_content = """#!/usr/bin/env python3
\"\"\"
Helper script to label Hindi audio files with emotions.

Usage:
    python label_audio.py <audio_file> <emotion>
    
Emotions: happy, sad, angry, neutral, excited, calm, surprised, fearful
\"\"\"

import sys
from pathlib import Path
import shutil

if len(sys.argv) != 3:
    print("Usage: python label_audio.py <audio_file> <emotion>")
    sys.exit(1)

audio_file = Path(sys.argv[1])
emotion = sys.argv[2].lower()

valid_emotions = ["happy", "sad", "angry", "neutral", "excited", "calm", "surprised", "fearful"]
if emotion not in valid_emotions:
    print(f"Invalid emotion. Must be one of: {', '.join(valid_emotions)}")
    sys.exit(1)

if not audio_file.exists():
    print(f"Error: {audio_file} does not exist")
    sys.exit(1)

# Move to emotion folder
output_dir = Path("data/hindi_emotions")
emotion_dir = output_dir / f"emotion_{emotion}"
emotion_dir.mkdir(parents=True, exist_ok=True)

dest_file = emotion_dir / audio_file.name
shutil.copy2(audio_file, dest_file)
print(f"Labeled {audio_file.name} as {emotion} -> {dest_file}")
"""
    
    script_path = output_dir / "label_audio.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"Created labeling script: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Hindi emotion-labeled audio dataset for fine-tuning"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hindi_emotions",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["indic_tts", "ai4bharat", "custom"],
        default="custom",
        help="Dataset type to setup"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up Hindi emotion dataset in {output_dir}")
    print("=" * 60)
    
    if args.dataset == "indic_tts":
        setup_indic_tts_structure(output_dir)
    elif args.dataset == "ai4bharat":
        setup_ai4bharat_indic_tts(output_dir)
    else:
        # Custom setup
        setup_indic_tts_structure(output_dir)
        create_sample_metadata(output_dir)
        create_labeling_script(output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset structure ready!")
    print("\nNext steps:")
    print("1. Add your Hindi audio files to emotion_* folders")
    print("2. Ensure audio files are in WAV format (16kHz or 24kHz)")
    print("3. Use label_audio.py to organize files by emotion")
    print("4. Run train_emotion_lora.py to start fine-tuning")
    print("\nRecommended datasets:")
    print("- IndicTTS: https://www.iitm.ac.in/donlab/tts/")
    print("- AI4Bharat IndicTTS: https://github.com/ai4bharat/IndicTTS")
    print("- Common Voice Hindi: https://commonvoice.mozilla.org/")


if __name__ == "__main__":
    main()

