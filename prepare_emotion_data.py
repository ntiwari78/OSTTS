#!/usr/bin/env python3
"""
Prepare RAVDESS and CREMA-D datasets for emotion fine-tuning.

This script:
1. Parses emotion labels from filenames
2. Resamples audio to consistent sample rate (24kHz for S3Gen)
3. Organizes files into emotion_* folders
4. Creates metadata.json with file mappings

Usage:
    python prepare_emotion_data.py --ravdess /path/to/ravdess --cremad /path/to/cremad --output /path/to/output
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import librosa
import soundfile as sf
from tqdm import tqdm


# RAVDESS emotion mapping (from filename position 3)
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}

# CREMA-D emotion mapping (from filename)
CREMAD_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgusted",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Map to our 11 supported emotions
# Some datasets have different names for same emotion
EMOTION_NORMALIZATION = {
    "neutral": "neutral",
    "calm": "calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "fear": "fearful",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "surprised": "surprised",
    "surprise": "surprised",
    "excited": "excited",
    "whisper": "whisper",
    "shout": "shout",
}

# Target sample rate for Chatterbox
TARGET_SR = 24000


def parse_ravdess_filename(filepath: Path) -> Tuple[str, Dict]:
    """
    Parse RAVDESS filename to extract emotion and metadata.

    Format: MM-VV-EM-EI-ST-RE-AC.wav
    - MM: Modality (01=full-AV, 02=video-only, 03=audio-only)
    - VV: Vocal channel (01=speech, 02=song)
    - EM: Emotion (01-08)
    - EI: Emotional intensity (01=normal, 02=strong)
    - ST: Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
    - RE: Repetition (01=1st, 02=2nd)
    - AC: Actor (01-24, odd=male, even=female)
    """
    parts = filepath.stem.split("-")
    if len(parts) != 7:
        return None, {}

    emotion_code = parts[2]
    emotion = RAVDESS_EMOTIONS.get(emotion_code)

    if emotion is None:
        return None, {}

    metadata = {
        "source": "ravdess",
        "modality": parts[0],
        "vocal_channel": "speech" if parts[1] == "01" else "song",
        "emotion_code": emotion_code,
        "intensity": "normal" if parts[3] == "01" else "strong",
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
        "gender": "male" if int(parts[6]) % 2 == 1 else "female",
    }

    return emotion, metadata


def parse_cremad_filename(filepath: Path) -> Tuple[str, Dict]:
    """
    Parse CREMA-D filename to extract emotion and metadata.

    Format: SSSS_UUU_EEE_II.wav
    - SSSS: Speaker ID (1001-1091)
    - UUU: Utterance (DFA, IEO, IOM, ITH, ITS, IWL, IWW, MTI, TAI, TIE, TSI, WSI)
    - EEE: Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
    - II: Intensity (XX, HI, LO, MD)
    """
    parts = filepath.stem.split("_")
    if len(parts) != 4:
        return None, {}

    emotion_code = parts[2]
    emotion = CREMAD_EMOTIONS.get(emotion_code)

    if emotion is None:
        return None, {}

    metadata = {
        "source": "cremad",
        "speaker_id": parts[0],
        "utterance": parts[1],
        "emotion_code": emotion_code,
        "intensity": parts[3],
    }

    return emotion, metadata


def process_audio_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SR,
) -> bool:
    """
    Load audio, resample to target sample rate, and save.

    Returns True if successful, False otherwise.
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None)

        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Normalize audio to prevent clipping
        max_val = abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95

        # Save
        sf.write(output_path, audio, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def find_ravdess_files(ravdess_dir: Path) -> List[Path]:
    """Find all RAVDESS audio files."""
    files = []

    # Look for Actor_XX directories
    for actor_dir in sorted(ravdess_dir.glob("Actor_*")):
        if actor_dir.is_dir():
            files.extend(actor_dir.glob("*.wav"))

    # Also check for flat structure
    if not files:
        files = list(ravdess_dir.glob("*.wav"))

    return sorted(files)


def find_cremad_files(cremad_dir: Path) -> List[Path]:
    """Find all CREMA-D audio files."""
    # CREMA-D typically has AudioWAV subdirectory
    audio_dir = cremad_dir / "AudioWAV"
    if audio_dir.exists():
        return sorted(audio_dir.glob("*.wav"))

    # Try flat structure
    return sorted(cremad_dir.glob("*.wav"))


def prepare_dataset(
    ravdess_dir: Path = None,
    cremad_dir: Path = None,
    output_dir: Path = None,
    target_sr: int = TARGET_SR,
    max_duration: float = 10.0,
):
    """
    Prepare combined emotion dataset from RAVDESS and CREMA-D.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    stats = defaultdict(lambda: {"count": 0, "files": []})
    all_metadata = []

    # Process RAVDESS
    if ravdess_dir:
        ravdess_dir = Path(ravdess_dir)
        ravdess_files = find_ravdess_files(ravdess_dir)
        print(f"Found {len(ravdess_files)} RAVDESS files")

        for filepath in tqdm(ravdess_files, desc="Processing RAVDESS"):
            emotion, metadata = parse_ravdess_filename(filepath)
            if emotion is None:
                continue

            # Normalize emotion name
            emotion = EMOTION_NORMALIZATION.get(emotion, emotion)

            # Create emotion directory
            emotion_dir = output_dir / f"emotion_{emotion}"
            emotion_dir.mkdir(exist_ok=True)

            # Output filename
            output_filename = f"ravdess_{filepath.stem}.wav"
            output_path = emotion_dir / output_filename

            # Process audio
            if process_audio_file(filepath, output_path, target_sr):
                stats[emotion]["count"] += 1
                stats[emotion]["files"].append(output_filename)

                metadata["emotion"] = emotion
                metadata["output_file"] = str(output_path.relative_to(output_dir))
                metadata["original_file"] = str(filepath)
                all_metadata.append(metadata)

    # Process CREMA-D
    if cremad_dir:
        cremad_dir = Path(cremad_dir)
        cremad_files = find_cremad_files(cremad_dir)
        print(f"Found {len(cremad_files)} CREMA-D files")

        for filepath in tqdm(cremad_files, desc="Processing CREMA-D"):
            emotion, metadata = parse_cremad_filename(filepath)
            if emotion is None:
                continue

            # Normalize emotion name
            emotion = EMOTION_NORMALIZATION.get(emotion, emotion)

            # Create emotion directory
            emotion_dir = output_dir / f"emotion_{emotion}"
            emotion_dir.mkdir(exist_ok=True)

            # Output filename
            output_filename = f"cremad_{filepath.stem}.wav"
            output_path = emotion_dir / output_filename

            # Process audio
            if process_audio_file(filepath, output_path, target_sr):
                stats[emotion]["count"] += 1
                stats[emotion]["files"].append(output_filename)

                metadata["emotion"] = emotion
                metadata["output_file"] = str(output_path.relative_to(output_dir))
                metadata["original_file"] = str(filepath)
                all_metadata.append(metadata)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "total_files": sum(s["count"] for s in stats.values()),
            "emotions": {k: v["count"] for k, v in stats.items()},
            "target_sample_rate": target_sr,
            "files": all_metadata,
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Target sample rate: {target_sr} Hz")
    print(f"\nEmotion distribution:")

    total = 0
    for emotion in sorted(stats.keys()):
        count = stats[emotion]["count"]
        total += count
        print(f"  {emotion:12s}: {count:5d} files")

    print(f"  {'TOTAL':12s}: {total:5d} files")
    print(f"\nMetadata saved to: {metadata_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RAVDESS and CREMA-D datasets for emotion fine-tuning"
    )
    parser.add_argument(
        "--ravdess",
        type=str,
        default="/Users/ntiwari/IITP/attempt4/data",
        help="Path to RAVDESS dataset directory (containing Actor_* folders)",
    )
    parser.add_argument(
        "--cremad",
        type=str,
        default="/Users/ntiwari/IITP/attempt4/data",
        help="Path to CREMA-D dataset directory (containing AudioWAV folder)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Users/ntiwari/IITP/attempt4/chatterbox/data/combined_emotions",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=TARGET_SR,
        help=f"Target sample rate (default: {TARGET_SR})",
    )

    args = parser.parse_args()

    prepare_dataset(
        ravdess_dir=Path(args.ravdess) if args.ravdess else None,
        cremad_dir=Path(args.cremad) if args.cremad else None,
        output_dir=Path(args.output),
        target_sr=args.sample_rate,
    )


if __name__ == "__main__":
    main()
