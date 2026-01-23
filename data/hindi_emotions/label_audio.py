#!/usr/bin/env python3
"""
Helper script to label Hindi audio files with emotions.

Usage:
    python label_audio.py <audio_file> <emotion>
    
Emotions: happy, sad, angry, neutral, excited, calm, surprised, fearful
"""

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
