#!/usr/bin/env python3
"""
Reorganize IESC (Indian Emotional Speech Corpora) data to match training requirements.

Maps the IESC structure (Speaker-X/Emotion/) to emotion_*/ structure.
"""

import shutil
from pathlib import Path
from collections import defaultdict
import sys


def reorganize_iesc_data(
    source_dir: str,
    target_dir: str = "data/hindi_emotions",
    copy_files: bool = True
):
    """
    Reorganize IESC data from Speaker-based to emotion-based structure.
    
    Args:
        source_dir: Source directory containing IESC data
        target_dir: Target directory for reorganized data
        copy_files: If True, copy files; if False, create symlinks
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Emotion mapping: IESC emotion names -> our emotion names
    emotion_mapping = {
        "Anger": "angry",
        "Happy": "happy",
        "Sad": "sad",
        "Neutral": "neutral",
        "Fear": "fearful",
    }
    
    # Create target directories
    for emotion in emotion_mapping.values():
        (target_path / f"emotion_{emotion}").mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = defaultdict(int)
    total_files = 0
    
    print(f"Scanning source directory: {source_path}")
    print(f"Target directory: {target_path}")
    print("=" * 60)
    
    # Find all speaker directories
    speaker_dirs = sorted([d for d in source_path.iterdir() if d.is_dir() and d.name.startswith("Speaker-")])
    print(f"Found {len(speaker_dirs)} speaker directories")
    
    # Process each speaker
    for speaker_dir in speaker_dirs:
        speaker_name = speaker_dir.name
        print(f"\nProcessing {speaker_name}...")
        
        # Process each emotion folder
        for emotion_folder in speaker_dir.iterdir():
            if not emotion_folder.is_dir():
                continue
            
            emotion_name = emotion_folder.name
            if emotion_name not in emotion_mapping:
                print(f"  Skipping unknown emotion: {emotion_name}")
                continue
            
            target_emotion = emotion_mapping[emotion_name]
            target_emotion_dir = target_path / f"emotion_{target_emotion}"
            
            # Process all audio files in this emotion folder
            audio_files = list(emotion_folder.glob("*.wav")) + \
                         list(emotion_folder.glob("*.mp3")) + \
                         list(emotion_folder.glob("*.flac"))
            
            for audio_file in audio_files:
                # Create unique filename: speaker_emotion_originalname
                new_filename = f"{speaker_name}_{audio_file.name}"
                target_file = target_emotion_dir / new_filename
                
                if copy_files:
                    shutil.copy2(audio_file, target_file)
                else:
                    # Create symlink
                    target_file.symlink_to(audio_file.absolute())
                
                stats[target_emotion] += 1
                total_files += 1
        
        print(f"  Processed {speaker_name}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Reorganization complete!")
    print(f"Total files processed: {total_files}")
    print("\nFiles per emotion:")
    for emotion, count in sorted(stats.items()):
        print(f"  emotion_{emotion}: {count} files")
    
    # Create summary file
    summary_path = target_path / "data_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("IESC Data Reorganization Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {source_path}\n")
        f.write(f"Target: {target_path}\n")
        f.write(f"Total files: {total_files}\n\n")
        f.write("Files per emotion:\n")
        for emotion, count in sorted(stats.items()):
            f.write(f"  emotion_{emotion}: {count} files\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reorganize IESC data for emotion fine-tuning"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/Users/ntiwari/IITP/data_emotions/Indian Emotional Speech Corpora (IESC)",
        help="Source directory containing IESC data"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/hindi_emotions",
        help="Target directory for reorganized data"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files (saves disk space)"
    )
    
    args = parser.parse_args()
    
    # Check if source exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        sys.exit(1)
    
    # Reorganize
    stats = reorganize_iesc_data(
        source_dir=args.source,
        target_dir=args.target,
        copy_files=not args.symlink
    )
    
    print("\n" + "=" * 60)
    print("Data is now ready for training!")
    print(f"Run: python train_emotion_lora.py --data_dir {args.target}")


if __name__ == "__main__":
    main()

