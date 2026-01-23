#!/usr/bin/env python3
"""
Gita Hindi TTS with Merged Emotion Checkpoints

This script reads Gita1.json and generates audio files for each entry,
maintaining the order, and then merges all audio files into one final file.
Each entry is assigned an appropriate emotion based on the speaker (Arjun or Krishna).
"""

import sys
import json
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Default merged checkpoint path
MERGED_CHECKPOINT = "checkpoints/emotion_merged/checkpoint_merged.pt"

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "cpu"  # MPS has compatibility issues, use CPU instead
    print("Note: MPS available but using CPU for better compatibility")
else:
    device = "cpu"

print(f"Using device: {device}")


def save_audio(filename, wav, sample_rate):
    """Save audio using soundfile (more reliable than torchaudio.save on Mac)"""
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(filename, wav, sample_rate)
    print(f"Saved: {filename} ({len(wav) / sample_rate:.2f}s)")


def load_audio(filename):
    """Load audio file and return as numpy array"""
    wav, sample_rate = sf.read(filename)
    return wav, sample_rate


def concatenate_audio_files(audio_files, output_file, sample_rate):
    """Concatenate multiple audio files into one"""
    if not audio_files:
        print("Error: No audio files to concatenate")
        return
    
    print(f"\nConcatenating {len(audio_files)} audio files...")
    concatenated = []
    
    for i, audio_file in enumerate(audio_files):
        wav, sr = load_audio(audio_file)
        if sr != sample_rate:
            print(f"Warning: Sample rate mismatch in {audio_file} ({sr} vs {sample_rate})")
            # Resample if needed (simple linear interpolation for now)
            # For proper resampling, you'd need librosa or scipy
            wav = np.interp(
                np.linspace(0, len(wav), int(len(wav) * sample_rate / sr)),
                np.arange(len(wav)),
                wav
            )
        concatenated.append(wav)
        print(f"  Added {audio_file} ({len(wav) / sample_rate:.2f}s)")
    
    merged_audio = np.concatenate(concatenated)
    save_audio(output_file, merged_audio, sample_rate)
    print(f"\n✓ Merged audio saved: {output_file} (Total: {len(merged_audio) / sample_rate:.2f}s)")


def get_audio_prompt_for_speaker(speaker):
    """
    Get the appropriate audio prompt file for the speaker.
    Arjun: gita1-Arjun.wav
    Krishna: gita1-Krishna.wav
    """
    if speaker == "Arjun":
        prompt_path = "gita1-Arjun.wav"
    elif speaker == "Krishna":
        prompt_path = "gita1-Krishna.wav"
    else:
        prompt_path = None
    
    if prompt_path and Path(prompt_path).exists():
        return prompt_path
    else:
        if prompt_path:
            print(f"Warning: Audio prompt {prompt_path} not found for {speaker}, using default voice")
        return None


def get_emotion_for_speaker(speaker, text):
    """
    Determine appropriate emotion based on speaker and text content.
    Arjun expresses sadness, fear, confusion - appropriate emotions: sad, fearful
    Krishna provides guidance, wisdom - appropriate emotions: calm, neutral
    """
    if speaker == "Arjun":
        # Arjun's statements are emotional - sadness, fear, confusion
        # Use sad or fearful emotions based on text content
        if any(word in text for word in ["शोक", "दुख", "पाप", "हाय"]):
            return "sad", 0.5
        elif any(word in text for word in ["डर", "काँप", "शिथिल"]):
            return "fearful", 0.5
        else:
            return "sad", 0.25  # Default to sad for Arjun's distress
    elif speaker == "Krishna":
        # Krishna's statements are calm, wise, guiding
        return "calm", 0.75
    else:
        return "neutral", 0.75


def load_model_with_merged_checkpoint(checkpoint_path=MERGED_CHECKPOINT):
    """
    Load model and apply merged emotion checkpoint using the new API.
    """
    print("Loading Chatterbox Multilingual Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if Path(checkpoint_path).exists():
        print(f"Loading merged emotion checkpoint: {checkpoint_path}")
        model.load_emotion_checkpoint(checkpoint_path)
    else:
        print(f"Warning: Merged checkpoint not found at {checkpoint_path}")
        print("Using base model without fine-tuned emotions")

    return model


def generate_gita_audio():
    """
    Read Gita1.json and generate audio files for each entry.
    Maintain order and merge all audio files into one final file.
    """
    json_path = Path("Gita1.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        return
    
    # Load JSON file
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        gita_data = json.load(f)
    
    print(f"Loaded {len(gita_data)} entries from Gita1.json")
    
    # Load model
    model = load_model_with_merged_checkpoint()
    
    # Generate audio for each entry
    audio_files = []
    output_dir = Path("gita_audio")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Generating audio files for each entry...")
    print("=" * 70)
    
    for idx, entry in enumerate(gita_data):
        # Get speaker and text
        if "Arjun" in entry:
            speaker = "Arjun"
            text = entry["Arjun"]
        elif "Krishna" in entry:
            speaker = "Krishna"
            text = entry["Krishna"]
        else:
            print(f"Warning: Unknown speaker in entry {idx}, skipping...")
            continue
        
        # Determine emotion
        emotion, intensity = get_emotion_for_speaker(speaker, text)
        
        # Get speaker-specific audio prompt
        audio_prompt_path = get_audio_prompt_for_speaker(speaker)
        
        print(f"\n[{idx+1}/{len(gita_data)}] {speaker}: {text[:50]}...")
        print(f"  Emotion: {emotion} (intensity={intensity})")
        if audio_prompt_path:
            print(f"  Audio prompt: {audio_prompt_path}")
        
        try:
            # Generate audio
            wav = model.generate(
                text=text,
                language_id="hi",
                audio_prompt_path=audio_prompt_path,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )
            
            # Save individual audio file
            filename = output_dir / f"gita_{idx+1:03d}_{speaker.lower()}_{emotion}.wav"
            save_audio(str(filename), wav, model.sr)
            audio_files.append(str(filename))
            
        except Exception as e:
            print(f"  Error generating audio: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Merge all audio files
    if audio_files:
        merged_output = "gita_merged_complete.wav"
        concatenate_audio_files(audio_files, merged_output, model.sr)
        print("\n" + "=" * 70)
        print(f"✓ Successfully generated and merged {len(audio_files)} audio files!")
        print(f"  Individual files: {output_dir}/")
        print(f"  Merged file: {merged_output}")
        print("=" * 70)
    else:
        print("\nError: No audio files were generated successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Hindi TTS audio for Gita1.json")
    parser.add_argument("--checkpoint", type=str, default=MERGED_CHECKPOINT,
                       help=f"Path to merged checkpoint (default: {MERGED_CHECKPOINT})")
    parser.add_argument("--json", type=str, default="Gita1.json",
                       help="Path to Gita1.json file (default: Gita1.json)")

    args = parser.parse_args()

    # Update global checkpoint path if provided
    if args.checkpoint != MERGED_CHECKPOINT:
        MERGED_CHECKPOINT = args.checkpoint

    generate_gita_audio()

