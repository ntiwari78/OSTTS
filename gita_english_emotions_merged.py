#!/usr/bin/env python3
"""
Gita English TTS with CREMA-D Emotion Checkpoints

This script reads Gita1_English.json and generates audio files for each entry,
maintaining the order, and then merges all audio files into one final file.
Each entry is assigned an appropriate emotion based on the speaker (Arjuna or Krishna).
Uses the CREMA-D fine-tuned emotion checkpoint for enhanced emotion control.
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

# Default CREMA-D checkpoint path
CREMAD_CHECKPOINT = "checkpoints/emotion_lora_cremad/checkpoint_early_stop.pt"

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


def trim_trailing_silence(wav, sample_rate, threshold_db=-40, min_silence_duration=0.1, padding=0.05):
    """
    Remove trailing silence/noise from audio.
    
    Args:
        wav: Audio array (numpy array or torch tensor)
        sample_rate: Sample rate of the audio
        threshold_db: Amplitude threshold in dB below which is considered silence (default: -40dB)
        min_silence_duration: Minimum duration of silence to trim (seconds)
        padding: Small padding to keep at the end (seconds) to avoid abrupt cuts
    
    Returns:
        Trimmed audio array
    """
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    
    if len(wav) == 0:
        return wav
    
    # Convert threshold from dB to linear amplitude
    threshold_linear = 10 ** (threshold_db / 20)
    
    # Calculate absolute amplitude
    abs_wav = np.abs(wav)
    
    # Find frames above threshold
    above_threshold = abs_wav > threshold_linear
    
    # Find the last frame that's above threshold
    if np.any(above_threshold):
        # Find indices where audio is above threshold
        audio_indices = np.where(above_threshold)[0]
        
        if len(audio_indices) > 0:
            # Get the last index with significant audio
            last_audio_idx = audio_indices[-1]
            
            # Add padding to avoid cutting off too abruptly
            padding_samples = int(padding * sample_rate)
            last_audio_idx = last_audio_idx + padding_samples
            
            # Make sure we don't exceed array bounds
            last_audio_idx = min(last_audio_idx, len(wav))
            
            # Trim the audio
            wav = wav[:last_audio_idx]
    
    return wav


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
        # Trim trailing silence from each audio file before concatenating
        wav = trim_trailing_silence(wav, sample_rate, threshold_db=-40, min_silence_duration=0.1, padding=0.05)
        concatenated.append(wav)
        print(f"  Added {audio_file} ({len(wav) / sample_rate:.2f}s)")
    
    merged_audio = np.concatenate(concatenated)
    save_audio(output_file, merged_audio, sample_rate)
    print(f"\n✓ Merged audio saved: {output_file} (Total: {len(merged_audio) / sample_rate:.2f}s)")


def get_audio_prompt_for_speaker(speaker):
    """
    Get the appropriate audio prompt file for the speaker.
    These audio prompts should be from Indian English speakers to ensure Indian accent.
    Arjuna: gita1-Arjun.wav (note: file uses "Arjun" but JSON uses "Arjuna")
    Krishna: gita1-Krishna.wav
    
    The accent is determined by the voice characteristics in the audio prompt,
    so using Indian speaker prompts will produce Indian-accented English.
    """
    if speaker == "Arjuna" or speaker == "Arjun":
        prompt_path = "gita1-Arjun.wav"
        # prompt_path = "Arjun.mp3"
    elif speaker == "Krishna":
        prompt_path = "gita1-Krishna.wav"
    else:
        prompt_path = None
    
    if prompt_path and Path(prompt_path).exists():
        return prompt_path
    else:
        if prompt_path:
            print(f"Warning: Audio prompt {prompt_path} not found for {speaker}, using default voice")
            print("  Note: Without Indian speaker prompts, accent may not be Indian")
        return None


def get_emotion_for_speaker(speaker, text):
    """
    Determine appropriate emotion based on speaker and text content.
    Arjuna expresses sadness, fear, confusion - appropriate emotions: sad, fearful
    Krishna provides guidance, wisdom - appropriate emotions: calm, neutral
    """
    text_lower = text.lower()
    
    if speaker == "Arjuna":
        # Arjuna's statements are emotional - sadness, fear, confusion
        # Use sad or fearful emotions based on text content
        if any(word in text_lower for word in ["sorrow", "grief", "sin", "alas", "evil", "hell", "destroy"]):
            return "sad", 1.0
        elif any(word in text_lower for word in ["trembles", "fear", "fails", "whirl", "confused", "unable"]):
            return "fearful", 1.0
        else:
            return "sad", 0.75  # Default to sad for Arjuna's distress
    elif speaker == "Krishna":
        # Krishna's statements are calm, wise, guiding
        return "calm", 0.25
    else:
        return "reassuring", .5


def load_model_with_cremad_checkpoint(checkpoint_path=CREMAD_CHECKPOINT):
    """
    Load model and apply CREMA-D emotion checkpoint using the new API.
    """
    print("Loading Chatterbox Multilingual Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if Path(checkpoint_path).exists():
        print(f"Loading CREMA-D emotion checkpoint: {checkpoint_path}")
        model.load_emotion_checkpoint(checkpoint_path)
    else:
        print(f"Warning: CREMA-D checkpoint not found at {checkpoint_path}")
        print("Using base model without fine-tuned emotions")

    return model


def generate_gita_audio(checkpoint_path=None):
    """
    Read Gita1_English.json and generate audio files for each entry.
    Maintain order and merge all audio files into one final file.
    """
    json_path = Path("Gita1_English.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        return
    
    # Load JSON file
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        gita_data = json.load(f)
    
    print(f"Loaded {len(gita_data)} entries from Gita1_English.json")
    
    # Load model with CREMA-D checkpoint
    if checkpoint_path is None:
        checkpoint_path = CREMAD_CHECKPOINT
    model = load_model_with_cremad_checkpoint(checkpoint_path)
    
    # Generate audio for each entry
    audio_files = []
    output_dir = Path("gita_audio_english")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Generating audio files for each entry...")
    print("=" * 70)
    
    for idx, entry in enumerate(gita_data):
        # Get speaker and text
        if "Arjuna" in entry:
            speaker = "Arjuna"
            text = entry["Arjuna"]
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
                language_id="en",
                audio_prompt_path=audio_prompt_path,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )
            
            # Trim trailing silence/noise from the generated audio
            wav = trim_trailing_silence(wav, model.sr, threshold_db=-40, min_silence_duration=0.1, padding=0.05)
            
            # Save individual audio file
            filename = output_dir / f"gita_english_{idx+1:03d}_{speaker.lower()}_{emotion}.wav"
            save_audio(str(filename), wav, model.sr)
            audio_files.append(str(filename))
            
        except Exception as e:
            print(f"  Error generating audio: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Merge all audio files
    if audio_files:
        merged_output = "gita_english_merged_complete.wav"
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

    parser = argparse.ArgumentParser(description="Generate English TTS audio for Gita1_English.json")
    parser.add_argument("--checkpoint", type=str, default=CREMAD_CHECKPOINT,
                       help=f"Path to CREMA-D checkpoint (default: {CREMAD_CHECKPOINT})")
    parser.add_argument("--json", type=str, default="Gita1_English.json",
                       help="Path to Gita1_English.json file (default: Gita1_English.json)")

    args = parser.parse_args()

    # Generate audio with specified checkpoint
    generate_gita_audio(checkpoint_path=args.checkpoint)

