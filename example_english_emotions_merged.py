#!/usr/bin/env python3
"""
Example: English TTS with Merged Emotion Checkpoints

This script demonstrates using the MERGED emotion checkpoint that combines
training from RAVDESS, CREMA-D, and IESC datasets.

Merged checkpoint: checkpoints/emotion_merged/checkpoint_merged.pt
    - Weighted average of 3 checkpoints:
      * RAVDESS (15.2%): 1,440 English samples, 8 emotions
      * CREMA-D (78.5%): 7,442 English samples, 6 emotions
      * IESC (6.3%): 600 Hindi samples, 5 emotions
    - Total: 9,482 samples across datasets
    - 22.5M merged parameters (LoRA + emotion cross-attention)

Supported emotions: neutral, happy, sad, angry, excited, calm,
                   surprised, fearful, disgusted, whisper, shout

Fine-tuned emotions from merged checkpoint:
    - RAVDESS: angry, calm, disgusted, fearful, happy, neutral, sad, surprised
    - CREMA-D: angry, disgusted, fearful, happy, neutral, sad
    - IESC (Hindi): angry, happy, neutral, sad, surprised
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
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


def example_merged_emotions():
    """
    Example showing all emotions with the merged checkpoint.
    Tests emotions that were fine-tuned across all three datasets.
    """
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"

    model = load_model_with_merged_checkpoint()

    # Display supported emotions
    supported_emotions = model.get_supported_emotions()
    print(f"\nSupported emotions: {', '.join(supported_emotions)}")

    # Emotions trained from each dataset (merged)
    print("\nEmotions fine-tuned from merged datasets:")
    print("  RAVDESS: angry, calm, disgusted, fearful, happy, neutral, sad, surprised")
    print("  CREMA-D: angry, disgusted, fearful, happy, neutral, sad")
    print("  IESC:    angry, happy, neutral, sad, surprised")

    # Test text
    text = "This is a test of the merged emotion checkpoint."

    print(f"\nGenerating speech with different emotions")
    print(f"Text: '{text}'")
    print("=" * 70)

    for emotion in supported_emotions:
        print(f"\nGenerating with emotion: {emotion}")

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=1.0,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"merged_emotion_{emotion}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error generating {emotion}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Done! Generated {len(supported_emotions)} audio files with merged checkpoint.")


def example_context_appropriate_emotions():
    """
    Example testing emotions with context-appropriate English texts.
    Each text is matched with an appropriate emotion.
    """
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"

    model = load_model_with_merged_checkpoint()

    # Test texts with context-appropriate emotions
    test_texts = [
        ("Hello! I'm so happy to see you today!", "greeting_happy", "happy", 1.0),
        ("I'm feeling really sad about what happened.", "sad_statement", "sad", 1.0),
        ("I'm absolutely furious about this situation!", "angry_statement", "angry", 1.2),
        ("This is a normal, neutral statement about the weather.", "neutral_statement", "neutral", 1.0),
        ("I'm really scared and worried about the future.", "fearful_statement", "fearful", 1.0),
        ("Wow! I can't believe this amazing news!", "surprised_statement", "surprised", 1.0),
        ("I'm feeling calm and peaceful right now.", "calm_statement", "calm", 1.0),
        ("This is absolutely disgusting and revolting!", "disgusted_statement", "disgusted", 1.0),
        ("I'm so excited about this incredible opportunity!", "excited_statement", "excited", 1.2),
        ("Can you hear me? I'm whispering quietly.", "whisper_statement", "whisper", 1.0),
        ("I'M SHOUTING AT THE TOP OF MY LUNGS!", "shout_statement", "shout", 1.0),
    ]

    print(f"\nTesting emotions with context-appropriate English texts")
    print("Using merged checkpoint from RAVDESS + CREMA-D + IESC")
    print("=" * 70)

    for text, label, emotion, intensity in test_texts:
        print(f"\nText: {text}")
        print(f"Emotion: {emotion} (intensity={intensity})")

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"merged_{emotion}_{label}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Generated emotion-controlled English speech with merged checkpoint.")


def example_emotion_intensity():
    """
    Example demonstrating emotion intensity control with merged checkpoint.
    """
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"

    model = load_model_with_merged_checkpoint()

    text = "I'm really excited about this amazing opportunity!"
    emotion = "excited"

    # Test different intensity levels
    intensities = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]

    print(f"\nGenerating '{emotion}' emotion at different intensities")
    print(f"Text: {text}")
    print("Using merged checkpoint")
    print("=" * 70)

    for intensity in intensities:
        print(f"\nIntensity: {intensity:.1f}", end="")
        if intensity == 0.0:
            print(" (neutral)")
        elif intensity < 0.5:
            print(" (subtle)")
        elif intensity == 1.0:
            print(" (full - default)")
        elif intensity > 1.0:
            print(" (exaggerated)")
        else:
            print()

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"merged_{emotion}_intensity_{intensity:.1f}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Done! Compare the different intensity levels with merged checkpoint.")


def example_emotion_blending():
    """
    Example demonstrating emotion blending with merged checkpoint.
    """
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"

    model = load_model_with_merged_checkpoint()

    # Test cases for emotion blending
    blend_examples = [
        (
            "I'm happy you're here, but sad you have to leave.",
            {"happy": 0.5, "sad": 0.5},
            "Bittersweet (50% happy + 50% sad)",
            "merged_blend_bittersweet.wav"
        ),
        (
            "I can't believe this is happening!",
            {"excited": 0.6, "fearful": 0.4},
            "Nervous excitement (60% excited + 40% fearful)",
            "merged_blend_nervous_excitement.wav"
        ),
        (
            "This is absolutely unbelievable news!",
            {"surprised": 0.7, "happy": 0.3},
            "Pleasant surprise (70% surprised + 30% happy)",
            "merged_blend_pleasant_surprise.wav"
        ),
        (
            "I'm so tired of this situation.",
            {"sad": 0.4, "angry": 0.6},
            "Frustrated sadness (40% sad + 60% angry)",
            "merged_blend_frustrated.wav"
        ),
        (
            "Everything is going to be okay.",
            {"calm": 0.7, "happy": 0.3},
            "Reassuring calm (70% calm + 30% happy)",
            "merged_blend_reassuring.wav"
        ),
    ]

    print("\nGenerating speech with blended emotions")
    print("Using merged checkpoint")
    print("=" * 70)

    for text, emotion_blend, description, filename in blend_examples:
        print(f"\nText: {text}")
        print(f"Blend: {description}")
        print(f"Weights: {emotion_blend}")

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion_blend=emotion_blend,
                temperature=0.8,
                cfg_weight=0.5,
            )

            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Generated blended emotion samples with merged checkpoint.")


def example_compare_checkpoints():
    """
    Compare outputs from individual checkpoints vs merged checkpoint.
    """
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"

    text = "I'm really happy to be here with all of you today!"
    emotion = "happy"

    checkpoints = [
        ("checkpoints/emotion_lora_ravdess/checkpoint_early_stop.pt", "ravdess"),
        ("checkpoints/emotion_lora_cremad/checkpoint_early_stop.pt", "cremad"),
        ("checkpoints/emotion_lora_iesc/checkpoint_early_stop.pt", "iesc"),
        ("checkpoints/emotion_merged/checkpoint_merged.pt", "merged"),
    ]

    print("\nComparing outputs from different checkpoints")
    print(f"Text: {text}")
    print(f"Emotion: {emotion}")
    print("=" * 70)

    for checkpoint_path, label in checkpoints:
        if not Path(checkpoint_path).exists():
            print(f"\nSkipping {label}: checkpoint not found")
            continue

        print(f"\nLoading {label} checkpoint...")
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        model.load_emotion_checkpoint(checkpoint_path)

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=1.0,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"compare_{label}_{emotion}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")

        # Clean up to free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done! Compare the outputs from different checkpoints.")
    print("The merged checkpoint should combine the strengths of all three.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="English TTS with Merged Emotion Checkpoint Examples")
    parser.add_argument("--example", type=str, default="all_emotions",
                       choices=["all_emotions", "context", "intensity", "blend", "compare"],
                       help="Which example to run")
    parser.add_argument("--checkpoint", type=str, default=MERGED_CHECKPOINT,
                       help=f"Path to merged checkpoint (default: {MERGED_CHECKPOINT})")

    args = parser.parse_args()

    # Update global checkpoint path if provided
    if args.checkpoint != MERGED_CHECKPOINT:
        MERGED_CHECKPOINT = args.checkpoint

    if args.example == "all_emotions":
        example_merged_emotions()
    elif args.example == "context":
        example_context_appropriate_emotions()
    elif args.example == "intensity":
        example_emotion_intensity()
    elif args.example == "blend":
        example_emotion_blending()
    elif args.example == "compare":
        example_compare_checkpoints()
