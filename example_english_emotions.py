#!/usr/bin/env python3
"""
Example: English TTS with Emotion Control (v0.1 API)

This script demonstrates the new emotion control features:
- 64D emotion embeddings
- Emotion intensity control (0.0 to 1.5+)
- Emotion blending/interpolation
- Cross-attention based conditioning

Supported emotions: neutral, happy, sad, angry, excited, calm,
                   surprised, fearful, disgusted, whisper, shout

Fine-tuned emotions (RAVDESS + CREMA-D dataset, 8,882 samples):
    angry, calm, disgusted, fearful, happy, neutral, sad, surprised

Latest checkpoint: checkpoints/emotion_lora/checkpoint_early_stop.pt
    - Trained with LoRA (rank=8, alpha=16.0)
    - 22.5M trainable parameters
    - Early stopped when loss < 0.4 for 50 consecutive batches
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


def main():
    """
    Basic example: Generate English speech with default emotion (neutral).
    """
    print("Loading Chatterbox Multilingual Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Model loaded successfully!")

    # English text samples
    texts = [
        ("Hello, how are you today?", "english_greeting.wav"),
        ("The weather is beautiful today.", "english_weather.wav"),
        ("This model supports 23 languages.", "english_tech.wav"),
        ("Artificial intelligence is transforming the way we interact with technology.", "english_ai.wav"),
    ]

    # Optional: Reference audio for voice cloning
    AUDIO_PROMPT_PATH = "Lady_voice_sample_1.wav"
    audio_prompt = AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None
    if audio_prompt:
        print(f"Using reference audio: {AUDIO_PROMPT_PATH}")
    else:
        print(f"No reference audio found at {AUDIO_PROMPT_PATH}, using default voice")

    print("\nGenerating English speech...")
    print("=" * 50)

    for text, output_file in texts:
        print(f"\nText: {text}")

        # Generate audio with default emotion (neutral)
        wav = model.generate(
            text=text,
            language_id="en",
            audio_prompt_path=audio_prompt,
            emotion="neutral",        # Default emotion
            emotion_intensity=1.0,    # Full intensity
            temperature=0.8,
            cfg_weight=0.5,
        )

        save_audio(output_file, wav, model.sr)

    print("\n" + "=" * 50)
    print("Done! Generated audio files:")
    for _, filename in texts:
        print(f"  - {filename}")


def example_with_different_emotions(use_finetuned=True, checkpoint_path=None):
    """
    Example showing all 11 emotion types with the same English text.

    Args:
        use_finetuned: If True, load fine-tuned model from checkpoint
        checkpoint_path: Path to fine-tuned checkpoint
    """
    AUDIO_PROMPT_PATH = "Male_voice_sample_1.wav"
    print("Loading model for emotion examples...")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if use_finetuned:
        if checkpoint_path is None:
            # Latest checkpoint from RAVDESS + CREMA-D training
            checkpoint_path = "checkpoints/emotion_lora/checkpoint_early_stop.pt"

        print(f"Loading fine-tuned model from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.t3.load_state_dict(checkpoint["t3_state_dict"], strict=False)
            model.emotion_embeddings.load_state_dict(checkpoint["emotion_embeddings_state_dict"])
            print(f"Loaded fine-tuned weights (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned weights: {e}")
            print("Using base model instead")

    # Display supported emotions
    supported_emotions = model.get_supported_emotions()
    print(f"\nSupported emotions: {', '.join(supported_emotions)}")

    # RAVDESS + CREMA-D fine-tuned emotions (8,882 samples)
    finetuned_emotions = ["angry", "calm", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    print(f"Fine-tuned emotions: {', '.join(finetuned_emotions)}")

    # Test text
    text = "This is a test of emotion control in speech synthesis."

    print(f"\nGenerating speech with different emotion types for: '{text}'")
    print("=" * 70)

    for emotion in supported_emotions:
        print(f"\nGenerating with emotion: {emotion}", end="")
        if emotion in finetuned_emotions:
            print(" (fine-tuned)", end="")
        print()

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=1.0,  # Full intensity
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"english_emotion_{emotion}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error generating {emotion}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Done! Generated {len(supported_emotions)} audio files with different emotions.")


def example_emotion_intensity():
    """
    Example demonstrating emotion intensity control.

    Intensity values:
    - 0.0: Neutral (no emotion)
    - 0.5: Subtle emotion
    - 1.0: Full emotion (default)
    - 1.5: Exaggerated emotion
    """
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"
    print("Loading model for intensity examples...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    text = "I'm really excited about this amazing opportunity!"
    emotion = "excited"

    # Test different intensity levels
    intensities = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]

    print(f"\nGenerating '{emotion}' emotion at different intensities")
    print(f"Text: {text}")
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

            filename = f"english_{emotion}_intensity_{intensity:.1f}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Done! Compare the different intensity levels.")
    print("Lower values = more neutral, Higher values = more emotional")


def example_emotion_blending():
    """
    Example demonstrating emotion blending/interpolation.

    Blend multiple emotions by specifying weights that sum to any positive value
    (they will be normalized internally).
    """
    AUDIO_PROMPT_PATH = "Male_voice_sample_1.wav"
    print("Loading model for emotion blending examples...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Test cases for emotion blending
    blend_examples = [
        # (text, emotion_blend, description, filename)
        (
            "I'm happy you're here, but sad you have to leave.",
            {"happy": 0.5, "sad": 0.5},
            "Bittersweet (50% happy + 50% sad)",
            "blend_bittersweet.wav"
        ),
        (
            "I can't believe this is happening!",
            {"excited": 0.6, "fearful": 0.4},
            "Nervous excitement (60% excited + 40% fearful)",
            "blend_nervous_excitement.wav"
        ),
        (
            "This is absolutely unbelievable news!",
            {"surprised": 0.7, "happy": 0.3},
            "Pleasant surprise (70% surprised + 30% happy)",
            "blend_pleasant_surprise.wav"
        ),
        (
            "I'm so tired of this situation.",
            {"sad": 0.4, "angry": 0.6},
            "Frustrated sadness (40% sad + 60% angry)",
            "blend_frustrated.wav"
        ),
        (
            "Everything is going to be okay.",
            {"calm": 0.7, "happy": 0.3},
            "Reassuring calm (70% calm + 30% happy)",
            "blend_reassuring.wav"
        ),
    ]

    print("\nGenerating speech with blended emotions")
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
                emotion_blend=emotion_blend,  # Use blend instead of single emotion
                temperature=0.8,
                cfg_weight=0.5,
            )

            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Generated blended emotion samples.")
    print("Emotion blending allows for more nuanced emotional expressions.")


def example_context_appropriate_emotions(use_finetuned=True, checkpoint_path=None):
    """
    Example testing emotions with context-appropriate English texts.
    Each text is matched with an appropriate emotion.
    """
    AUDIO_PROMPT_PATH = "Male_voice_sample_1.wav"
    print("Loading model for context-appropriate emotion examples...")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if use_finetuned:
        if checkpoint_path is None:
            # Latest checkpoint from RAVDESS + CREMA-D training
            checkpoint_path = "checkpoints/emotion_lora/checkpoint_early_stop.pt"

        print(f"Loading fine-tuned model from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.t3.load_state_dict(checkpoint["t3_state_dict"], strict=False)
            model.emotion_embeddings.load_state_dict(checkpoint["emotion_embeddings_state_dict"])
            print(f"Loaded fine-tuned weights (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned weights: {e}")
            print("Using base model instead")

    # RAVDESS + CREMA-D fine-tuned emotions
    finetuned_emotions = ["angry", "calm", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

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
    print("=" * 70)

    for text, label, emotion, intensity in test_texts:
        print(f"\nText: {text}")
        print(f"Emotion: {emotion} (intensity={intensity})", end="")
        if emotion in finetuned_emotions:
            print(" [fine-tuned]", end="")
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

            filename = f"english_{emotion}_{label}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Generated emotion-controlled English speech samples.")


def example_finetuned_emotions_only(use_finetuned=True, checkpoint_path=None):
    """
    Example testing only the RAVDESS + CREMA-D fine-tuned emotions.
    Fine-tuned emotions: angry, calm, disgusted, fearful, happy, neutral, sad, surprised
    Uses English texts that match the emotion context.
    """
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"
    print("Loading model for fine-tuned emotion examples...")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if use_finetuned:
        if checkpoint_path is None:
            # Latest checkpoint from RAVDESS + CREMA-D training
            checkpoint_path = "checkpoints/emotion_lora/checkpoint_early_stop.pt"

        print(f"Loading fine-tuned model from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.t3.load_state_dict(checkpoint["t3_state_dict"], strict=False)
            model.emotion_embeddings.load_state_dict(checkpoint["emotion_embeddings_state_dict"])
            print(f"Loaded fine-tuned weights (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned weights: {e}")
            print("Using base model instead")

    # RAVDESS + CREMA-D fine-tuned emotions with varied intensities
    test_texts = [
        ("Hello! I'm so happy to see you!", "happy_greeting", "happy", 1.0),
        ("I'm feeling really sad about this.", "sad_statement", "sad", 1.0),
        ("I'm absolutely furious about this!", "angry_statement", "angry", 1.2),
        ("This is a normal, neutral statement.", "neutral_statement", "neutral", 1.0),
        ("I'm really scared and worried.", "fearful_statement", "fearful", 1.0),
        ("I'm feeling calm and peaceful right now.", "calm_statement", "calm", 1.0),
        ("This is absolutely disgusting!", "disgusted_statement", "disgusted", 1.0),
        ("Wow! I can't believe this happened!", "surprised_statement", "surprised", 1.0),
    ]

    print(f"\nTesting RAVDESS + CREMA-D fine-tuned emotions")
    print("=" * 70)

    for text, label, emotion, intensity in test_texts:
        print(f"\nText: {text}")
        print(f"Emotion: {emotion} (intensity={intensity}) [fine-tuned]")

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

            filename = f"english_finetuned_{emotion}_{label}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Generated fine-tuned emotion samples.")


def example_compare_intensity_vs_blend():
    """
    Compare emotion intensity vs emotion blending approaches.

    Shows how to achieve similar effects using different methods:
    1. Using intensity to scale a single emotion
    2. Using blending to mix with neutral
    """
    AUDIO_PROMPT_PATH = "Male_voice_sample_1.wav"
    print("Loading model for intensity vs blend comparison...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    text = "This is a test of subtle emotional expression."
    emotion = "happy"

    print(f"\nComparing intensity control vs neutral blending")
    print(f"Text: {text}")
    print("=" * 70)

    comparisons = [
        # (method, params, description)
        ("intensity", {"emotion": "happy", "emotion_intensity": 0.3}, "30% intensity"),
        ("blend", {"emotion_blend": {"happy": 0.3, "neutral": 0.7}}, "30% happy + 70% neutral blend"),
        ("intensity", {"emotion": "happy", "emotion_intensity": 0.5}, "50% intensity"),
        ("blend", {"emotion_blend": {"happy": 0.5, "neutral": 0.5}}, "50% happy + 50% neutral blend"),
        ("intensity", {"emotion": "happy", "emotion_intensity": 1.0}, "100% intensity (full)"),
        ("blend", {"emotion_blend": {"happy": 1.0}}, "100% happy blend"),
    ]

    for method, params, description in comparisons:
        print(f"\n{method.upper()}: {description}")
        print(f"Params: {params}")

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                temperature=0.8,
                cfg_weight=0.5,
                **params
            )

            filename = f"compare_{method}_{description.replace(' ', '_').replace('%', 'pct')}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Done! Compare the outputs to see differences between methods.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="English TTS with Emotion Control Examples")
    parser.add_argument("--example", type=str, default="all_emotions",
                       choices=["basic", "all_emotions", "intensity", "blend",
                               "context", "finetuned", "compare"],
                       help="Which example to run")
    parser.add_argument("--finetuned", action="store_true", default=True,
                       help="Use fine-tuned model")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to fine-tuned checkpoint (default: checkpoints/emotion_lora/checkpoint_early_stop.pt)")

    args = parser.parse_args()

    if args.example == "basic":
        main()
    elif args.example == "all_emotions":
        example_with_different_emotions(use_finetuned=args.finetuned,
                                        checkpoint_path=args.checkpoint)
    elif args.example == "intensity":
        example_emotion_intensity()
    elif args.example == "blend":
        example_emotion_blending()
    elif args.example == "context":
        example_context_appropriate_emotions(use_finetuned=args.finetuned,
                                             checkpoint_path=args.checkpoint)
    elif args.example == "finetuned":
        example_finetuned_emotions_only(use_finetuned=args.finetuned,
                                        checkpoint_path=args.checkpoint)
    elif args.example == "compare":
        example_compare_intensity_vs_blend()
