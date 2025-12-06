#!/usr/bin/env python3
"""
Example: Hindi TTS with Emotion Control (v0.1 API)

This script demonstrates the emotion control features for Hindi TTS:
- 64D emotion embeddings
- Emotion intensity control (0.0 to 1.5+)
- Emotion blending/interpolation
- Cross-attention based conditioning

Supported emotions: neutral, happy, sad, angry, excited, calm,
                   surprised, fearful, disgusted, whisper, shout

Fine-tuned emotions from datasets:
    - IESC (Hindi): angry, happy, neutral, sad, surprised (600 samples)
    - RAVDESS (English): angry, calm, disgusted, fearful, happy, neutral, sad, surprised
    - CREMA-D (English): angry, disgusted, fearful, happy, neutral, sad

Merged checkpoint: checkpoints/emotion_merged/checkpoint_merged.pt
    - Combines RAVDESS (15.2%), CREMA-D (78.5%), IESC (6.3%)
    - 22.5M trainable parameters
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

# Default checkpoint paths
MERGED_CHECKPOINT = "checkpoints/emotion_merged/checkpoint_merged.pt"
IESC_CHECKPOINT = "checkpoints/emotion_lora_iesc/checkpoint_early_stop.pt"

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
    Basic example: Generate Hindi speech with default emotion (neutral).
    """
    print("Loading Chatterbox Multilingual Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Model loaded successfully!")

    # Hindi text samples (Devanagari script)
    texts = [
        ("नमस्ते, आप कैसे हैं?", "hindi_greeting.wav"),
        ("आज मौसम बहुत अच्छा है।", "hindi_weather.wav"),
        ("यह मॉडल 23 भाषाओं का समर्थन करता है।", "hindi_tech.wav"),
        ("कृत्रिम बुद्धिमत्ता हमारे जीवन को बदल रही है।", "hindi_ai.wav"),
    ]

    # Optional: Reference audio for voice cloning
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    audio_prompt = AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None
    if audio_prompt:
        print(f"Using reference audio: {AUDIO_PROMPT_PATH}")
    else:
        print(f"No reference audio found at {AUDIO_PROMPT_PATH}, using default voice")

    print("\nGenerating Hindi speech...")
    print("=" * 50)

    for text, output_file in texts:
        print(f"\nText: {text}")

        # Generate audio with default emotion (neutral)
        wav = model.generate(
            text=text,
            language_id="hi",  # Hindi language code
            audio_prompt_path=audio_prompt,
            emotion="neutral",
            emotion_intensity=1.0,
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
    Example showing all 11 emotion types with Hindi text.

    Args:
        use_finetuned: If True, load fine-tuned model from checkpoint
        checkpoint_path: Path to fine-tuned checkpoint
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for Hindi emotion examples...")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if use_finetuned:
        if checkpoint_path is None:
            # Use merged checkpoint by default
            checkpoint_path = MERGED_CHECKPOINT

        if Path(checkpoint_path).exists():
            print(f"Loading fine-tuned model from: {checkpoint_path}")
            model.load_emotion_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using base model instead")

    # Display supported emotions
    supported_emotions = model.get_supported_emotions()
    print(f"\nSupported emotions: {', '.join(supported_emotions)}")

    # IESC Hindi fine-tuned emotions
    iesc_emotions = ["angry", "happy", "neutral", "sad", "surprised"]
    print(f"IESC Hindi fine-tuned emotions: {', '.join(iesc_emotions)}")

    # Test text in Hindi
    text = "यह भावनात्मक नियंत्रण का परीक्षण है।"  # "This is a test of emotion control."

    print(f"\nGenerating Hindi speech with different emotions")
    print(f"Text: '{text}'")
    print("=" * 70)

    for emotion in supported_emotions:
        print(f"\nGenerating with emotion: {emotion}", end="")
        if emotion in iesc_emotions:
            print(" (IESC fine-tuned)", end="")
        print()

        try:
            wav = model.generate(
                text=text,
                language_id="hi",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=1.0,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"hindi_emotion_{emotion}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error generating {emotion}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Done! Generated {len(supported_emotions)} Hindi audio files with different emotions.")


def example_emotion_intensity():
    """
    Example demonstrating emotion intensity control for Hindi.

    Intensity values:
    - 0.0: Neutral (no emotion)
    - 0.5: Subtle emotion
    - 1.0: Full emotion (default)
    - 1.5: Exaggerated emotion
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for Hindi intensity examples...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Load merged checkpoint
    if Path(MERGED_CHECKPOINT).exists():
        model.load_emotion_checkpoint(MERGED_CHECKPOINT)

    text = "मैं इस अद्भुत अवसर के लिए बहुत उत्साहित हूं!"  # "I'm very excited about this amazing opportunity!"
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
                language_id="hi",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"hindi_{emotion}_intensity_{intensity:.1f}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Done! Compare the different intensity levels.")
    print("Lower values = more neutral, Higher values = more emotional")


def example_emotion_blending():
    """
    Example demonstrating emotion blending/interpolation for Hindi.

    Blend multiple emotions by specifying weights that sum to any positive value
    (they will be normalized internally).
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for Hindi emotion blending examples...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Load merged checkpoint
    if Path(MERGED_CHECKPOINT).exists():
        model.load_emotion_checkpoint(MERGED_CHECKPOINT)

    # Test cases for emotion blending with Hindi texts
    blend_examples = [
        (
            "मुझे खुशी है कि तुम यहाँ हो, लेकिन दुख है कि तुम्हें जाना होगा।",
            {"happy": 0.5, "sad": 0.5},
            "Bittersweet (50% happy + 50% sad)",
            "hindi_blend_bittersweet.wav"
        ),
        (
            "मुझे विश्वास नहीं हो रहा कि यह हो रहा है!",
            {"excited": 0.6, "fearful": 0.4},
            "Nervous excitement (60% excited + 40% fearful)",
            "hindi_blend_nervous_excitement.wav"
        ),
        (
            "यह बिल्कुल अविश्वसनीय खबर है!",
            {"surprised": 0.7, "happy": 0.3},
            "Pleasant surprise (70% surprised + 30% happy)",
            "hindi_blend_pleasant_surprise.wav"
        ),
        (
            "मैं इस स्थिति से बहुत थक गया हूं।",
            {"sad": 0.4, "angry": 0.6},
            "Frustrated sadness (40% sad + 60% angry)",
            "hindi_blend_frustrated.wav"
        ),
        (
            "सब कुछ ठीक हो जाएगा।",
            {"calm": 0.7, "happy": 0.3},
            "Reassuring calm (70% calm + 30% happy)",
            "hindi_blend_reassuring.wav"
        ),
    ]

    print("\nGenerating Hindi speech with blended emotions")
    print("=" * 70)

    for text, emotion_blend, description, filename in blend_examples:
        print(f"\nText: {text}")
        print(f"Blend: {description}")
        print(f"Weights: {emotion_blend}")

        try:
            wav = model.generate(
                text=text,
                language_id="hi",
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
    print("Done! Generated blended emotion samples in Hindi.")
    print("Emotion blending allows for more nuanced emotional expressions.")


def example_context_appropriate_emotions(use_finetuned=True, checkpoint_path=None):
    """
    Example testing emotions with context-appropriate Hindi texts.
    Each text is matched with an appropriate emotion.
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for context-appropriate Hindi emotion examples...")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if use_finetuned:
        if checkpoint_path is None:
            checkpoint_path = MERGED_CHECKPOINT

        if Path(checkpoint_path).exists():
            print(f"Loading fine-tuned model from: {checkpoint_path}")
            model.load_emotion_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using base model instead")

    # IESC Hindi fine-tuned emotions
    iesc_emotions = ["angry", "happy", "neutral", "sad", "surprised"]

    # Test texts with context-appropriate emotions (Hindi)
    test_texts = [
        ("नमस्ते! आज आपसे मिलकर बहुत खुशी हुई!", "greeting_happy", "happy", 1.0),
        ("मुझे जो हुआ उसके बारे में बहुत दुख है।", "sad_statement", "sad", 1.0),
        ("मैं इस स्थिति से बिल्कुल नाराज़ हूं!", "angry_statement", "angry", 1.2),
        ("यह मौसम के बारे में एक सामान्य बात है।", "neutral_statement", "neutral", 1.0),
        ("मुझे भविष्य के बारे में बहुत डर लग रहा है।", "fearful_statement", "fearful", 1.0),
        ("वाह! यह अद्भुत खबर है!", "surprised_statement", "surprised", 1.0),
        ("मैं अभी बहुत शांत और स्थिर महसूस कर रहा हूं।", "calm_statement", "calm", 1.0),
        ("यह बिल्कुल घृणित और बुरा है!", "disgusted_statement", "disgusted", 1.0),
        ("मैं इस अवसर के लिए बहुत उत्साहित हूं!", "excited_statement", "excited", 1.2),
        ("क्या आप मुझे सुन सकते हैं? मैं धीरे बोल रहा हूं।", "whisper_statement", "whisper", 1.0),
        ("मैं जोर से चिल्ला रहा हूं!", "shout_statement", "shout", 1.0),
    ]

    print(f"\nTesting emotions with context-appropriate Hindi texts")
    print("=" * 70)

    for text, label, emotion, intensity in test_texts:
        print(f"\nText: {text}")
        print(f"Emotion: {emotion} (intensity={intensity})", end="")
        if emotion in iesc_emotions:
            print(" [IESC fine-tuned]", end="")
        print()

        try:
            wav = model.generate(
                text=text,
                language_id="hi",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"hindi_{emotion}_{label}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Generated emotion-controlled Hindi speech samples.")


def example_iesc_finetuned_emotions(use_finetuned=True, checkpoint_path=None):
    """
    Example testing only the IESC Hindi fine-tuned emotions.
    Fine-tuned emotions: angry, happy, neutral, sad, surprised
    Uses Hindi texts that match the emotion context.
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for IESC fine-tuned emotion examples...")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if use_finetuned:
        if checkpoint_path is None:
            # Use IESC-specific checkpoint or merged checkpoint
            if Path(IESC_CHECKPOINT).exists():
                checkpoint_path = IESC_CHECKPOINT
            else:
                checkpoint_path = MERGED_CHECKPOINT

        if Path(checkpoint_path).exists():
            print(f"Loading fine-tuned model from: {checkpoint_path}")
            model.load_emotion_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using base model instead")

    # IESC Hindi fine-tuned emotions with varied intensities
    test_texts = [
        ("नमस्ते! आज आपसे मिलकर बहुत खुशी हुई!", "happy_greeting", "happy", 1.0),
        ("मुझे इसके बारे में बहुत दुख है।", "sad_statement", "sad", 1.0),
        ("मैं इससे बिल्कुल नाराज़ हूं!", "angry_statement", "angry", 1.2),
        ("यह एक सामान्य बात है।", "neutral_statement", "neutral", 1.0),
        ("वाह! यह अविश्वसनीय है!", "surprised_statement", "surprised", 1.0),
    ]

    print(f"\nTesting IESC Hindi fine-tuned emotions")
    print("Emotions: angry, happy, neutral, sad, surprised")
    print("=" * 70)

    for text, label, emotion, intensity in test_texts:
        print(f"\nText: {text}")
        print(f"Emotion: {emotion} (intensity={intensity}) [IESC fine-tuned]")

        try:
            wav = model.generate(
                text=text,
                language_id="hi",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"hindi_iesc_{emotion}_{label}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Generated IESC fine-tuned Hindi emotion samples.")


def example_compare_intensity_vs_blend():
    """
    Compare emotion intensity vs emotion blending approaches for Hindi.

    Shows how to achieve similar effects using different methods:
    1. Using intensity to scale a single emotion
    2. Using blending to mix with neutral
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for Hindi intensity vs blend comparison...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Load merged checkpoint
    if Path(MERGED_CHECKPOINT).exists():
        model.load_emotion_checkpoint(MERGED_CHECKPOINT)

    text = "यह सूक्ष्म भावनात्मक अभिव्यक्ति का परीक्षण है।"  # "This is a test of subtle emotional expression."
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
                language_id="hi",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                temperature=0.8,
                cfg_weight=0.5,
                **params
            )

            filename = f"hindi_compare_{method}_{description.replace(' ', '_').replace('%', 'pct')}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Done! Compare the outputs to see differences between methods.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hindi TTS with Emotion Control Examples")
    parser.add_argument("--example", type=str, default="all_emotions",
                       choices=["basic", "all_emotions", "intensity", "blend",
                               "context", "iesc", "compare"],
                       help="Which example to run")
    parser.add_argument("--finetuned", action="store_true", default=True,
                       help="Use fine-tuned model")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to fine-tuned checkpoint (default: merged checkpoint)")

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
    elif args.example == "iesc":
        example_iesc_finetuned_emotions(use_finetuned=args.finetuned,
                                        checkpoint_path=args.checkpoint)
    elif args.example == "compare":
        example_compare_intensity_vs_blend()
