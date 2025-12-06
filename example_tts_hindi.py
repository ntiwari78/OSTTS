#!/usr/bin/env python3
"""
Example: Hindi TTS with Chatterbox Multilingual Model

This script demonstrates how to use the Chatterbox multilingual model
for Hindi text-to-speech synthesis.
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
    device = "mps"
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
    # Load multilingual model
    print("Loading Chatterbox Multilingual Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Model loaded successfully!")

    # Hindi text samples
    texts = [
        # Greeting
        ("नमस्ते, मेरा नाम चैटरबॉक्स है।", "hindi_greeting.wav"),
        # Introduction
        ("यह एक बहुभाषी टेक्स्ट-टू-स्पीच मॉडल है।", "hindi_intro.wav"),
        # Question
        ("आप कैसे हैं?", "hindi_howru.wav"),
        # Weather
        ("आज मौसम बहुत अच्छा है।", "hindi_weather.wav"),
        # Technology
        ("यह मॉडल 23 भाषाओं का समर्थन करता है।", "hindi_tech.wav"),
        # Longer sentence
        ("भारत विविधता में एकता का देश है, यहाँ विभिन्न भाषाएँ, संस्कृतियाँ और परंपराएँ हैं।", "hindi_india.wav"),
    ]

    # Optional: Reference audio for voice cloning
    AUDIO_PROMPT_PATH = "Neha_1.wav"  # Change to your reference audio file
    audio_prompt = None
    if Path(AUDIO_PROMPT_PATH).exists():
        audio_prompt = AUDIO_PROMPT_PATH
        print(f"Using reference audio: {AUDIO_PROMPT_PATH}")
    else:
        print(f"No reference audio found at {AUDIO_PROMPT_PATH}, using default voice")

    # Generate speech for each text
    print("\nGenerating Hindi speech...")
    print("=" * 50)

    for text, output_file in texts:
        print(f"\nText: {text}")

        # Generate audio
        wav = model.generate(
            text=text,
            language_id="hi",       # Hindi language code
            audio_prompt_path=audio_prompt,
            exaggeration=0.5,       # Emotion intensity (0.0 to 1.0)
            temperature=0.8,        # Sampling temperature (higher = more varied)
            cfg_weight=0.5,         # Classifier-free guidance (0.0 to 1.0)
        )

        # Save output
        save_audio(output_file, wav, model.sr)

    print("\n" + "=" * 50)
    print("Done! Generated audio files:")
    for _, filename in texts:
        print(f"  - {filename}")


def example_with_custom_voice():
    """
    Example showing voice cloning with a custom reference audio.
    """
    AUDIO_PROMPT_PATH = "YOUR_REFERENCE_AUDIO.wav"

    if not Path(AUDIO_PROMPT_PATH).exists():
        print(f"Reference audio not found: {AUDIO_PROMPT_PATH}")
        print("Please provide a valid audio file for voice cloning.")
        return

    print("Loading model with custom voice...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Hindi text
    text = "यह आपकी आवाज़ में हिंदी भाषण उत्पन्न करने का एक उदाहरण है।"

    # Generate with custom voice
    print(f"Generating: {text}")
    wav = model.generate(
        text=text,
        language_id="hi",
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=0.5,
        temperature=0.8,
        cfg_weight=0.5,
    )

    save_audio("hindi_custom_voice.wav", wav, model.sr)
    print("Generated audio with custom voice!")


def example_with_different_emotions():
    """
    Example showing different emotion/exaggeration levels.
    """
    print("Loading model for emotion examples...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    text = "यह एक परीक्षण है।"

    # Different exaggeration levels
    exaggerations = [
        (0.0, "hindi_neutral.wav"),      # Neutral
        (0.5, "hindi_moderate.wav"),     # Moderate emotion
        (1.0, "hindi_expressive.wav"),   # Very expressive
    ]

    print("\nGenerating speech with different emotion levels...")
    for exag, filename in exaggerations:
        print(f"Exaggeration: {exag} -> {filename}")
        wav = model.generate(
            text=text,
            language_id="hi",
            exaggeration=exag,
            temperature=0.8,
            cfg_weight=0.5,
        )
        save_audio(filename, wav, model.sr)

    print("Done! Compare the different emotion levels.")


if __name__ == "__main__":
    # Run main example
    main()

    # Uncomment to try other examples:
    # example_with_custom_voice()
    # example_with_different_emotions()
