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
# Note: MPS has compatibility issues with some operations, so we default to CPU for stability
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    # MPS has issues with voice encoder and some operations, use CPU instead
    device = "cpu"  # Changed from "mps" to "cpu" for better compatibility
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
    # Load multilingual model
    print("Loading Chatterbox Multilingual Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Model loaded successfully!")

    # Hindi text samples
    texts = [
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
    AUDIO_PROMPT_PATH = "Neha_1.wav"

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


def example_with_different_emotions(use_finetuned=False, checkpoint_path=None):
    """
    Example showing different emotion types (happy, sad, angry, etc.).
    Tests all supported emotion types with the same text.
    
    Args:
        use_finetuned: If True, load fine-tuned model from checkpoint
        checkpoint_path: Path to fine-tuned checkpoint (default: checkpoints/emotion_lora/checkpoint_epoch_1.pt)
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for emotion examples...")
    
    if use_finetuned:
        import torch
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/emotion_lora/checkpoint_epoch_1.pt"
        
        print(f"Loading fine-tuned model from: {checkpoint_path}")
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        
        # Load fine-tuned weights
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.t3.load_state_dict(checkpoint["t3_state_dict"], strict=False)
            model.emotion_embeddings.load_state_dict(checkpoint["emotion_embeddings_state_dict"])
            print(f"✓ Loaded fine-tuned weights (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            print(f"⚠️  Warning: Could not load fine-tuned weights: {e}")
            print("   Using base model instead")
    else:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Display supported emotions
    supported_emotions = model.get_supported_emotions()
    print(f"\nSupported emotions: {', '.join(supported_emotions)}")
    
    # IESC fine-tuned emotions (from training data)
    iesc_emotions = ["happy", "sad", "angry", "neutral", "fearful"]
    print(f"\nIESC fine-tuned emotions: {', '.join(iesc_emotions)}")

    # Test text in Hindi
    text = "यह एक परीक्षण है।"

    # Test all emotion types
    # Get all supported emotions from the model
    emotions_to_test = supported_emotions

    print(f"\nGenerating speech with different emotion types for: '{text}'")
    print("=" * 60)

    for emotion in emotions_to_test:
        print(f"\nGenerating with emotion: {emotion}", end="")
        if emotion in iesc_emotions:
            print(" (fine-tuned)", end="")
        print()
        try:
            # If MPS device, temporarily switch to CPU for generation
            original_device = model.device
            if original_device == "mps":
                # Move model to CPU temporarily
                model.t3.to("cpu")
                model.s3gen.to("cpu")
                model.ve.to("cpu")
                model.emotion_embeddings.to("cpu")
            
            wav = model.generate(
                text=text,
                language_id="hi",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,  # Use emotion type
                temperature=0.8,
                cfg_weight=0.5,
            )
            
            # Move back to original device if needed
            if original_device == "mps":
                model.t3.to(original_device)
                model.s3gen.to(original_device)
                model.ve.to(original_device)
                model.emotion_embeddings.to(original_device)
            
            filename = f"hindi_emotion_{emotion}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  ❌ Error generating {emotion}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done! Compare the different emotion types.")
    print(f"Generated {len(emotions_to_test)} audio files with different emotions.")
    if use_finetuned:
        print("\nNote: Emotions marked as 'fine-tuned' were trained on IESC dataset.")
        print("      Other emotions use base model embeddings.")


def example_iesc_emotions_only(use_finetuned=True, checkpoint_path=None):
    """
    Example testing only the IESC fine-tuned emotions (happy, sad, angry, neutral, fearful).
    These emotions were fine-tuned on the Indian Emotional Speech Corpora dataset.
    """
    AUDIO_PROMPT_PATH = "Neha_1.wav"
    print("Loading model for IESC emotion examples...")
    
    if use_finetuned:
        import torch
        if checkpoint_path is None:
            # checkpoint_path = "checkpoints/emotion_lora/checkpoint_epoch_1.pt"
            checkpoint_path = "checkpoints/emotion_lora_v3/checkpoint_epoch_1.pt"
        
        print(f"Loading fine-tuned model from: {checkpoint_path}")
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        
        # Load fine-tuned weights
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.t3.load_state_dict(checkpoint["t3_state_dict"], strict=False)
            model.emotion_embeddings.load_state_dict(checkpoint["emotion_embeddings_state_dict"])
            print(f"✓ Loaded fine-tuned weights (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            print(f"⚠️  Warning: Could not load fine-tuned weights: {e}")
            print("   Using base model instead")
    else:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # IESC emotions (fine-tuned on 600 samples)
    iesc_emotions = ["happy", "sad", "angry", "neutral", "fearful"]
    
    # Test texts in Hindi
    test_texts = [
        ("नमस्ते, मैं खुश हूँ।", "greeting_happy"),
        ("मुझे दुख है।", "sad_statement"),
        ("मैं बहुत गुस्से में हूँ!", "angry_statement"),
        ("यह सामान्य बात है।", "neutral_statement"),
        ("मुझे डर लग रहा है।", "fearful_statement"),
    ]

    print(f"\nTesting IESC fine-tuned emotions with context-appropriate texts")
    print("=" * 60)

    for text, label in test_texts:
        # Match emotion to text context
        if "खुश" in text or "happy" in label:
            emotion = "happy"
        elif "दुख" in text or "sad" in label:
            emotion = "sad"
        elif "गुस्से" in text or "angry" in label:
            emotion = "angry"
        elif "सामान्य" in text or "neutral" in label:
            emotion = "neutral"
        elif "डर" in text or "fearful" in label:
            emotion = "fearful"
        else:
            emotion = "neutral"
        
        print(f"\nText: {text}")
        print(f"Emotion: {emotion} (fine-tuned)")
        try:
            # If MPS device, temporarily switch to CPU for generation
            original_device = model.device
            if original_device == "mps":
                # Move model to CPU temporarily
                model.t3.to("cpu")
                model.s3gen.to("cpu")
                model.ve.to("cpu")
                model.emotion_embeddings.to("cpu")
                temp_device = "cpu"
            else:
                temp_device = original_device
            
            wav = model.generate(
                text=text,
                language_id="hi",
                audio_prompt_path=AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None,
                emotion=emotion,
                temperature=0.8,
                cfg_weight=0.5,
            )
            
            # Move back to original device if needed
            if original_device == "mps":
                model.t3.to(original_device)
                model.s3gen.to(original_device)
                model.ve.to(original_device)
                model.emotion_embeddings.to(original_device)
            
            filename = f"hindi_iesc_{emotion}_{label}.wav"
            save_audio(filename, wav, model.sr)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done! Generated IESC fine-tuned emotion samples.")
    print("These emotions were trained on 600 Hindi audio samples (120 per emotion).")


if __name__ == "__main__":
    # Run main example
    # main()

    # Uncomment to try other examples:
    # example_with_custom_voice()
    
    # Test all emotion types (base model)
    # example_with_different_emotions(use_finetuned=False)
    
    # Test all emotion types with fine-tuned model
    #example_with_different_emotions(use_finetuned=True)
    
    # Test only IESC fine-tuned emotions with context-appropriate texts
    example_iesc_emotions_only(use_finetuned=True)
