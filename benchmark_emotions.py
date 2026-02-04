#!/usr/bin/env python3
"""
benchmark_emotions.py

Complete benchmark script for emotion TTS validation.
Generates audio, analyzes features, and computes metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import soundfile as sf
import json
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Benchmark test cases
BENCHMARK_CASES = {
    "basic_emotions": [
        # (text, emotion, intensity, description)
        ("This is a test of emotion control.", "neutral", 1.0, "baseline"),
        ("I am so happy to see you today!", "happy", 1.0, "positive"),
        ("I feel really sad about this news.", "sad", 1.0, "negative"),
        ("I am absolutely furious right now!", "angry", 1.0, "high_arousal"),
    ],
    "intensity_test": [
        ("I am excited about this!", "happy", 0.0, "zero_intensity"),
        ("I am excited about this!", "happy", 0.5, "half_intensity"),
        ("I am excited about this!", "happy", 1.0, "full_intensity"),
        ("I am excited about this!", "happy", 1.5, "high_intensity"),
    ],
    "blend_test": [],
}

def run_benchmark(
    checkpoint_path: str,
    output_dir: str,
    device: str = "auto",
    audio_prompt: str = None,
):
    """
    Run complete emotion benchmark.

    Args:
        checkpoint_path: Path to emotion checkpoint
        output_dir: Directory to save generated audio
        device: Device to use
        audio_prompt: Optional reference audio for voice cloning
    """
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "cpu"  # MPS has issues
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Load emotion checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading emotion checkpoint: {checkpoint_path}")
        model.load_emotion_checkpoint(checkpoint_path)
    else:
        print("Warning: No checkpoint loaded, using base model")

    # Get supported emotions
    if hasattr(model, 'get_supported_emotions'):
        supported_emotions = model.get_supported_emotions()
        print(f"Supported emotions: {supported_emotions}")
    else:
        supported_emotions = ["neutral", "happy", "sad", "angry", "excited", "calm", "surprised", "fearful", "disgusted"]

    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "device": device,
        "audio_prompt": audio_prompt,
        "supported_emotions": supported_emotions,
        "test_cases": [],
    }

    # Run basic emotion tests
    print("\n" + "=" * 60)
    print("BASIC EMOTION TESTS")
    print("=" * 60)

    for text, emotion, intensity, desc in BENCHMARK_CASES["basic_emotions"]:
        # Skip unsupported emotions
        if emotion not in supported_emotions:
            print(f"\nSkipping unsupported emotion: {emotion}")
            continue

        print(f"\nGenerating: {emotion} ({desc})")
        print(f"  Text: {text}")

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=audio_prompt,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"benchmark_{emotion}_{desc}.wav"
            filepath = output_dir / filename

            # Save audio
            if isinstance(wav, torch.Tensor):
                wav_np = wav.cpu().numpy().squeeze()
            else:
                wav_np = np.array(wav).squeeze()
            sf.write(filepath, wav_np, model.sr)

            duration = len(wav_np) / model.sr
            print(f"  Saved: {filename} ({duration:.2f}s)")

            results["test_cases"].append({
                "type": "basic",
                "text": text,
                "emotion": emotion,
                "intensity": intensity,
                "description": desc,
                "file": filename,
                "duration": duration,
                "status": "success",
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results["test_cases"].append({
                "type": "basic",
                "text": text,
                "emotion": emotion,
                "intensity": intensity,
                "description": desc,
                "status": "error",
                "error": str(e),
            })

    # Run intensity tests
    print("\n" + "=" * 60)
    print("INTENSITY VARIATION TESTS")
    print("=" * 60)

    for text, emotion, intensity, desc in BENCHMARK_CASES["intensity_test"]:
        if emotion not in supported_emotions:
            print(f"\nSkipping unsupported emotion: {emotion}")
            continue

        print(f"\nGenerating: {emotion} at intensity {intensity}")

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=audio_prompt,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"benchmark_intensity_{emotion}_{intensity:.1f}.wav"
            filepath = output_dir / filename

            if isinstance(wav, torch.Tensor):
                wav_np = wav.cpu().numpy().squeeze()
            else:
                wav_np = np.array(wav).squeeze()
            sf.write(filepath, wav_np, model.sr)

            duration = len(wav_np) / model.sr
            print(f"  Saved: {filename} ({duration:.2f}s)")

            results["test_cases"].append({
                "type": "intensity",
                "text": text,
                "emotion": emotion,
                "intensity": intensity,
                "description": desc,
                "file": filename,
                "duration": duration,
                "status": "success",
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results["test_cases"].append({
                "type": "intensity",
                "text": text,
                "emotion": emotion,
                "intensity": intensity,
                "description": desc,
                "status": "error",
                "error": str(e),
            })

    # Run blend tests
    print("\n" + "=" * 60)
    print("EMOTION BLENDING TESTS")
    print("=" * 60)

    for text, blend, desc in BENCHMARK_CASES["blend_test"]:
        print(f"\nGenerating blend: {desc}")
        print(f"  Weights: {blend}")

        try:
            wav = model.generate(
                text=text,
                language_id="en",
                audio_prompt_path=audio_prompt,
                emotion_blend=blend,
                temperature=0.8,
                cfg_weight=0.5,
            )

            filename = f"benchmark_blend_{desc}.wav"
            filepath = output_dir / filename

            if isinstance(wav, torch.Tensor):
                wav_np = wav.cpu().numpy().squeeze()
            else:
                wav_np = np.array(wav).squeeze()
            sf.write(filepath, wav_np, model.sr)

            duration = len(wav_np) / model.sr
            print(f"  Saved: {filename} ({duration:.2f}s)")

            results["test_cases"].append({
                "type": "blend",
                "text": text,
                "emotion_blend": blend,
                "description": desc,
                "file": filename,
                "duration": duration,
                "status": "success",
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results["test_cases"].append({
                "type": "blend",
                "text": text,
                "emotion_blend": blend,
                "description": desc,
                "status": "error",
                "error": str(e),
            })

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")

    # Print summary
    successful = sum(1 for tc in results["test_cases"] if tc["status"] == "success")
    total = len(results["test_cases"])
    print(f"\nBenchmark complete: {successful}/{total} tests successful")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run emotion TTS benchmark")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/emotion_lora_combined_v07/checkpoint_epoch_5.pt",
                       help="Path to emotion checkpoint")
    parser.add_argument("--output_dir", type=str, default="benchmark_output/combined_v07/audio",
                       help="Output directory for generated audio")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, cpu)")
    parser.add_argument("--audio_prompt", type=str, default=None,
                       help="Reference audio for voice cloning")

    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        audio_prompt=args.audio_prompt,
    )
