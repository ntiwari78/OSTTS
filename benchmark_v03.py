#!/usr/bin/env python3
"""
benchmark_v03.py

Comprehensive benchmark script for v0.3 emotion system.
Tests RAVDESS, CREMA-D, and IESC checkpoints individually.

Usage:
    python benchmark_v03.py --all                    # Benchmark all checkpoints
    python benchmark_v03.py --checkpoint ravdess    # Benchmark specific checkpoint
    python benchmark_v03.py --compare               # Generate comparison report
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import json
import torch
import soundfile as sf
import numpy as np
import librosa
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

# Checkpoint configurations
CHECKPOINT_CONFIGS = {
    "ravdess": {
        "path": "checkpoints/emotion_lora_ravdess/checkpoint_epoch_10.pt",
        "dataset": "RAVDESS",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "cremad": {
        "path": "checkpoints/emotion_lora_cremad/checkpoint_epoch_10.pt",
        "dataset": "CREMA-D",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "iesc": {
        "path": "checkpoints/emotion_lora_iesc/checkpoint_epoch_10.pt",
        "dataset": "IESC",
        "samples": 600,
        "language": "hi",
        "emotions": ["neutral", "happy", "sad", "angry", "surprised"],
    },
    "iesc_ser": {
        "path": "checkpoints/emotion_lora_iesc_ser/checkpoint_epoch_12.pt",
        "dataset": "IESC-SER",
        "samples": 600,
        "language": "hi",
        "emotions": ["neutral", "happy", "sad", "angry", "surprised"],
    },
    "ravdess_ser": {
        "path": "checkpoints/emotion_lora_ravdess_ser/checkpoint_epoch_2.pt",
        "dataset": "RAVDESS-SER",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "cremad_ser": {
        "path": "checkpoints/emotion_lora_cremad_ser/checkpoint_epoch_2.pt",
        "dataset": "CREMA-D-SER",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "v04_full": {
        "path": "checkpoints/emotion_lora_v04_full/checkpoint_epoch_3.pt",
        "dataset": "Combined-V04",
        "samples": 8882,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "excited"],
    },
    "v05_ravdess": {
        "path": "checkpoints/emotion_lora_ravdess_v05/checkpoint_epoch_15.pt",
        "dataset": "RAVDESS-V05",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    # === Experiment checkpoints (V06 improvement sweep) ===
    "exp_a_epoch1": {
        "path": "checkpoints/exp_a_v03_replica/checkpoint_epoch_1.pt",
        "dataset": "ExpA-Epoch1",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_a_epoch2": {
        "path": "checkpoints/exp_a_v03_replica/checkpoint_epoch_2.pt",
        "dataset": "ExpA-Epoch2",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_a_epoch3": {
        "path": "checkpoints/exp_a_v03_replica/checkpoint_epoch_3.pt",
        "dataset": "ExpA-Epoch3",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_b_epoch2": {
        "path": "checkpoints/exp_b_cosine/checkpoint_epoch_2.pt",
        "dataset": "ExpB-Epoch2",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_b_epoch3": {
        "path": "checkpoints/exp_b_cosine/checkpoint_epoch_3.pt",
        "dataset": "ExpB-Epoch3",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_c1_epoch2": {
        "path": "checkpoints/exp_c1_ser02/checkpoint_epoch_2.pt",
        "dataset": "ExpC1-Epoch2",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_c3_epoch2": {
        "path": "checkpoints/exp_c3_ser05/checkpoint_epoch_2.pt",
        "dataset": "ExpC3-Epoch2",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_d2_epoch2": {
        "path": "checkpoints/exp_d2_wd5e3/checkpoint_epoch_2.pt",
        "dataset": "ExpD2-Epoch2",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    "exp_d3_epoch2": {
        "path": "checkpoints/exp_d3_wd01/checkpoint_epoch_2.pt",
        "dataset": "ExpD3-Epoch2",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    # === V06 Production Checkpoint ===
    # Best available: V03-SER epoch 2 (avg ~65.5% emotion2vec, range 62-69%)
    "v06_ravdess": {
        "path": "checkpoints/emotion_lora_ravdess_v06/checkpoint_best.pt",
        "dataset": "RAVDESS-V06",
        "samples": 1440,
        "language": "en",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
    },
    # === CREMA-D V05 (trained with V03-SER recipe, 15 epochs) ===
    "cremad_v05": {
        "path": "checkpoints/emotion_lora_cremad_v05/checkpoint_epoch_15.pt",
        "dataset": "CREMAD-V05",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    # === CREMA-D Experiments ===
    "exp_ca_epoch2": {
        "path": "checkpoints/exp_ca_v03_replica/checkpoint_epoch_2.pt",
        "dataset": "ExpCA-Epoch2",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "exp_ca_epoch3": {
        "path": "checkpoints/exp_ca_v03_replica/checkpoint_epoch_3.pt",
        "dataset": "ExpCA-Epoch3",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "exp_cb_epoch2": {
        "path": "checkpoints/exp_cb_cosine/checkpoint_epoch_2.pt",
        "dataset": "ExpCB-Epoch2",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "exp_cc1_epoch2": {
        "path": "checkpoints/exp_cc1_ser02/checkpoint_epoch_2.pt",
        "dataset": "ExpCC1-Epoch2",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "exp_cc3_epoch2": {
        "path": "checkpoints/exp_cc3_ser05/checkpoint_epoch_2.pt",
        "dataset": "ExpCC3-Epoch2",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "exp_cd2_epoch2": {
        "path": "checkpoints/exp_cd2_wd5e3/checkpoint_epoch_2.pt",
        "dataset": "ExpCD2-Epoch2",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "exp_cd3_epoch2": {
        "path": "checkpoints/exp_cd3_wd01/checkpoint_epoch_2.pt",
        "dataset": "ExpCD3-Epoch2",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
    "exp_ce_epoch2": {
        "path": "checkpoints/exp_ce_balanced/checkpoint_epoch_2.pt",
        "dataset": "ExpCE-Epoch2",
        "samples": 7442,
        "language": "en",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted"],
    },
}

# New emotions in v0.3
NEW_EMOTIONS_V03 = ["sarcastic", "bored", "affectionate", "contemptuous", "awed"]

# Test cases
ENGLISH_TEST_CASES = {
    "basic_emotions": [
        ("This is a test of emotion control.", "neutral", 1.0),
        ("I am so happy to see you today!", "happy", 1.0),
        ("I feel really sad about this news.", "sad", 1.0),
        ("I am absolutely furious right now!", "angry", 1.0),
        ("I am really scared and worried.", "fearful", 1.0),
        ("Wow, I cannot believe this happened!", "surprised", 1.0),
        ("I feel calm and peaceful right now.", "calm", 1.0),
        ("This is absolutely disgusting!", "disgusted", 1.0),
        ("I am so excited about this opportunity!", "excited", 1.0),
    ],
    "new_emotions": [
        ("Oh, how fascinating. Really.", "sarcastic", 1.0),
        ("I suppose this is interesting.", "bored", 1.0),
        ("I care about you so much, darling.", "affectionate", 1.0),
        ("How utterly predictable of you.", "contemptuous", 1.0),
        ("This is absolutely magnificent!", "awed", 1.0),
    ],
    "intensity_tests": [
        ("I am really happy about this!", "happy", 0.0),
        ("I am really happy about this!", "happy", 0.5),
        ("I am really happy about this!", "happy", 1.0),
        ("I am really happy about this!", "happy", 1.5),
        ("I am absolutely furious!", "angry", 0.0),
        ("I am absolutely furious!", "angry", 0.5),
        ("I am absolutely furious!", "angry", 1.0),
        ("I am absolutely furious!", "angry", 1.5),
        ("I feel so sad today.", "sad", 0.0),
        ("I feel so sad today.", "sad", 0.5),
        ("I feel so sad today.", "sad", 1.0),
        ("I feel so sad today.", "sad", 1.5),
    ],
    "transition_tests": [
        ("I started feeling neutral, then became happy!", "neutral", "happy"),
        ("Though I was sad, I'm feeling hopeful now.", "sad", "happy"),
        ("Starting calm but getting more excited!", "calm", "excited"),
    ],
}

HINDI_TEST_CASES = {
    "basic_emotions": [
        ("यह भावना नियंत्रण का परीक्षण है।", "neutral", 1.0),
        ("मैं आज आपसे मिलकर बहुत खुश हूं!", "happy", 1.0),
        ("मुझे इस खबर से बहुत दुख हुआ।", "sad", 1.0),
        ("मैं अभी बहुत गुस्से में हूं!", "angry", 1.0),
        ("वाह, यह आश्चर्यजनक है!", "surprised", 1.0),
    ],
    "intensity_tests": [
        ("मैं बहुत खुश हूं!", "happy", 0.0),
        ("मैं बहुत खुश हूं!", "happy", 0.5),
        ("मैं बहुत खुश हूं!", "happy", 1.0),
        ("मैं बहुत खुश हूं!", "happy", 1.5),
    ],
}


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    test_type: str
    text: str
    emotion: str
    intensity: float
    status: str  # "success" or "error"
    file_path: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    prosody: Optional[Dict] = None


@dataclass
class BenchmarkResults:
    """Complete benchmark results for a checkpoint."""
    timestamp: str
    checkpoint_name: str
    checkpoint_path: str
    dataset: str
    language: str
    device: str
    test_results: Dict[str, List[Dict]] = field(default_factory=dict)
    prosody_analysis: Dict = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)


def extract_prosody(audio_path: str, sr: int = 24000) -> Dict:
    """Extract prosodic features from audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        duration = len(y) / sr

        # Pitch (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = tempo[0] if len(tempo) > 0 else 0

        return {
            "duration_s": float(duration),
            "pitch_mean_hz": float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0,
            "pitch_std_hz": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0,
            "pitch_min_hz": float(np.min(f0_clean)) if len(f0_clean) > 0 else 0,
            "pitch_max_hz": float(np.max(f0_clean)) if len(f0_clean) > 0 else 0,
            "energy_mean": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
            "energy_max": float(np.max(rms)),
            "tempo_bpm": float(tempo),
        }
    except Exception as e:
        return {"error": str(e)}


def run_checkpoint_benchmark(
    checkpoint_name: str,
    output_dir: Path,
    device: str = "auto",
    audio_prompt: Optional[str] = None,
    skip_transitions: bool = False,
    skip_new_emotions: bool = False,
    use_calibration: bool = True,
) -> BenchmarkResults:
    """
    Run benchmark for a single checkpoint.

    Args:
        checkpoint_name: Name of checkpoint (ravdess, cremad, iesc, ravdess_ser, cremad_ser, iesc_ser)
        output_dir: Directory to save results
        device: Device to use
        audio_prompt: Optional reference audio
        skip_transitions: Skip transition tests
        skip_new_emotions: Skip new v0.3 emotion tests

    Returns:
        BenchmarkResults object
    """
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    config = CHECKPOINT_CONFIGS[checkpoint_name]
    checkpoint_path = config["path"]
    language = config["language"]
    supported_emotions = config["emotions"]

    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "cpu"  # MPS has issues
        else:
            device = "cpu"

    print(f"\n{'=' * 60}")
    print(f"BENCHMARKING: {checkpoint_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Language: {language}")
    print(f"Supported emotions: {supported_emotions}")

    # Create output directories
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results
    results = BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
        dataset=config["dataset"],
        language=language,
        device=device,
    )

    # Load model
    print("\nLoading model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Load checkpoint
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_emotion_checkpoint(checkpoint_path)
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        results.summary["error"] = "Checkpoint not found"
        return results

    # Select test cases based on language
    test_cases = HINDI_TEST_CASES if language == "hi" else ENGLISH_TEST_CASES

    all_results = []
    prosody_by_emotion = defaultdict(list)

    # ===== Basic Emotion Tests =====
    print(f"\n--- Basic Emotion Tests ---")
    results.test_results["basic_emotions"] = []

    for text, emotion, intensity in test_cases["basic_emotions"]:
        # Skip if emotion not supported by this checkpoint
        if emotion not in supported_emotions and emotion not in ["excited"]:
            print(f"  Skipping {emotion} (not in {checkpoint_name})")
            continue

        test_id = f"basic_{emotion}_{intensity}"
        print(f"  Generating: {emotion}")

        try:
            wav = model.generate(
                text=text,
                language_id=language,
                audio_prompt_path=audio_prompt,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
                use_calibration=use_calibration,
            )

            filename = f"basic_{emotion}_{intensity}.wav"
            filepath = audio_dir / filename

            if isinstance(wav, torch.Tensor):
                wav_np = wav.cpu().numpy().squeeze()
            else:
                wav_np = np.array(wav).squeeze()

            sf.write(filepath, wav_np, model.sr)
            duration = len(wav_np) / model.sr

            # Extract prosody
            prosody = extract_prosody(str(filepath), model.sr)
            prosody_by_emotion[emotion].append(prosody)

            result = TestResult(
                test_id=test_id,
                test_type="basic",
                text=text,
                emotion=emotion,
                intensity=intensity,
                status="success",
                file_path=str(filepath),
                duration=duration,
                prosody=prosody,
            )
            print(f"    Saved: {filename} ({duration:.2f}s)")

        except Exception as e:
            result = TestResult(
                test_id=test_id,
                test_type="basic",
                text=text,
                emotion=emotion,
                intensity=intensity,
                status="error",
                error=str(e),
            )
            print(f"    ERROR: {e}")

        results.test_results["basic_emotions"].append(asdict(result))
        all_results.append(result)

    # ===== Intensity Tests =====
    print(f"\n--- Intensity Variation Tests ---")
    results.test_results["intensity_tests"] = []

    for text, emotion, intensity in test_cases.get("intensity_tests", []):
        if emotion not in supported_emotions:
            continue

        test_id = f"intensity_{emotion}_{intensity}"
        print(f"  Generating: {emotion} @ intensity={intensity}")

        try:
            wav = model.generate(
                text=text,
                language_id=language,
                audio_prompt_path=audio_prompt,
                emotion=emotion,
                emotion_intensity=intensity,
                temperature=0.8,
                cfg_weight=0.5,
                use_calibration=use_calibration,
            )

            filename = f"intensity_{emotion}_{intensity}.wav"
            filepath = audio_dir / filename

            if isinstance(wav, torch.Tensor):
                wav_np = wav.cpu().numpy().squeeze()
            else:
                wav_np = np.array(wav).squeeze()

            sf.write(filepath, wav_np, model.sr)
            duration = len(wav_np) / model.sr

            prosody = extract_prosody(str(filepath), model.sr)

            result = TestResult(
                test_id=test_id,
                test_type="intensity",
                text=text,
                emotion=emotion,
                intensity=intensity,
                status="success",
                file_path=str(filepath),
                duration=duration,
                prosody=prosody,
            )
            print(f"    Saved: {filename} ({duration:.2f}s)")

        except Exception as e:
            result = TestResult(
                test_id=test_id,
                test_type="intensity",
                text=text,
                emotion=emotion,
                intensity=intensity,
                status="error",
                error=str(e),
            )
            print(f"    ERROR: {e}")

        results.test_results["intensity_tests"].append(asdict(result))
        all_results.append(result)

    # ===== New Emotions (v0.3) =====
    if not skip_new_emotions and language == "en":
        print(f"\n--- New Emotions (v0.3) ---")
        results.test_results["new_emotions"] = []

        for text, emotion, intensity in test_cases.get("new_emotions", []):
            test_id = f"new_{emotion}_{intensity}"
            print(f"  Generating: {emotion}")

            try:
                wav = model.generate(
                    text=text,
                    language_id=language,
                    audio_prompt_path=audio_prompt,
                    emotion=emotion,
                    emotion_intensity=intensity,
                    temperature=0.8,
                    cfg_weight=0.5,
                )

                filename = f"new_{emotion}_{intensity}.wav"
                filepath = audio_dir / filename

                if isinstance(wav, torch.Tensor):
                    wav_np = wav.cpu().numpy().squeeze()
                else:
                    wav_np = np.array(wav).squeeze()

                sf.write(filepath, wav_np, model.sr)
                duration = len(wav_np) / model.sr

                prosody = extract_prosody(str(filepath), model.sr)
                prosody_by_emotion[emotion].append(prosody)

                result = TestResult(
                    test_id=test_id,
                    test_type="new_emotion",
                    text=text,
                    emotion=emotion,
                    intensity=intensity,
                    status="success",
                    file_path=str(filepath),
                    duration=duration,
                    prosody=prosody,
                )
                print(f"    Saved: {filename} ({duration:.2f}s)")

            except Exception as e:
                result = TestResult(
                    test_id=test_id,
                    test_type="new_emotion",
                    text=text,
                    emotion=emotion,
                    intensity=intensity,
                    status="error",
                    error=str(e),
                )
                print(f"    ERROR: {e}")

            results.test_results["new_emotions"].append(asdict(result))
            all_results.append(result)

    # ===== Transition Tests =====
    if not skip_transitions and language == "en":
        print(f"\n--- Emotion Transition Tests ---")
        results.test_results["transition_tests"] = []

        for text, start_emotion, end_emotion in test_cases.get("transition_tests", []):
            if start_emotion not in supported_emotions or end_emotion not in supported_emotions:
                if start_emotion not in ["calm"] and end_emotion not in ["excited", "happy"]:
                    continue

            test_id = f"transition_{start_emotion}_{end_emotion}"
            print(f"  Generating: {start_emotion} -> {end_emotion}")

            try:
                # For transition, we use emotion_blend as approximation
                # (full trajectory support depends on model implementation)
                wav = model.generate(
                    text=text,
                    language_id=language,
                    audio_prompt_path=audio_prompt,
                    emotion=end_emotion,  # Use end emotion
                    emotion_intensity=1.0,
                    temperature=0.8,
                    cfg_weight=0.5,
                )

                filename = f"transition_{start_emotion}_{end_emotion}.wav"
                filepath = audio_dir / filename

                if isinstance(wav, torch.Tensor):
                    wav_np = wav.cpu().numpy().squeeze()
                else:
                    wav_np = np.array(wav).squeeze()

                sf.write(filepath, wav_np, model.sr)
                duration = len(wav_np) / model.sr

                prosody = extract_prosody(str(filepath), model.sr)

                result = TestResult(
                    test_id=test_id,
                    test_type="transition",
                    text=text,
                    emotion=f"{start_emotion}->{end_emotion}",
                    intensity=1.0,
                    status="success",
                    file_path=str(filepath),
                    duration=duration,
                    prosody=prosody,
                )
                print(f"    Saved: {filename} ({duration:.2f}s)")

            except Exception as e:
                result = TestResult(
                    test_id=test_id,
                    test_type="transition",
                    text=text,
                    emotion=f"{start_emotion}->{end_emotion}",
                    intensity=1.0,
                    status="error",
                    error=str(e),
                )
                print(f"    ERROR: {e}")

            results.test_results["transition_tests"].append(asdict(result))
            all_results.append(result)

    # ===== Compute Summary =====
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r.status == "success")
    failed_tests = total_tests - passed_tests

    # Aggregate prosody by emotion
    prosody_summary = {}
    for emotion, prosodies in prosody_by_emotion.items():
        valid_prosodies = [p for p in prosodies if "error" not in p]
        if valid_prosodies:
            prosody_summary[emotion] = {
                "pitch_mean": np.mean([p["pitch_mean_hz"] for p in valid_prosodies]),
                "pitch_std": np.mean([p["pitch_std_hz"] for p in valid_prosodies]),
                "energy_mean": np.mean([p["energy_mean"] for p in valid_prosodies]),
                "tempo": np.mean([p["tempo_bpm"] for p in valid_prosodies]),
                "sample_count": len(valid_prosodies),
            }

    results.prosody_analysis = prosody_summary
    results.summary = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "average_duration": np.mean([r.duration for r in all_results if r.duration]),
    }

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK COMPLETE: {checkpoint_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {results.summary['success_rate']:.1%}")
    print(f"Results saved to: {results_path}")

    return results


def generate_comparison_report(
    results_dir: Path,
    output_path: Path,
) -> str:
    """Generate markdown comparison report."""
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    # Load all results
    all_results = {}
    for checkpoint in CHECKPOINT_CONFIGS.keys():
        results_file = results_dir / checkpoint / "benchmark_results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_results[checkpoint] = json.load(f)

    if not all_results:
        return "No benchmark results found."

    # Generate report
    report = f"""# Emotion TTS Benchmark Results v0.3

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Checkpoint | Dataset | Tests | Passed | Failed | Success Rate |
|------------|---------|-------|--------|--------|--------------|
"""

    for name, results in all_results.items():
        summary = results.get("summary", {})
        report += f"| {name.upper()} | {results.get('dataset', 'N/A')} | "
        report += f"{summary.get('total_tests', 0)} | {summary.get('passed', 0)} | "
        report += f"{summary.get('failed', 0)} | {summary.get('success_rate', 0):.1%} |\n"

    # Prosody comparison
    # Get all checkpoint names for comparison (base and SER variants)
    base_checkpoints = ["ravdess", "cremad", "iesc"]
    ser_checkpoints = ["ravdess_ser", "cremad_ser", "iesc_ser"]
    all_checkpoint_names = base_checkpoints + ser_checkpoints
    
    all_emotions = set()
    for results in all_results.values():
        all_emotions.update(results.get("prosody_analysis", {}).keys())

    # Generate header for tables with all checkpoints
    header_row = "| Emotion |"
    separator_row = "|---------|"
    for checkpoint in all_checkpoint_names:
        display_name = checkpoint.upper().replace("_", "-")
        header_row += f" {display_name} |"
        separator_row += "---------|"
    
    report += """
## Prosody Analysis by Emotion

### Pitch (Mean Hz)

""" + header_row + "\n" + separator_row + "\n"
    
    for emotion in sorted(all_emotions):
        report += f"| {emotion} |"
        for checkpoint in all_checkpoint_names:
            if checkpoint in all_results:
                prosody = all_results[checkpoint].get("prosody_analysis", {}).get(emotion, {})
                pitch = prosody.get("pitch_mean", "-")
                if isinstance(pitch, (int, float)):
                    report += f" {pitch:.1f} |"
                else:
                    report += f" {pitch} |"
            else:
                report += " - |"
        report += "\n"

    # Energy comparison
    report += """
### Energy (Mean RMS)

""" + header_row + "\n" + separator_row + "\n"

    for emotion in sorted(all_emotions):
        report += f"| {emotion} |"
        for checkpoint in all_checkpoint_names:
            if checkpoint in all_results:
                prosody = all_results[checkpoint].get("prosody_analysis", {}).get(emotion, {})
                energy = prosody.get("energy_mean", "-")
                if isinstance(energy, (int, float)):
                    report += f" {energy:.4f} |"
                else:
                    report += f" {energy} |"
            else:
                report += " - |"
        report += "\n"

    # Test details
    report += """
## Test Details

### Basic Emotion Tests
"""

    for checkpoint, results in all_results.items():
        report += f"\n#### {checkpoint.upper()}\n\n"
        basic_tests = results.get("test_results", {}).get("basic_emotions", [])
        if basic_tests:
            report += "| Emotion | Status | Duration | Pitch (Hz) |\n"
            report += "|---------|--------|----------|------------|\n"
            for test in basic_tests:
                emotion = test.get("emotion", "-")
                status = "✓" if test.get("status") == "success" else "✗"
                duration = test.get("duration", 0)
                prosody = test.get("prosody", {})
                pitch = prosody.get("pitch_mean_hz", "-")
                if isinstance(duration, (int, float)):
                    duration_str = f"{duration:.2f}s"
                else:
                    duration_str = str(duration)
                if isinstance(pitch, (int, float)):
                    pitch_str = f"{pitch:.1f}"
                else:
                    pitch_str = str(pitch)
                report += f"| {emotion} | {status} | {duration_str} | {pitch_str} |\n"

    report += """
## Notes

- All benchmarks run with `checkpoint_epoch_10.pt`
- Audio generated at 24kHz sample rate
- Prosody extracted using librosa
- New emotions (v0.3): sarcastic, bored, affectionate, contemptuous, awed
"""

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nComparison report saved to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark v0.3 emotion system")
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["ravdess", "cremad", "iesc", "iesc_ser", "ravdess_ser", "cremad_ser",
                 "v04_full", "v05_ravdess",
                 "exp_a_epoch1", "exp_a_epoch2", "exp_a_epoch3",
                 "exp_b_epoch2", "exp_b_epoch3",
                 "exp_c1_epoch2", "exp_c3_epoch2",
                 "exp_d2_epoch2", "exp_d3_epoch2",
                 "v06_ravdess", "cremad_v05",
                 "exp_ca_epoch2", "exp_ca_epoch3",
                 "exp_cb_epoch2",
                 "exp_cc1_epoch2", "exp_cc3_epoch2",
                 "exp_cd2_epoch2", "exp_cd3_epoch2",
                 "exp_ce_epoch2",
                 "all"],
        default="all",
        help="Checkpoint to benchmark (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_output",
        help="Output directory (default: benchmark_output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--audio_prompt",
        type=str,
        default=None,
        help="Reference audio for voice cloning",
    )
    parser.add_argument(
        "--skip-transitions",
        action="store_true",
        help="Skip transition tests",
    )
    parser.add_argument(
        "--skip-new-emotions",
        action="store_true",
        help="Skip new v0.3 emotion tests",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison report only (no generation)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comparison report after benchmarking",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable intensity calibration during audio generation",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.compare:
        # Only generate comparison report
        generate_comparison_report(
            output_dir,
            output_dir / "comparison" / "BENCHMARK_RESULTS_V03.md",
        )
        return

    # Run benchmarks
    if args.checkpoint == "all":
        checkpoints = ["ravdess", "cremad", "iesc", "ravdess_ser", "cremad_ser", "iesc_ser"]
    else:
        checkpoints = [args.checkpoint]

    use_calibration = not args.no_calibration
    all_results = {}
    for checkpoint in checkpoints:
        results = run_checkpoint_benchmark(
            checkpoint_name=checkpoint,
            output_dir=output_dir / checkpoint,
            device=args.device,
            audio_prompt=args.audio_prompt,
            skip_transitions=args.skip_transitions,
            skip_new_emotions=args.skip_new_emotions,
            use_calibration=use_calibration,
        )
        all_results[checkpoint] = results

    # Generate comparison report if requested
    if args.generate_report or args.checkpoint == "all":
        generate_comparison_report(
            output_dir,
            output_dir / "comparison" / "BENCHMARK_RESULTS_V03.md",
        )

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 60)
    for name, results in all_results.items():
        print(f"  {name}: {results.summary.get('passed', 0)}/{results.summary.get('total_tests', 0)} passed")


if __name__ == "__main__":
    main()
