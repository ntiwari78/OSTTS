#!/usr/bin/env python3
"""
benchmark_llm_emotions.py

Evaluate generated emotions using multiple emotion recognition models.
Uses audio files from benchmark_output folder and compares predictions
with expected emotions.

Models used (based on EMOTION_RECOGNITION_MODELS.md):
1. emotion2vec+ (base) - Primary, best accuracy
2. Dpngtm/wav2vec2-emotion-recognition - Secondary, 8 emotions
3. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition - Tertiary

Usage:
    python benchmark_llm_emotions.py --all
    python benchmark_llm_emotions.py --checkpoint ravdess
    python benchmark_llm_emotions.py --models emotion2vec
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Try importing required libraries
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/torchaudio not available")

# Checkpoint configurations
CHECKPOINT_CONFIGS = {
    "ravdess": {
        "audio_dir": "benchmark_output/ravdess/audio",
        "dataset": "RAVDESS",
        "language": "en",
    },
    "cremad": {
        "audio_dir": "benchmark_output/cremad/audio",
        "dataset": "CREMA-D",
        "language": "en",
    },
    "iesc": {
        "audio_dir": "benchmark_output/iesc/audio",
        "dataset": "IESC",
        "language": "hi",
    },
}

# Emotion mappings from filename patterns to expected emotions
EMOTION_PATTERNS = {
    # Basic emotions
    "basic_neutral": "neutral",
    "basic_happy": "happy",
    "basic_sad": "sad",
    "basic_angry": "angry",
    "basic_fearful": "fearful",
    "basic_surprised": "surprised",
    "basic_calm": "calm",
    "basic_disgusted": "disgusted",
    "basic_excited": "excited",
    # New v0.3 emotions
    "new_sarcastic": "sarcastic",
    "new_bored": "bored",
    "new_affectionate": "affectionate",
    "new_contemptuous": "contemptuous",
    "new_awed": "awed",
    # Intensity tests - map to base emotion
    "intensity_happy": "happy",
    "intensity_angry": "angry",
    "intensity_sad": "sad",
    # Transition tests - map to end emotion
    "transition_neutral_happy": "happy",
    "transition_sad_happy": "happy",
    "transition_calm_excited": "excited",
}

# Model emotion mappings (model output -> our emotion system)
# emotion2vec+ uses bilingual labels like "开心/happy"
EMOTION2VEC_MAPPING = {
    # English labels
    "Angry": "angry",
    "Happy": "happy",
    "Sad": "sad",
    "Neutral": "neutral",
    "Fearful": "fearful",
    "Disgusted": "disgusted",
    "Surprised": "surprised",
    "Other": "neutral",
    "Unknown": "neutral",
    # Lowercase English
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "fearful": "fearful",
    "disgusted": "disgusted",
    "surprised": "surprised",
    "other": "neutral",
    "unknown": "neutral",
    # Bilingual labels (Chinese/English format)
    "生气/angry": "angry",
    "厌恶/disgusted": "disgusted",
    "恐惧/fearful": "fearful",
    "开心/happy": "happy",
    "中立/neutral": "neutral",
    "其他/other": "neutral",
    "难过/sad": "sad",
    "吃惊/surprised": "surprised",
    "<unk>": "neutral",
}

WAV2VEC2_EHCALABRES_MAPPING = {
    "angry": "angry",
    "calm": "calm",
    "disgust": "disgusted",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}

DPNGTM_MAPPING = {
    "angry": "angry",
    "calm": "calm",
    "disgust": "disgusted",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprised",
}


class EmotionEvaluator:
    """Base class for emotion evaluators."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "cpu"  # MPS has issues with some models
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.model_name = "base"

    def load_model(self):
        """Load the model. Override in subclasses."""
        raise NotImplementedError

    def classify(self, audio_path: str) -> Dict:
        """Classify emotion from audio file. Override in subclasses."""
        raise NotImplementedError

    def map_emotion(self, predicted: str) -> str:
        """Map model output to our emotion system. Override in subclasses."""
        return predicted.lower()


class Emotion2VecEvaluator(EmotionEvaluator):
    """emotion2vec+ evaluator - Primary model."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        super().__init__(device)
        self.model_size = model_size
        self.model_name = f"emotion2vec_{model_size}"

    def load_model(self):
        """Load emotion2vec+ model."""
        try:
            from funasr import AutoModel

            # Try different model name formats
            model_names = [
                f"iic/emotion2vec_plus_{self.model_size}",
                f"emotion2vec_plus_{self.model_size}",
                f"iic/emotion2vec_{self.model_size}",
            ]

            last_error = None
            for model_name in model_names:
                try:
                    print(f"Trying to load: {model_name}")
                    self.model = AutoModel(
                        model=model_name,
                        device=self.device,
                        disable_update=True,
                    )
                    print(f"Successfully loaded {model_name}")
                    return True
                except Exception as e:
                    last_error = e
                    print(f"  Failed: {e}")
                    continue

            # If all failed, raise the last error
            raise last_error

        except Exception as e:
            print(f"Failed to load emotion2vec: {e}")
            print("Note: emotion2vec requires funasr and modelscope packages.")
            print("Install with: pip install -U funasr modelscope")
            return False

    def classify(self, audio_path: str, debug: bool = False) -> Dict:
        """Classify emotion using emotion2vec+."""
        try:
            res = self.model.generate(
                input=str(audio_path),
                granularity="utterance",
                extract_embedding=False,
                cache={}
            )

            if debug:
                print(f"DEBUG emotion2vec raw output: {res}")
                print(f"DEBUG output type: {type(res)}")
                if res:
                    print(f"DEBUG first element type: {type(res[0]) if len(res) > 0 else 'empty'}")

            if res and len(res) > 0:
                result = res[0]
                predicted = None
                confidence = 0.0

                # Handle different output formats from emotion2vec+
                if isinstance(result, dict):
                    # Format 1: {"labels": [...], "scores": [...]}
                    if "labels" in result and "scores" in result:
                        labels = result["labels"]
                        scores = result["scores"]
                        if isinstance(scores, list) and len(scores) > 0:
                            max_score = max(scores)
                            top_idx = scores.index(max_score)
                            predicted = labels[top_idx]
                            confidence = max_score
                    # Format 2: {"label": "...", "score": ...}
                    elif "label" in result:
                        predicted = result["label"]
                        confidence = result.get("score", 0.5)
                    # Format 3: {"text": "emotion_name", ...}
                    elif "text" in result:
                        predicted = result["text"]
                        confidence = result.get("score", result.get("confidence", 0.5))
                    # Format 4: {"key": "emotion_name"} - try first value
                    elif len(result) > 0:
                        # Get the first non-None value as prediction
                        for key, val in result.items():
                            if val is not None and key not in ["cache", "key", "feats"]:
                                if isinstance(val, str):
                                    predicted = val
                                    confidence = 0.5
                                    break
                                elif isinstance(val, list) and len(val) > 0:
                                    if isinstance(val[0], str):
                                        predicted = val[0]
                                        confidence = 0.5
                                        break

                elif isinstance(result, str):
                    predicted = result
                    confidence = 0.5

                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    # Could be [(label, score), ...] or [label, ...]
                    first = result[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 2:
                        predicted = str(first[0])
                        confidence = float(first[1]) if len(first) > 1 else 0.5
                    else:
                        predicted = str(first)
                        confidence = 0.5

                if predicted is not None:
                    return {
                        "predicted_raw": predicted,
                        "predicted": self.map_emotion(predicted),
                        "confidence": float(confidence),
                        "status": "success"
                    }
                else:
                    return {"status": "error", "error": f"Could not parse result: {result}"}
            else:
                return {"status": "error", "error": "Empty response"}
        except Exception as e:
            import traceback
            return {"status": "error", "error": f"{str(e)}\n{traceback.format_exc()}"}

    def map_emotion(self, predicted: str) -> str:
        """Map emotion2vec output to our system."""
        predicted_clean = predicted.strip()

        # Direct lookup first
        if predicted_clean in EMOTION2VEC_MAPPING:
            return EMOTION2VEC_MAPPING[predicted_clean]

        # Try extracting English part from bilingual format (e.g., "开心/happy" -> "happy")
        if "/" in predicted_clean:
            english_part = predicted_clean.split("/")[-1].strip().lower()
            if english_part in EMOTION2VEC_MAPPING:
                return EMOTION2VEC_MAPPING[english_part]
            return english_part

        return predicted_clean.lower()


class Wav2Vec2EhcalabresEvaluator(EmotionEvaluator):
    """ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition evaluator."""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.model_name = "wav2vec2_ehcalabres"
        self.processor = None

    def load_model(self):
        """Load wav2vec2 ehcalabres model."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.processor = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded {model_id}")
            return True
        except Exception as e:
            print(f"Failed to load wav2vec2 ehcalabres: {e}")
            return False

    def classify(self, audio_path: str) -> Dict:
        """Classify emotion using wav2vec2."""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # Process
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                pred_label = self.model.config.id2label[pred_id]
                confidence = probs[0, pred_id].item()

            return {
                "predicted_raw": pred_label,
                "predicted": self.map_emotion(pred_label),
                "confidence": float(confidence),
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def map_emotion(self, predicted: str) -> str:
        """Map wav2vec2 ehcalabres output to our system."""
        return WAV2VEC2_EHCALABRES_MAPPING.get(predicted.lower(), predicted.lower())


class DpngtmEvaluator(EmotionEvaluator):
    """Dpngtm/wav2vec2-emotion-recognition evaluator."""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.model_name = "dpngtm"
        self.processor = None

    def load_model(self):
        """Load Dpngtm wav2vec2 model."""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

            model_id = "Dpngtm/wav2vec2-emotion-recognition"
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded {model_id}")
            return True
        except Exception as e:
            print(f"Failed to load Dpngtm model: {e}")
            return False

    def classify(self, audio_path: str) -> Dict:
        """Classify emotion using Dpngtm model."""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # Process
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                pred_label = self.model.config.id2label[pred_id]
                confidence = probs[0, pred_id].item()

            return {
                "predicted_raw": pred_label,
                "predicted": self.map_emotion(pred_label),
                "confidence": float(confidence),
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def map_emotion(self, predicted: str) -> str:
        """Map Dpngtm output to our system."""
        return DPNGTM_MAPPING.get(predicted.lower(), predicted.lower())


def get_expected_emotion(filename: str) -> Optional[str]:
    """Extract expected emotion from filename."""
    for pattern, emotion in EMOTION_PATTERNS.items():
        if pattern in filename:
            return emotion
    return None


def is_compatible_emotion(expected: str, model_emotions: List[str]) -> bool:
    """Check if expected emotion can be recognized by the model."""
    # Map expected to possible model outputs
    emotion_aliases = {
        "happy": ["happy", "happiness"],
        "sad": ["sad", "sadness"],
        "angry": ["angry", "anger"],
        "fearful": ["fearful", "fear"],
        "disgusted": ["disgusted", "disgust"],
        "surprised": ["surprised", "surprise"],
        "calm": ["calm"],
        "neutral": ["neutral"],
        "excited": ["excited", "happy"],  # Some models map excited to happy
        # New emotions likely map to "Other" or closest match
        "sarcastic": ["other", "angry", "neutral"],
        "bored": ["other", "neutral", "sad"],
        "affectionate": ["other", "happy", "neutral"],
        "contemptuous": ["other", "angry", "disgusted"],
        "awed": ["other", "surprised", "happy"],
    }

    expected_aliases = emotion_aliases.get(expected.lower(), [expected.lower()])
    return any(alias in [e.lower() for e in model_emotions] for alias in expected_aliases)


def evaluate_checkpoint(
    checkpoint_name: str,
    evaluators: Dict[str, EmotionEvaluator],
    output_dir: Path,
    debug: bool = False,
) -> Dict:
    """Evaluate all audio files for a checkpoint."""
    config = CHECKPOINT_CONFIGS[checkpoint_name]
    audio_dir = Path(config["audio_dir"])

    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return {"error": f"Audio directory not found: {audio_dir}"}

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {checkpoint_name.upper()} ({config['dataset']})")
    print(f"{'=' * 60}")

    results = {
        "checkpoint": checkpoint_name,
        "dataset": config["dataset"],
        "language": config["language"],
        "timestamp": datetime.now().isoformat(),
        "model_results": {},
        "detailed_results": [],
    }

    # Get all audio files
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    # Evaluate with each model
    for model_name, evaluator in evaluators.items():
        print(f"\n--- {model_name} ---")

        model_stats = {
            "total": 0,
            "correct": 0,
            "by_emotion": defaultdict(lambda: {"total": 0, "correct": 0}),
            "predictions": [],
        }

        for audio_file in audio_files:
            expected = get_expected_emotion(audio_file.name)
            if expected is None:
                continue

            # Classify (pass debug flag for emotion2vec)
            if hasattr(evaluator, 'classify') and 'debug' in evaluator.classify.__code__.co_varnames:
                result = evaluator.classify(str(audio_file), debug=debug and model_name == "emotion2vec_base")
            else:
                result = evaluator.classify(str(audio_file))

            if result["status"] == "success":
                predicted = result["predicted"]

                # Check if correct (with fuzzy matching for similar emotions)
                is_correct = predicted.lower() == expected.lower()

                # Also accept close matches for emotions that models don't support
                if not is_correct:
                    close_matches = {
                        "excited": ["happy"],
                        "sarcastic": ["angry", "neutral", "other"],
                        "bored": ["neutral", "sad", "other"],
                        "affectionate": ["happy", "neutral", "other"],
                        "contemptuous": ["angry", "disgusted", "other"],
                        "awed": ["surprised", "happy", "other"],
                        "fearful": ["fear"],
                        "disgusted": ["disgust"],
                    }
                    if expected.lower() in close_matches:
                        is_correct = predicted.lower() in close_matches[expected.lower()]

                model_stats["total"] += 1
                if is_correct:
                    model_stats["correct"] += 1

                model_stats["by_emotion"][expected]["total"] += 1
                if is_correct:
                    model_stats["by_emotion"][expected]["correct"] += 1

                model_stats["predictions"].append({
                    "file": audio_file.name,
                    "expected": expected,
                    "predicted": predicted,
                    "predicted_raw": result.get("predicted_raw", predicted),
                    "confidence": result.get("confidence", 0),
                    "correct": is_correct,
                })

                status = "✓" if is_correct else "✗"
                print(f"  {status} {audio_file.name}: {expected} -> {predicted} ({result.get('confidence', 0):.2f})")
            else:
                print(f"  ERROR {audio_file.name}: {result.get('error', 'Unknown error')}")

        # Calculate accuracy
        if model_stats["total"] > 0:
            model_stats["accuracy"] = model_stats["correct"] / model_stats["total"]
        else:
            model_stats["accuracy"] = 0

        # Convert defaultdict to dict for JSON
        model_stats["by_emotion"] = dict(model_stats["by_emotion"])

        results["model_results"][model_name] = model_stats

        print(f"\n  Accuracy: {model_stats['accuracy']:.1%} ({model_stats['correct']}/{model_stats['total']})")

    return results


def generate_markdown_report(all_results: Dict, output_path: Path):
    """Generate markdown report from results."""
    report = f"""# Emotion Recognition Benchmark Results (LLM-based)

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report evaluates the generated TTS emotions using multiple emotion recognition models:
1. **emotion2vec+ (base)** - Foundation model, best accuracy, multilingual
2. **wav2vec2 (ehcalabres)** - 7 emotions, well-established
3. **wav2vec2 (Dpngtm)** - 8 emotions, trained on multiple datasets

## Summary

### Model Accuracy by Checkpoint

| Checkpoint | Dataset | emotion2vec | wav2vec2 (ehcalabres) | wav2vec2 (Dpngtm) |
|------------|---------|-------------|----------------------|-------------------|
"""

    for checkpoint, results in all_results.items():
        if "error" in results:
            report += f"| {checkpoint.upper()} | - | ERROR | ERROR | ERROR |\n"
            continue

        dataset = results.get("dataset", "-")
        model_results = results.get("model_results", {})

        e2v_acc = model_results.get("emotion2vec_base", {}).get("accuracy", 0)
        ehc_acc = model_results.get("wav2vec2_ehcalabres", {}).get("accuracy", 0)
        dpn_acc = model_results.get("dpngtm", {}).get("accuracy", 0)

        report += f"| {checkpoint.upper()} | {dataset} | {e2v_acc:.1%} | {ehc_acc:.1%} | {dpn_acc:.1%} |\n"

    # Detailed results per checkpoint
    report += "\n## Detailed Results\n"

    for checkpoint, results in all_results.items():
        if "error" in results:
            continue

        report += f"\n### {checkpoint.upper()} ({results.get('dataset', '-')})\n\n"

        model_results = results.get("model_results", {})

        for model_name, stats in model_results.items():
            report += f"#### {model_name}\n\n"
            report += f"**Overall Accuracy**: {stats.get('accuracy', 0):.1%} ({stats.get('correct', 0)}/{stats.get('total', 0)})\n\n"

            # Per-emotion breakdown
            by_emotion = stats.get("by_emotion", {})
            if by_emotion:
                report += "| Emotion | Correct | Total | Accuracy |\n"
                report += "|---------|---------|-------|----------|\n"

                for emotion, emotion_stats in sorted(by_emotion.items()):
                    total = emotion_stats.get("total", 0)
                    correct = emotion_stats.get("correct", 0)
                    acc = correct / total if total > 0 else 0
                    report += f"| {emotion} | {correct} | {total} | {acc:.1%} |\n"

                report += "\n"

            # Sample predictions
            predictions = stats.get("predictions", [])[:10]
            if predictions:
                report += "**Sample Predictions**:\n\n"
                report += "| File | Expected | Predicted | Confidence | Correct |\n"
                report += "|------|----------|-----------|------------|--------|\n"

                for pred in predictions:
                    status = "✓" if pred.get("correct") else "✗"
                    report += f"| {pred.get('file', '-')} | {pred.get('expected', '-')} | {pred.get('predicted', '-')} | {pred.get('confidence', 0):.2f} | {status} |\n"

                report += "\n"

    # Notes
    report += """## Notes

### Model Limitations

1. **emotion2vec+**: Recognizes 9 emotions (Angry, Disgusted, Fearful, Happy, Neutral, Other, Sad, Surprised, Unknown)
2. **wav2vec2 (ehcalabres)**: Recognizes 7 emotions (angry, calm, disgust, fear, happy, neutral, sad)
3. **wav2vec2 (Dpngtm)**: Recognizes 8 emotions (angry, calm, disgust, fearful, happy, neutral, sad, surprised)

### New Emotions (v0.3)

The new emotions (sarcastic, bored, affectionate, contemptuous, awed) are not directly recognized by these models.
They are mapped to closest emotions:
- **sarcastic** -> angry, neutral, or other
- **bored** -> neutral, sad, or other
- **affectionate** -> happy, neutral, or other
- **contemptuous** -> angry, disgusted, or other
- **awed** -> surprised, happy, or other

### Evaluation Criteria

- **Correct**: Model prediction matches expected emotion (or close match for unsupported emotions)
- **Confidence**: Model's confidence score for the prediction
- **Close matches**: Emotions like "excited" accepted as "happy", "fearful" as "fear", etc.

## References

1. [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec)
2. [wav2vec2 ehcalabres](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
3. [wav2vec2 Dpngtm](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition)
"""

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark emotions using LLM-based recognition")
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["ravdess", "cremad", "iesc", "all"],
        default="all",
        help="Checkpoint to evaluate (default: all)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["emotion2vec", "ehcalabres", "dpngtm", "all"],
        default=["all"],
        help="Models to use (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_output/comparison/BENCHMARK_LLM_RESULTS_V03.md",
        help="Output file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for emotion2vec parsing",
    )
    parser.add_argument(
        "--test_emotion2vec",
        type=str,
        default=None,
        help="Test emotion2vec on a single audio file and print raw output",
    )

    args = parser.parse_args()

    # Test emotion2vec on single file if requested
    if args.test_emotion2vec:
        print("=" * 60)
        print("TESTING EMOTION2VEC OUTPUT FORMAT")
        print("=" * 60)
        print(f"Audio file: {args.test_emotion2vec}")

        e2v = Emotion2VecEvaluator(model_size="base", device=args.device)
        if e2v.load_model():
            result = e2v.classify(args.test_emotion2vec, debug=True)
            print(f"\nParsed result: {result}")
        else:
            print("Failed to load emotion2vec model")
        return

    print("=" * 60)
    print("EMOTION RECOGNITION BENCHMARK (LLM-based)")
    print("=" * 60)

    # Initialize evaluators
    evaluators = {}
    use_all = "all" in args.models

    if use_all or "emotion2vec" in args.models:
        e2v = Emotion2VecEvaluator(model_size="base", device=args.device)
        if e2v.load_model():
            evaluators["emotion2vec_base"] = e2v

    if use_all or "ehcalabres" in args.models:
        ehc = Wav2Vec2EhcalabresEvaluator(device=args.device)
        if ehc.load_model():
            evaluators["wav2vec2_ehcalabres"] = ehc

    if use_all or "dpngtm" in args.models:
        dpn = DpngtmEvaluator(device=args.device)
        if dpn.load_model():
            evaluators["dpngtm"] = dpn

    if not evaluators:
        print("ERROR: No models could be loaded. Please install required dependencies:")
        print("  pip install -U funasr modelscope transformers torchaudio")
        sys.exit(1)

    print(f"\nLoaded {len(evaluators)} models: {list(evaluators.keys())}")

    # Evaluate checkpoints
    checkpoints = (
        ["ravdess", "cremad", "iesc"]
        if args.checkpoint == "all"
        else [args.checkpoint]
    )

    all_results = {}
    for checkpoint in checkpoints:
        results = evaluate_checkpoint(
            checkpoint,
            evaluators,
            Path(args.output).parent,
            debug=args.debug,
        )
        all_results[checkpoint] = results

    # Generate report
    generate_markdown_report(all_results, Path(args.output))

    # Save JSON results
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON results saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    for checkpoint, results in all_results.items():
        if "error" in results:
            print(f"  {checkpoint}: ERROR - {results['error']}")
            continue

        print(f"\n  {checkpoint.upper()}:")
        for model_name, stats in results.get("model_results", {}).items():
            acc = stats.get("accuracy", 0)
            correct = stats.get("correct", 0)
            total = stats.get("total", 0)
            print(f"    {model_name}: {acc:.1%} ({correct}/{total})")


if __name__ == "__main__":
    main()
