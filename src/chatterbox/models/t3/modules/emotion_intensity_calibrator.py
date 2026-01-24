"""
Emotion Intensity Calibrator (Phase 2)

This module provides tools to find optimal emotion intensity values
that maximize SER (Speech Emotion Recognition) model recognition.

Key components:
1. IntensityCalibrator: Finds optimal intensity for each emotion
2. CalibrationResult: Stores calibration results for all emotions
3. apply_calibration: Apply calibrated intensities during inference

The calibrator generates audio at various intensity levels and uses
SER models to determine which intensity produces the most recognizable
emotional expression.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import torch
import numpy as np


# =============================================================================
# Calibration Result Data Structures
# =============================================================================

@dataclass
class EmotionCalibration:
    """Calibration result for a single emotion."""
    emotion: str
    recommended_intensity: float
    intensity_std: float
    best_confidence: float
    ser_accuracy: float
    samples_tested: int
    intensity_scores: Dict[float, float] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """Full calibration result for all emotions."""
    emotions: Dict[str, EmotionCalibration] = field(default_factory=dict)
    checkpoint_name: str = ""
    ser_model: str = "emotion2vec"
    timestamp: str = ""

    def save(self, path: str):
        """Save calibration results to JSON."""
        data = {
            "checkpoint_name": self.checkpoint_name,
            "ser_model": self.ser_model,
            "timestamp": self.timestamp,
            "emotions": {
                name: asdict(cal) for name, cal in self.emotions.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        """Load calibration results from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        result = cls(
            checkpoint_name=data.get("checkpoint_name", ""),
            ser_model=data.get("ser_model", ""),
            timestamp=data.get("timestamp", ""),
        )

        for name, cal_data in data.get("emotions", {}).items():
            result.emotions[name] = EmotionCalibration(**cal_data)

        return result

    def get_intensity(self, emotion: str) -> float:
        """Get recommended intensity for an emotion."""
        if emotion in self.emotions:
            return self.emotions[emotion].recommended_intensity
        return 1.0  # Default

    def summary(self) -> str:
        """Get a summary of calibration results."""
        lines = [
            f"Calibration Results ({self.checkpoint_name})",
            f"SER Model: {self.ser_model}",
            "-" * 50,
            f"{'Emotion':<15} {'Intensity':>10} {'Confidence':>12} {'Accuracy':>10}",
            "-" * 50,
        ]
        for name, cal in sorted(self.emotions.items()):
            lines.append(
                f"{name:<15} {cal.recommended_intensity:>10.2f} "
                f"{cal.best_confidence:>12.2%} {cal.ser_accuracy:>10.1%}"
            )
        return "\n".join(lines)


# =============================================================================
# Intensity Calibrator
# =============================================================================

class IntensityCalibrator:
    """
    Finds optimal intensity for each emotion based on SER recognition.

    The calibrator:
    1. Generates audio at multiple intensity levels
    2. Evaluates each with SER model
    3. Selects intensity that maximizes recognition confidence

    Example:
        >>> calibrator = IntensityCalibrator(tts_model, ser_evaluator)
        >>> result = calibrator.calibrate_all(test_texts)
        >>> print(result.get_intensity("happy"))  # e.g., 1.3
    """

    def __init__(
        self,
        tts_model: Any,
        ser_evaluator: Any,
        intensity_range: tuple = (0.5, 2.0),
        intensity_steps: int = 10,
        device: str = "cuda",
    ):
        """
        Args:
            tts_model: TTS model (ChatterboxMultilingualTTS or similar)
            ser_evaluator: SER evaluator from benchmark_llm_emotions.py
            intensity_range: Range of intensities to test (min, max)
            intensity_steps: Number of intensity values to test
            device: Device for model inference
        """
        self.tts = tts_model
        self.ser = ser_evaluator
        self.intensity_range = intensity_range
        self.intensity_steps = intensity_steps
        self.device = device

        # Generate intensity values to test
        self.intensities = np.linspace(
            intensity_range[0],
            intensity_range[1],
            intensity_steps,
        ).tolist()

    def calibrate_emotion(
        self,
        emotion: str,
        test_texts: List[str],
        voice_sample_path: Optional[str] = None,
    ) -> EmotionCalibration:
        """
        Calibrate intensity for a single emotion.

        Args:
            emotion: Emotion name to calibrate
            test_texts: List of test texts to generate
            voice_sample_path: Optional path to voice sample for cloning

        Returns:
            EmotionCalibration with optimal intensity
        """
        intensity_scores = {}
        intensity_confidences = {}
        intensity_accuracies = {}

        for intensity in self.intensities:
            scores = []
            confidences = []
            correct = 0
            total = 0

            for text in test_texts:
                try:
                    # Generate audio with this intensity
                    audio = self._generate_audio(
                        text=text,
                        emotion=emotion,
                        intensity=intensity,
                        voice_sample_path=voice_sample_path,
                    )

                    if audio is None:
                        continue

                    # Evaluate with SER
                    result = self._evaluate_ser(audio)

                    if result["status"] == "success":
                        predicted = result["predicted"].lower()
                        confidence = result.get("confidence", 0.5)

                        # Check if prediction matches
                        is_correct = predicted == emotion.lower()

                        if is_correct:
                            correct += 1
                            # Score is confidence when correct
                            scores.append(confidence)
                        else:
                            # Negative score when wrong
                            scores.append(-confidence)

                        confidences.append(confidence if is_correct else 0.0)
                        total += 1

                except Exception as e:
                    print(f"Error calibrating {emotion} at intensity {intensity}: {e}")
                    continue

            # Aggregate scores for this intensity
            if scores:
                intensity_scores[intensity] = float(np.mean(scores))
                intensity_confidences[intensity] = float(np.mean(confidences))
                intensity_accuracies[intensity] = correct / total if total > 0 else 0.0
            else:
                intensity_scores[intensity] = -1.0
                intensity_confidences[intensity] = 0.0
                intensity_accuracies[intensity] = 0.0

        # Find best intensity
        if intensity_scores:
            best_intensity = max(intensity_scores.keys(), key=lambda k: intensity_scores[k])
            best_confidence = intensity_confidences.get(best_intensity, 0.0)
            best_accuracy = intensity_accuracies.get(best_intensity, 0.0)

            # Calculate intensity std across good results
            good_intensities = [k for k, v in intensity_scores.items() if v > 0]
            intensity_std = float(np.std(good_intensities)) if len(good_intensities) > 1 else 0.0
        else:
            best_intensity = 1.0
            best_confidence = 0.0
            best_accuracy = 0.0
            intensity_std = 0.0

        return EmotionCalibration(
            emotion=emotion,
            recommended_intensity=best_intensity,
            intensity_std=intensity_std,
            best_confidence=best_confidence,
            ser_accuracy=best_accuracy,
            samples_tested=len(test_texts),
            intensity_scores=intensity_scores,
        )

    def calibrate_all(
        self,
        emotions: List[str],
        test_texts: List[str],
        voice_sample_path: Optional[str] = None,
        checkpoint_name: str = "",
    ) -> CalibrationResult:
        """
        Calibrate all emotions.

        Args:
            emotions: List of emotion names to calibrate
            test_texts: Test texts for calibration
            voice_sample_path: Optional voice sample path
            checkpoint_name: Name of checkpoint being calibrated

        Returns:
            CalibrationResult with all emotion calibrations
        """
        from datetime import datetime

        result = CalibrationResult(
            checkpoint_name=checkpoint_name,
            ser_model=getattr(self.ser, "model_name", "unknown"),
            timestamp=datetime.now().isoformat(),
        )

        for emotion in emotions:
            print(f"Calibrating {emotion}...")
            cal = self.calibrate_emotion(
                emotion=emotion,
                test_texts=test_texts,
                voice_sample_path=voice_sample_path,
            )
            result.emotions[emotion] = cal
            print(f"  -> Recommended intensity: {cal.recommended_intensity:.2f}")
            print(f"  -> Best confidence: {cal.best_confidence:.2%}")

        return result

    def _generate_audio(
        self,
        text: str,
        emotion: str,
        intensity: float,
        voice_sample_path: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Generate audio with specified emotion and intensity."""
        try:
            # This is a simplified interface - adjust based on actual TTS API
            if hasattr(self.tts, 'generate'):
                # Direct generation
                audio = self.tts.generate(
                    text=text,
                    emotion=emotion,
                    emotion_intensity=intensity,
                    audio_prompt_path=voice_sample_path,
                )
            elif hasattr(self.tts, 'synthesize'):
                audio = self.tts.synthesize(
                    text=text,
                    emotion=emotion,
                    intensity=intensity,
                )
            else:
                raise AttributeError("TTS model has no generate or synthesize method")

            return audio

        except Exception as e:
            print(f"Audio generation failed: {e}")
            return None

    def _evaluate_ser(self, audio: torch.Tensor) -> Dict:
        """Evaluate audio with SER model."""
        import tempfile
        import soundfile as sf

        # Save to temp file for SER evaluation
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Convert to numpy and save
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = np.array(audio)

            # Handle batched audio
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            sf.write(temp_path, audio_np, 24000)

            # Evaluate
            result = self.ser.classify(temp_path)

            return result

        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)


# =============================================================================
# Calibration Application
# =============================================================================

def apply_calibration(
    emotion: str,
    base_intensity: float,
    calibration: CalibrationResult,
    blend_factor: float = 0.7,
) -> float:
    """
    Apply calibration to get adjusted intensity.

    Blends the user-specified intensity with the calibrated optimal.

    Args:
        emotion: Emotion name
        base_intensity: User-specified intensity
        calibration: Calibration results
        blend_factor: How much to weight calibration vs user (0=user, 1=calibration)

    Returns:
        Adjusted intensity value
    """
    if emotion not in calibration.emotions:
        return base_intensity

    calibrated = calibration.get_intensity(emotion)

    # Blend user intensity with calibrated
    adjusted = (1 - blend_factor) * base_intensity + blend_factor * calibrated

    # Clamp to reasonable range
    return max(0.1, min(2.5, adjusted))


def load_calibration(
    checkpoint_name: str,
    calibration_dir: str = "calibrations",
) -> Optional[CalibrationResult]:
    """
    Load calibration for a checkpoint if available.

    Args:
        checkpoint_name: Name of the checkpoint
        calibration_dir: Directory containing calibration files

    Returns:
        CalibrationResult or None if not found
    """
    path = Path(calibration_dir) / f"{checkpoint_name}_calibration.json"
    if path.exists():
        return CalibrationResult.load(str(path))
    return None


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for intensity calibration."""
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate emotion intensities")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to TTS checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration.json",
        help="Output path for calibration results",
    )
    parser.add_argument(
        "--emotions",
        type=str,
        nargs="+",
        default=["happy", "sad", "angry", "fearful", "surprised", "neutral"],
        help="Emotions to calibrate",
    )
    parser.add_argument(
        "--intensity-steps",
        type=int,
        default=10,
        help="Number of intensity levels to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EMOTION INTENSITY CALIBRATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Emotions: {args.emotions}")
    print(f"Intensity steps: {args.intensity_steps}")
    print()

    # Note: This requires actual model loading - implement as needed
    print("Note: This script requires TTS and SER models to be loaded.")
    print("See the docstrings for usage examples.")

    # Example test texts (in production, use more diverse texts)
    test_texts = [
        "This is a test of emotional speech synthesis.",
        "I am feeling this emotion very strongly right now.",
        "The weather today is quite pleasant.",
    ]

    print(f"\nTest texts: {len(test_texts)}")
    print("\nTo use calibration:")
    print("  1. Load TTS model and SER evaluator")
    print("  2. Create IntensityCalibrator(tts, ser)")
    print("  3. Call calibrator.calibrate_all(emotions, texts)")
    print("  4. Save results with result.save(path)")


if __name__ == "__main__":
    main()
