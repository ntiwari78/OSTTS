"""
Ensemble Emotion Evaluator (Phase 4)

This module provides multi-model ensemble evaluation for emotion recognition,
combining predictions from multiple SER models for more robust results.

Key components:
1. EnsembleEmotionEvaluator: Weighted voting across SER models
2. ModelAgreementAnalyzer: Analyzes agreement between models
3. ConfidenceCalibrator: Calibrates confidence scores

The ensemble approach:
- Combines predictions from emotion2vec, wav2vec2, and other models
- Uses weighted voting based on model accuracy
- Provides confidence calibration for more reliable scores
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field
import json

import numpy as np


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    predicted: str
    confidence: float
    agreement_score: float
    individual_predictions: Dict[str, Dict[str, Any]]
    all_votes: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "predicted": self.predicted,
            "confidence": self.confidence,
            "agreement_score": self.agreement_score,
            "individual_predictions": self.individual_predictions,
            "all_votes": self.all_votes,
        }


class EnsembleEmotionEvaluator:
    """
    Ensemble of multiple SER models for robust evaluation.

    Combines predictions using weighted voting where weights
    are based on model accuracy and confidence.

    Example:
        >>> evaluator = EnsembleEmotionEvaluator()
        >>> result = evaluator.classify("audio.wav")
        >>> print(f"Predicted: {result.predicted} ({result.confidence:.2%})")
    """

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.2,
        use_calibration: bool = True,
    ):
        """
        Args:
            model_weights: Weights for each model (default: based on accuracy)
            min_confidence: Minimum confidence to include prediction
            use_calibration: Whether to calibrate confidence scores
        """
        self.evaluators: Dict[str, Any] = {}
        self.model_weights = model_weights or {
            "emotion2vec": 0.5,   # Best accuracy
            "dpngtm": 0.3,        # Good for happy/surprised
            "ehcalabres": 0.2,    # Backup
        }
        self.min_confidence = min_confidence
        self.use_calibration = use_calibration

        # Confidence calibration factors (learned from benchmarks)
        self.calibration_factors = {
            "emotion2vec": 0.9,  # Usually well-calibrated
            "dpngtm": 1.2,       # Often under-confident
            "ehcalabres": 1.5,   # Usually under-confident
        }

    def load_evaluators(self) -> bool:
        """
        Load all SER evaluators.

        Returns:
            True if at least one evaluator loaded successfully
        """
        loaded = 0

        # Try to import evaluators
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[5]))

            # Try emotion2vec
            try:
                from benchmark_llm_emotions import Emotion2VecEvaluator
                self.evaluators["emotion2vec"] = Emotion2VecEvaluator(model_size="base")
                if self.evaluators["emotion2vec"].load_model():
                    loaded += 1
                    print("Loaded emotion2vec evaluator")
                else:
                    del self.evaluators["emotion2vec"]
            except Exception as e:
                print(f"Could not load emotion2vec: {e}")

            # Try dpngtm
            try:
                from benchmark_llm_emotions import DpngtmEvaluator
                self.evaluators["dpngtm"] = DpngtmEvaluator()
                if self.evaluators["dpngtm"].load_model():
                    loaded += 1
                    print("Loaded dpngtm evaluator")
                else:
                    del self.evaluators["dpngtm"]
            except Exception as e:
                print(f"Could not load dpngtm: {e}")

            # Try ehcalabres
            try:
                from benchmark_llm_emotions import Wav2Vec2EhcalabresEvaluator
                self.evaluators["ehcalabres"] = Wav2Vec2EhcalabresEvaluator()
                if self.evaluators["ehcalabres"].load_model():
                    loaded += 1
                    print("Loaded ehcalabres evaluator")
                else:
                    del self.evaluators["ehcalabres"]
            except Exception as e:
                print(f"Could not load ehcalabres: {e}")

        except Exception as e:
            print(f"Failed to import evaluators: {e}")

        print(f"Loaded {loaded} evaluators")
        return loaded > 0

    def calibrate_confidence(
        self,
        confidence: float,
        model_name: str,
    ) -> float:
        """Calibrate confidence score based on model characteristics."""
        if not self.use_calibration:
            return confidence

        factor = self.calibration_factors.get(model_name, 1.0)
        calibrated = confidence * factor

        # Clamp to valid range
        return min(1.0, max(0.0, calibrated))

    def classify(self, audio_path: str) -> EnsemblePrediction:
        """
        Classify emotion using ensemble voting.

        Args:
            audio_path: Path to audio file

        Returns:
            EnsemblePrediction with combined result
        """
        if not self.evaluators:
            self.load_evaluators()

        if not self.evaluators:
            return EnsemblePrediction(
                predicted="neutral",
                confidence=0.0,
                agreement_score=0.0,
                individual_predictions={},
                all_votes={},
            )

        # Collect predictions from all models
        individual_predictions = {}
        votes: Dict[str, float] = defaultdict(float)

        for model_name, evaluator in self.evaluators.items():
            try:
                result = evaluator.classify(audio_path)

                if result.get("status") == "success":
                    predicted = result.get("predicted", "neutral").lower()
                    confidence = result.get("confidence", 0.5)

                    # Calibrate confidence
                    calibrated_conf = self.calibrate_confidence(confidence, model_name)

                    individual_predictions[model_name] = {
                        "predicted": predicted,
                        "confidence": confidence,
                        "calibrated_confidence": calibrated_conf,
                    }

                    # Add weighted vote
                    if calibrated_conf >= self.min_confidence:
                        weight = self.model_weights.get(model_name, 0.1)
                        votes[predicted] += weight * calibrated_conf

            except Exception as e:
                individual_predictions[model_name] = {"error": str(e)}

        # Find winner
        if votes:
            best_emotion = max(votes.keys(), key=lambda k: votes[k])
            total_weight = sum(votes.values())
            confidence = votes[best_emotion] / total_weight if total_weight > 0 else 0.0
        else:
            best_emotion = "neutral"
            confidence = 0.0

        # Compute agreement score
        num_models = len(individual_predictions)
        if num_models > 0:
            agreeing = sum(
                1 for p in individual_predictions.values()
                if p.get("predicted", "").lower() == best_emotion
            )
            agreement_score = agreeing / num_models
        else:
            agreement_score = 0.0

        return EnsemblePrediction(
            predicted=best_emotion,
            confidence=confidence,
            agreement_score=agreement_score,
            individual_predictions=individual_predictions,
            all_votes=dict(votes),
        )

    def classify_batch(
        self,
        audio_paths: List[str],
        show_progress: bool = True,
    ) -> List[EnsemblePrediction]:
        """
        Classify multiple audio files.

        Args:
            audio_paths: List of audio file paths
            show_progress: Whether to show progress bar

        Returns:
            List of EnsemblePrediction results
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc="Classifying")
            except ImportError:
                iterator = audio_paths
        else:
            iterator = audio_paths

        for audio_path in iterator:
            result = self.classify(audio_path)
            results.append(result)

        return results

    def evaluate_accuracy(
        self,
        audio_paths: List[str],
        expected_emotions: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble accuracy on labeled data.

        Args:
            audio_paths: List of audio file paths
            expected_emotions: List of expected emotion labels

        Returns:
            Dictionary with accuracy metrics
        """
        predictions = self.classify_batch(audio_paths)

        correct = 0
        total = len(predictions)
        per_emotion: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

        for pred, expected in zip(predictions, expected_emotions):
            expected_lower = expected.lower()
            predicted_lower = pred.predicted.lower()

            per_emotion[expected_lower]["total"] += 1

            if predicted_lower == expected_lower:
                correct += 1
                per_emotion[expected_lower]["correct"] += 1

        accuracy = correct / total if total > 0 else 0.0

        # Per-emotion accuracy
        emotion_accuracy = {}
        for emotion, stats in per_emotion.items():
            emotion_accuracy[emotion] = (
                stats["correct"] / stats["total"]
                if stats["total"] > 0 else 0.0
            )

        return {
            "overall_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_emotion_accuracy": emotion_accuracy,
            "per_emotion_stats": dict(per_emotion),
        }


class ModelAgreementAnalyzer:
    """
    Analyzes agreement patterns between SER models.

    Useful for understanding model strengths and weaknesses.
    """

    def __init__(self, evaluator: EnsembleEmotionEvaluator):
        self.evaluator = evaluator

    def analyze_agreement(
        self,
        audio_paths: List[str],
        expected_emotions: List[str],
    ) -> Dict[str, Any]:
        """
        Analyze agreement patterns across models.

        Returns:
            Dictionary with agreement statistics
        """
        predictions = self.evaluator.classify_batch(audio_paths)

        # Agreement matrix
        model_names = list(self.evaluator.evaluators.keys())
        agreement_matrix = {
            m1: {m2: 0 for m2 in model_names}
            for m1 in model_names
        }
        total_comparisons = {m: 0 for m in model_names}

        # Model-specific accuracy
        model_accuracy = {m: {"correct": 0, "total": 0} for m in model_names}

        for pred, expected in zip(predictions, expected_emotions):
            expected_lower = expected.lower()

            for m1, p1 in pred.individual_predictions.items():
                if "error" in p1:
                    continue

                pred1 = p1.get("predicted", "").lower()
                model_accuracy[m1]["total"] += 1
                if pred1 == expected_lower:
                    model_accuracy[m1]["correct"] += 1

                for m2, p2 in pred.individual_predictions.items():
                    if "error" in p2:
                        continue

                    pred2 = p2.get("predicted", "").lower()
                    if pred1 == pred2:
                        agreement_matrix[m1][m2] += 1
                    total_comparisons[m1] += 1

        # Normalize agreement matrix
        for m1 in model_names:
            for m2 in model_names:
                if total_comparisons[m1] > 0:
                    agreement_matrix[m1][m2] /= (total_comparisons[m1] / len(model_names))

        # Compute model accuracies
        accuracies = {}
        for m, stats in model_accuracy.items():
            accuracies[m] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        return {
            "agreement_matrix": agreement_matrix,
            "model_accuracies": accuracies,
            "model_stats": model_accuracy,
        }


# =============================================================================
# Confidence Calibration
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibrates confidence scores using validation data.

    Uses temperature scaling or Platt scaling to improve
    confidence calibration.
    """

    def __init__(self, method: str = "temperature"):
        """
        Args:
            method: Calibration method ("temperature" or "platt")
        """
        self.method = method
        self.temperature = 1.0
        self.platt_a = 1.0
        self.platt_b = 0.0

    def fit(
        self,
        confidences: List[float],
        correct: List[bool],
    ):
        """
        Fit calibration parameters.

        Args:
            confidences: Raw confidence scores
            correct: Whether each prediction was correct
        """
        import numpy as np

        confidences = np.array(confidences)
        correct = np.array(correct, dtype=float)

        if self.method == "temperature":
            # Find temperature that minimizes calibration error
            best_temp = 1.0
            best_ece = float('inf')

            for temp in np.linspace(0.5, 2.0, 31):
                calibrated = self._apply_temperature(confidences, temp)
                ece = self._expected_calibration_error(calibrated, correct)
                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp

            self.temperature = best_temp

        elif self.method == "platt":
            # Platt scaling: logistic regression
            from sklearn.linear_model import LogisticRegression

            log_odds = np.log(confidences / (1 - confidences + 1e-8))
            lr = LogisticRegression()
            lr.fit(log_odds.reshape(-1, 1), correct)
            self.platt_a = lr.coef_[0][0]
            self.platt_b = lr.intercept_[0]

    def calibrate(self, confidence: float) -> float:
        """Calibrate a single confidence score."""
        if self.method == "temperature":
            return self._apply_temperature(np.array([confidence]), self.temperature)[0]
        elif self.method == "platt":
            log_odds = np.log(confidence / (1 - confidence + 1e-8))
            calibrated_odds = self.platt_a * log_odds + self.platt_b
            return 1 / (1 + np.exp(-calibrated_odds))
        return confidence

    def _apply_temperature(
        self,
        confidences: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Apply temperature scaling."""
        # Convert confidence to logits
        logits = np.log(confidences / (1 - confidences + 1e-8))
        # Scale by temperature
        scaled = logits / temperature
        # Convert back to probability
        return 1 / (1 + np.exp(-scaled))

    def _expected_calibration_error(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = correct[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)

        return ece / len(confidences)


# =============================================================================
# Factory Functions
# =============================================================================

def create_ensemble_evaluator(
    weights: Optional[Dict[str, float]] = None,
) -> EnsembleEmotionEvaluator:
    """Create an ensemble evaluator with default settings."""
    return EnsembleEmotionEvaluator(model_weights=weights)


def evaluate_with_ensemble(
    audio_paths: List[str],
    expected_emotions: List[str],
) -> Dict[str, Any]:
    """
    Quick evaluation using ensemble.

    Args:
        audio_paths: List of audio file paths
        expected_emotions: List of expected emotions

    Returns:
        Evaluation results dictionary
    """
    evaluator = create_ensemble_evaluator()
    evaluator.load_evaluators()
    return evaluator.evaluate_accuracy(audio_paths, expected_emotions)
