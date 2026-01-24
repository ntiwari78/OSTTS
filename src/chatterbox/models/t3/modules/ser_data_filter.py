"""
SER-Guided Data Filtering (Phase 3)

This module filters training data to only include samples where
SER (Speech Emotion Recognition) models correctly recognize the emotion.

This ensures the model learns from "clean" examples that already have
recognizable emotional patterns, improving training efficiency.

Key components:
1. SERDataFilter: Filters dataset based on SER model agreement
2. filter_dataset: Standalone function for filtering audio files
3. create_filtered_dataloader: Creates a filtered PyTorch DataLoader
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np


@dataclass
class FilterResult:
    """Result of filtering a dataset."""
    total_samples: int
    kept_samples: int
    filtered_samples: int
    keep_ratio: float
    per_emotion_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    sample_indices: List[int] = field(default_factory=list)

    def summary(self) -> str:
        """Get a summary of filtering results."""
        lines = [
            f"Total samples: {self.total_samples}",
            f"Kept samples: {self.kept_samples} ({self.keep_ratio:.1%})",
            f"Filtered out: {self.filtered_samples}",
            "",
            "Per-emotion breakdown:",
        ]
        for emotion, stats in sorted(self.per_emotion_stats.items()):
            kept = stats.get("kept", 0)
            total = stats.get("total", 0)
            ratio = kept / total if total > 0 else 0
            lines.append(f"  {emotion}: {kept}/{total} ({ratio:.1%})")
        return "\n".join(lines)

    def save(self, path: str):
        """Save filter result to JSON."""
        data = {
            "total_samples": self.total_samples,
            "kept_samples": self.kept_samples,
            "filtered_samples": self.filtered_samples,
            "keep_ratio": self.keep_ratio,
            "per_emotion_stats": self.per_emotion_stats,
            "sample_indices": self.sample_indices,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FilterResult":
        """Load filter result from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class SERDataFilter:
    """
    Filters training data based on SER model recognition.

    Only keeps samples where at least one SER model correctly
    recognizes the labeled emotion.

    Example:
        >>> filter = SERDataFilter(min_agreement=1)
        >>> result = filter.filter_dataset(audio_files, emotions)
        >>> print(f"Kept {result.kept_samples}/{result.total_samples}")
    """

    def __init__(
        self,
        evaluators: Optional[Dict[str, Any]] = None,
        min_agreement: int = 1,
        min_confidence: float = 0.3,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            evaluators: Dictionary of SER evaluators (name -> evaluator)
                       If None, will use default evaluators when needed
            min_agreement: Minimum number of models that must agree (default: 1)
            min_confidence: Minimum confidence for a prediction to count (default: 0.3)
            cache_dir: Directory to cache filter results
        """
        self.evaluators = evaluators or {}
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_evaluators(self):
        """Load default SER evaluators if not already loaded."""
        if self.evaluators:
            return

        try:
            # Import evaluators from benchmark script
            import sys
            from pathlib import Path

            # Try to import from benchmark_llm_emotions
            try:
                from benchmark_llm_emotions import (
                    Emotion2VecEvaluator,
                    DpngtmEvaluator,
                )

                self.evaluators = {
                    "emotion2vec": Emotion2VecEvaluator(model_size="base"),
                    "dpngtm": DpngtmEvaluator(),
                }

                for name, evaluator in self.evaluators.items():
                    print(f"Loading {name} evaluator...")
                    evaluator.load_model()

            except ImportError:
                print("Warning: Could not import evaluators from benchmark_llm_emotions")
                print("SER filtering will be disabled")

        except Exception as e:
            print(f"Warning: Failed to load SER evaluators: {e}")

    def check_sample(
        self,
        audio_path: str,
        expected_emotion: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a sample should be kept.

        Args:
            audio_path: Path to audio file
            expected_emotion: Expected emotion label

        Returns:
            Tuple of (should_keep, details)
        """
        if not self.evaluators:
            # No evaluators loaded, keep all samples
            return True, {"reason": "no_evaluators"}

        votes = 0
        details = {
            "expected": expected_emotion,
            "predictions": {},
        }

        for name, evaluator in self.evaluators.items():
            try:
                result = evaluator.classify(audio_path)

                if result.get("status") == "success":
                    predicted = result.get("predicted", "").lower()
                    confidence = result.get("confidence", 0.0)

                    details["predictions"][name] = {
                        "predicted": predicted,
                        "confidence": confidence,
                    }

                    # Check if prediction matches and confidence is high enough
                    if (predicted == expected_emotion.lower() and
                        confidence >= self.min_confidence):
                        votes += 1

            except Exception as e:
                details["predictions"][name] = {"error": str(e)}

        details["votes"] = votes
        details["keep"] = votes >= self.min_agreement

        return votes >= self.min_agreement, details

    def filter_dataset(
        self,
        audio_files: List[str],
        emotions: List[str],
        show_progress: bool = True,
    ) -> FilterResult:
        """
        Filter a dataset based on SER agreement.

        Args:
            audio_files: List of audio file paths
            emotions: List of corresponding emotion labels
            show_progress: Whether to show progress bar

        Returns:
            FilterResult with filtering statistics
        """
        self.load_evaluators()

        total = len(audio_files)
        kept_indices = []
        per_emotion_stats: Dict[str, Dict[str, int]] = {}

        # Initialize per-emotion stats
        for emotion in set(emotions):
            per_emotion_stats[emotion] = {"total": 0, "kept": 0}

        # Optional progress bar
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    zip(range(total), audio_files, emotions),
                    total=total,
                    desc="Filtering",
                )
            except ImportError:
                iterator = zip(range(total), audio_files, emotions)
        else:
            iterator = zip(range(total), audio_files, emotions)

        for idx, audio_path, emotion in iterator:
            per_emotion_stats[emotion]["total"] += 1

            should_keep, details = self.check_sample(audio_path, emotion)

            if should_keep:
                kept_indices.append(idx)
                per_emotion_stats[emotion]["kept"] += 1

        kept = len(kept_indices)
        filtered = total - kept

        result = FilterResult(
            total_samples=total,
            kept_samples=kept,
            filtered_samples=filtered,
            keep_ratio=kept / total if total > 0 else 0.0,
            per_emotion_stats=per_emotion_stats,
            sample_indices=kept_indices,
        )

        return result

    def filter_torch_dataset(
        self,
        dataset: Dataset,
        audio_key: str = "audio_path",
        emotion_key: str = "emotion",
    ) -> Tuple[Subset, FilterResult]:
        """
        Filter a PyTorch Dataset.

        Args:
            dataset: PyTorch Dataset to filter
            audio_key: Key for audio path in dataset items
            emotion_key: Key for emotion label in dataset items

        Returns:
            Tuple of (filtered Subset, FilterResult)
        """
        # Extract audio paths and emotions from dataset
        audio_files = []
        emotions = []

        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, dict):
                audio_files.append(item.get(audio_key, ""))
                emotions.append(item.get(emotion_key, ""))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                audio_files.append(item[0])
                emotions.append(item[1])

        # Filter
        result = self.filter_dataset(audio_files, emotions)

        # Create subset
        filtered_dataset = Subset(dataset, result.sample_indices)

        return filtered_dataset, result


def create_filtered_dataloader(
    dataset: Dataset,
    filter_result: Optional[FilterResult] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader from filtered dataset.

    Args:
        dataset: Original PyTorch Dataset
        filter_result: FilterResult with sample indices (if None, use all)
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader for filtered samples
    """
    if filter_result is not None:
        # Create subset with filtered indices
        filtered_dataset = Subset(dataset, filter_result.sample_indices)
    else:
        filtered_dataset = dataset

    return DataLoader(
        filtered_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )


# =============================================================================
# Standalone filtering function
# =============================================================================

def filter_audio_files(
    audio_dir: str,
    emotion_labels: Dict[str, str],
    output_path: Optional[str] = None,
    min_agreement: int = 1,
) -> FilterResult:
    """
    Filter audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        emotion_labels: Dictionary mapping filename -> emotion
        output_path: Optional path to save filter results
        min_agreement: Minimum SER model agreement

    Returns:
        FilterResult with filtering statistics
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav"))

    files = []
    emotions = []

    for audio_file in audio_files:
        if audio_file.name in emotion_labels:
            files.append(str(audio_file))
            emotions.append(emotion_labels[audio_file.name])

    filter = SERDataFilter(min_agreement=min_agreement)
    result = filter.filter_dataset(files, emotions)

    if output_path:
        result.save(output_path)
        print(f"Filter results saved to: {output_path}")

    return result


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for SER data filtering."""
    import argparse

    parser = argparse.ArgumentParser(description="Filter dataset using SER models")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="JSON file with filename -> emotion mapping",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="filter_result.json",
        help="Output path for filter results",
    )
    parser.add_argument(
        "--min_agreement",
        type=int,
        default=1,
        help="Minimum number of SER models that must agree",
    )

    args = parser.parse_args()

    # Load labels
    with open(args.labels, "r") as f:
        emotion_labels = json.load(f)

    # Filter
    result = filter_audio_files(
        audio_dir=args.audio_dir,
        emotion_labels=emotion_labels,
        output_path=args.output,
        min_agreement=args.min_agreement,
    )

    print("\n" + result.summary())


if __name__ == "__main__":
    main()
