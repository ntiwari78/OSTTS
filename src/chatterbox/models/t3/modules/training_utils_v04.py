"""
V0.4 Training Utilities for Emotion TTS

This module provides advanced training utilities for improving emotion recognition:

1. DynamicEmotionLossWeight: Adaptively weight losses based on per-emotion performance
2. HardNegativeSampler: Mine hard negatives for contrastive learning
3. EmotionCurriculum: Curriculum learning from easy to hard emotions

These utilities work together to improve training efficiency and emotion accuracy.
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
import numpy as np


# =============================================================================
# Emotion Confusion Matrix for Hard Negative Mining
# =============================================================================

# Pre-defined confusion pairs based on SER benchmark analysis
# These are emotions that are commonly confused with each other
EMOTION_CONFUSION_MATRIX = {
    "angry": ["fearful", "disgusted", "excited", "shout"],
    "fearful": ["angry", "surprised", "sad"],
    "disgusted": ["angry", "contemptuous", "sad"],
    "happy": ["excited", "surprised", "affectionate"],
    "sad": ["calm", "bored", "fearful", "neutral"],
    "neutral": ["calm", "bored", "sad"],
    "calm": ["neutral", "sad", "bored", "affectionate"],
    "surprised": ["fearful", "excited", "happy", "awed"],
    "excited": ["happy", "angry", "surprised"],
    "whisper": ["calm", "sad", "neutral"],
    "shout": ["angry", "excited", "fearful"],
    "sarcastic": ["contemptuous", "bored", "disgusted"],
    "bored": ["neutral", "sad", "calm"],
    "affectionate": ["happy", "calm", "awed"],
    "contemptuous": ["disgusted", "sarcastic", "angry"],
    "awed": ["surprised", "fearful", "affectionate"],
}

# Emotion difficulty levels for curriculum learning
# Level 1: Easily distinguishable emotions
# Level 2: Moderate difficulty
# Level 3: Hard to distinguish
EMOTION_DIFFICULTY = {
    # Level 1: High arousal contrast or unique characteristics
    "happy": 1,
    "angry": 1,
    "sad": 1,
    "surprised": 1,
    "whisper": 1,
    "shout": 1,

    # Level 2: Moderate distinguishability
    "fearful": 2,
    "disgusted": 2,
    "excited": 2,
    "neutral": 2,

    # Level 3: Often confused with similar emotions
    "calm": 3,
    "bored": 3,
    "sarcastic": 3,
    "affectionate": 3,
    "contemptuous": 3,
    "awed": 3,
}


# =============================================================================
# Dynamic Emotion Loss Weight
# =============================================================================

class DynamicEmotionLossWeight(nn.Module):
    """
    Dynamically adjust loss weights based on per-emotion performance.

    Emotions that the model struggles with receive higher loss weights,
    while well-performing emotions receive lower weights. This helps
    balance learning across all emotions.

    The weights are updated based on:
    1. Running accuracy per emotion
    2. Confusion rate with similar emotions
    3. Loss magnitude per emotion

    Example:
        >>> weight_module = DynamicEmotionLossWeight(num_emotions=16)
        >>> # During training
        >>> loss = compute_loss(...)
        >>> weighted_loss = weight_module(loss, emotion_labels, predictions)
        >>> # Periodically update weights
        >>> weight_module.update_weights(accuracy_per_emotion)
    """

    def __init__(
        self,
        emotion_names: Optional[List[str]] = None,
        initial_weight: float = 1.0,
        min_weight: float = 0.5,
        max_weight: float = 3.0,
        momentum: float = 0.9,
        update_frequency: int = 100,
    ):
        """
        Args:
            emotion_names: List of emotion names (determines number of emotions)
            initial_weight: Initial weight for all emotions (default: 1.0)
            min_weight: Minimum weight allowed (default: 0.5)
            max_weight: Maximum weight allowed (default: 3.0)
            momentum: Momentum for weight updates (default: 0.9)
            update_frequency: How often to update weights in batches (default: 100)
        """
        super().__init__()

        if emotion_names is None:
            emotion_names = list(EMOTION_DIFFICULTY.keys())

        self.emotion_names = emotion_names
        self.num_emotions = len(emotion_names)
        self.emotion_to_idx = {e: i for i, e in enumerate(emotion_names)}

        self.min_weight = min_weight
        self.max_weight = max_weight
        self.momentum = momentum
        self.update_frequency = update_frequency

        # Learnable weights per emotion
        self.register_buffer(
            "weights",
            torch.ones(self.num_emotions) * initial_weight
        )

        # Running statistics
        self.register_buffer(
            "running_accuracy",
            torch.ones(self.num_emotions) * 0.5  # Start at 50%
        )
        self.register_buffer(
            "running_loss",
            torch.ones(self.num_emotions)
        )
        self.register_buffer("batch_count", torch.tensor(0))

        # Accumulate stats between updates
        self.correct_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        self.loss_sums = defaultdict(float)

    def forward(
        self,
        loss: torch.Tensor,
        emotion_labels: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply dynamic weights to loss.

        Args:
            loss: Per-sample loss tensor (B,) or scalar
            emotion_labels: Emotion indices (B,) or emotion names
            predictions: Optional predictions for accuracy tracking (B,)

        Returns:
            Weighted loss (scalar)
        """
        # Convert emotion names to indices if needed
        # Note: indices must be on CPU to index into self.weights buffer
        if isinstance(emotion_labels, (list, tuple)):
            indices = torch.tensor(
                [self.emotion_to_idx.get(e, 0) for e in emotion_labels],
                device="cpu"  # Keep on CPU to match weights buffer
            )
        else:
            indices = emotion_labels.cpu()  # Move to CPU if on GPU

        # Get weights for each sample (weights are on CPU)
        sample_weights = self.weights[indices]
        # Move weights to same device as loss for computation
        sample_weights = sample_weights.to(loss.device)

        # Apply weights
        if loss.dim() == 0:
            # Scalar loss - can't apply per-sample weights
            weighted_loss = loss * sample_weights.mean()
        else:
            # Per-sample loss
            weighted_loss = (loss * sample_weights).mean()

        # Track statistics for weight updates
        if predictions is not None:
            self._accumulate_stats(indices, predictions, loss)

        # Periodically update weights
        self.batch_count += 1
        if self.batch_count % self.update_frequency == 0:
            self._update_weights()

        return weighted_loss

    def _accumulate_stats(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        loss: torch.Tensor,
    ):
        """Accumulate statistics for weight updates."""
        labels_np = labels.cpu().numpy()
        preds_np = predictions.cpu().numpy()

        if loss.dim() == 0:
            loss_np = np.array([loss.item()])
        else:
            loss_np = loss.detach().cpu().numpy()

        for i, (label, pred) in enumerate(zip(labels_np, preds_np)):
            self.total_counts[label] += 1
            if label == pred:
                self.correct_counts[label] += 1
            if i < len(loss_np):
                self.loss_sums[label] += loss_np[i]

    def _update_weights(self):
        """Update weights based on accumulated statistics."""
        new_weights = self.weights.clone()

        for emotion_idx in range(self.num_emotions):
            total = self.total_counts.get(emotion_idx, 0)
            if total == 0:
                continue

            # Calculate accuracy
            correct = self.correct_counts.get(emotion_idx, 0)
            accuracy = correct / total

            # Calculate average loss
            avg_loss = self.loss_sums.get(emotion_idx, 1.0) / total

            # Update running stats with momentum
            self.running_accuracy[emotion_idx] = (
                self.momentum * self.running_accuracy[emotion_idx]
                + (1 - self.momentum) * accuracy
            )
            self.running_loss[emotion_idx] = (
                self.momentum * self.running_loss[emotion_idx]
                + (1 - self.momentum) * avg_loss
            )

            # Calculate new weight: higher weight for lower accuracy
            # Weight = base_weight * (1 + (1 - accuracy))
            # Range: [1.0, 2.0] for accuracy [1.0, 0.0]
            new_weight = 1.0 + (1.0 - self.running_accuracy[emotion_idx])

            # Clamp to allowed range
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            new_weights[emotion_idx] = new_weight

        # Apply update
        self.weights.copy_(new_weights)

        # Reset accumulators
        self.correct_counts.clear()
        self.total_counts.clear()
        self.loss_sums.clear()

    def get_weights_dict(self) -> Dict[str, float]:
        """Get current weights as a dictionary."""
        return {
            self.emotion_names[i]: self.weights[i].item()
            for i in range(self.num_emotions)
        }

    def set_weights(self, weights_dict: Dict[str, float]):
        """Set weights from a dictionary."""
        for emotion, weight in weights_dict.items():
            if emotion in self.emotion_to_idx:
                idx = self.emotion_to_idx[emotion]
                self.weights[idx] = weight


# =============================================================================
# Hard Negative Sampler
# =============================================================================

class HardNegativeSampler(Sampler):
    """
    Sampler that mines hard negatives for contrastive learning.

    Hard negatives are samples from emotions that are commonly confused
    with the anchor emotion. This helps the model learn to discriminate
    between similar emotions.

    The sampler ensures that each batch contains:
    1. Positive pairs (same emotion)
    2. Hard negatives (confusable emotions)
    3. Easy negatives (different emotions)

    Example:
        >>> sampler = HardNegativeSampler(dataset, hard_negative_ratio=0.3)
        >>> dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """

    def __init__(
        self,
        dataset,
        hard_negative_ratio: float = 0.3,
        easy_negative_ratio: float = 0.2,
        batch_size: int = 32,
        confusion_matrix: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
            dataset: Dataset with 'emotion' field in samples
            hard_negative_ratio: Ratio of hard negatives per batch (default: 0.3)
            easy_negative_ratio: Ratio of easy negatives per batch (default: 0.2)
            batch_size: Batch size for sampling (default: 32)
            confusion_matrix: Optional custom confusion matrix
        """
        self.dataset = dataset
        self.hard_negative_ratio = hard_negative_ratio
        self.easy_negative_ratio = easy_negative_ratio
        self.batch_size = batch_size
        self.confusion_matrix = confusion_matrix or EMOTION_CONFUSION_MATRIX

        # Build emotion-to-indices mapping
        self.emotion_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.samples[idx] if hasattr(dataset, 'samples') else dataset[idx]
            emotion = sample.get('emotion', sample.get('label', 'neutral'))
            self.emotion_indices[emotion].append(idx)

        self.emotions = list(self.emotion_indices.keys())
        self.num_samples = len(dataset)

    def __iter__(self):
        """Generate batches with hard negatives."""
        indices = []
        used_indices: Set[int] = set()

        while len(indices) < self.num_samples:
            # Select anchor emotion
            anchor_emotion = random.choice(self.emotions)
            anchor_indices = self.emotion_indices[anchor_emotion]

            if not anchor_indices:
                continue

            # Select anchor sample
            anchor_idx = random.choice(anchor_indices)
            if anchor_idx in used_indices:
                continue

            batch_indices = [anchor_idx]
            used_indices.add(anchor_idx)

            # Add positive samples (same emotion)
            positive_count = int(self.batch_size * (1 - self.hard_negative_ratio - self.easy_negative_ratio))
            available_positives = [i for i in anchor_indices if i not in used_indices]
            positive_samples = random.sample(
                available_positives,
                min(positive_count, len(available_positives))
            )
            batch_indices.extend(positive_samples)
            used_indices.update(positive_samples)

            # Add hard negatives (confusable emotions)
            hard_neg_count = int(self.batch_size * self.hard_negative_ratio)
            hard_neg_emotions = self.confusion_matrix.get(anchor_emotion, [])

            for neg_emotion in hard_neg_emotions:
                if len(batch_indices) >= len(batch_indices) + hard_neg_count:
                    break
                if neg_emotion not in self.emotion_indices:
                    continue

                neg_indices = self.emotion_indices[neg_emotion]
                available_negs = [i for i in neg_indices if i not in used_indices]
                if available_negs:
                    neg_sample = random.choice(available_negs)
                    batch_indices.append(neg_sample)
                    used_indices.add(neg_sample)

            # Add easy negatives (random different emotions)
            easy_neg_count = int(self.batch_size * self.easy_negative_ratio)
            other_emotions = [e for e in self.emotions if e != anchor_emotion and e not in hard_neg_emotions]

            for _ in range(easy_neg_count):
                if not other_emotions:
                    break
                neg_emotion = random.choice(other_emotions)
                neg_indices = self.emotion_indices[neg_emotion]
                available_negs = [i for i in neg_indices if i not in used_indices]
                if available_negs:
                    neg_sample = random.choice(available_negs)
                    batch_indices.append(neg_sample)
                    used_indices.add(neg_sample)

            indices.extend(batch_indices)

        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples


# =============================================================================
# Emotion Curriculum Learning
# =============================================================================

class EmotionCurriculum:
    """
    Curriculum learning for emotions.

    Starts training with easily distinguishable emotions and gradually
    introduces harder emotions as training progresses. This helps the
    model build a strong foundation before tackling confusing cases.

    Curriculum stages:
    1. Easy emotions (high arousal contrast): happy, angry, sad, surprised, whisper, shout
    2. Medium emotions: fearful, disgusted, excited, neutral
    3. Hard emotions: calm, bored, sarcastic, affectionate, contemptuous, awed

    Example:
        >>> curriculum = EmotionCurriculum(total_epochs=30)
        >>> for epoch in range(30):
        ...     allowed_emotions = curriculum.get_emotions_for_epoch(epoch)
        ...     # Filter dataset to only include allowed emotions
    """

    def __init__(
        self,
        total_epochs: int = 30,
        difficulty_levels: Optional[Dict[str, int]] = None,
        warmup_epochs: int = 3,
        stage_transition: str = "linear",  # "linear" or "step"
    ):
        """
        Args:
            total_epochs: Total number of training epochs
            difficulty_levels: Optional custom difficulty levels per emotion
            warmup_epochs: Epochs to stay at each difficulty level (for step)
            stage_transition: How to transition between stages
        """
        self.total_epochs = total_epochs
        self.difficulty_levels = difficulty_levels or EMOTION_DIFFICULTY
        self.warmup_epochs = warmup_epochs
        self.stage_transition = stage_transition

        # Group emotions by difficulty
        self.emotions_by_level = defaultdict(list)
        for emotion, level in self.difficulty_levels.items():
            self.emotions_by_level[level].append(emotion)

        self.max_level = max(self.difficulty_levels.values())
        self.min_level = min(self.difficulty_levels.values())

        # Calculate stage boundaries
        self.stage_epochs = total_epochs // self.max_level

    def get_emotions_for_epoch(self, epoch: int) -> List[str]:
        """
        Get allowed emotions for a given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            List of allowed emotion names
        """
        if self.stage_transition == "linear":
            # Linearly increase difficulty
            progress = min(1.0, epoch / max(1, self.total_epochs - 1))
            current_level = self.min_level + progress * (self.max_level - self.min_level)
        else:  # step
            # Step-wise increase
            stage = epoch // self.warmup_epochs
            current_level = min(self.min_level + stage, self.max_level)

        # Include all emotions up to current level
        allowed_emotions = []
        for level in range(self.min_level, int(current_level) + 1):
            allowed_emotions.extend(self.emotions_by_level[level])

        return allowed_emotions

    def get_current_difficulty(self, epoch: int) -> float:
        """Get the current difficulty level (for logging)."""
        if self.stage_transition == "linear":
            progress = min(1.0, epoch / max(1, self.total_epochs - 1))
            return self.min_level + progress * (self.max_level - self.min_level)
        else:
            stage = epoch // self.warmup_epochs
            return min(self.min_level + stage, self.max_level)

    def should_add_new_emotions(self, epoch: int) -> bool:
        """Check if new emotions should be added this epoch."""
        if epoch == 0:
            return True
        prev_level = int(self.get_current_difficulty(epoch - 1))
        curr_level = int(self.get_current_difficulty(epoch))
        return curr_level > prev_level

    def get_stage_name(self, epoch: int) -> str:
        """Get human-readable stage name."""
        level = int(self.get_current_difficulty(epoch))
        if level == 1:
            return "Easy (basic emotions)"
        elif level == 2:
            return "Medium (add complex emotions)"
        else:
            return "Hard (all emotions)"


# =============================================================================
# Combined Training Manager
# =============================================================================

class V04TrainingManager:
    """
    Combined manager for V0.4 training improvements.

    Integrates:
    - Dynamic loss weighting
    - Hard negative sampling
    - Curriculum learning

    Example:
        >>> manager = V04TrainingManager(
        ...     emotion_names=["happy", "sad", "angry", ...],
        ...     total_epochs=30,
        ... )
        >>> for epoch in range(30):
        ...     manager.on_epoch_start(epoch)
        ...     for batch in dataloader:
        ...         loss = manager.apply_loss_weights(loss, emotions, preds)
        ...     manager.on_epoch_end(epoch, metrics)
    """

    def __init__(
        self,
        emotion_names: List[str],
        total_epochs: int = 30,
        use_dynamic_weights: bool = True,
        use_curriculum: bool = True,
        hard_negative_ratio: float = 0.3,
    ):
        """
        Args:
            emotion_names: List of all emotion names
            total_epochs: Total training epochs
            use_dynamic_weights: Enable dynamic loss weighting
            use_curriculum: Enable curriculum learning
            hard_negative_ratio: Ratio for hard negative sampling
        """
        self.emotion_names = emotion_names
        self.total_epochs = total_epochs
        self.use_dynamic_weights = use_dynamic_weights
        self.use_curriculum = use_curriculum
        self.hard_negative_ratio = hard_negative_ratio

        # Initialize components
        if use_dynamic_weights:
            self.loss_weight = DynamicEmotionLossWeight(
                emotion_names=emotion_names,
            )

        if use_curriculum:
            self.curriculum = EmotionCurriculum(
                total_epochs=total_epochs,
            )

        self.current_epoch = 0
        self.allowed_emotions: Set[str] = set(emotion_names)

    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        self.current_epoch = epoch

        if self.use_curriculum:
            self.allowed_emotions = set(
                self.curriculum.get_emotions_for_epoch(epoch)
            )

            if self.curriculum.should_add_new_emotions(epoch):
                print(f"  Curriculum: {self.curriculum.get_stage_name(epoch)}")
                print(f"  Allowed emotions ({len(self.allowed_emotions)}): {sorted(self.allowed_emotions)}")

    def on_epoch_end(self, epoch: int, metrics: Optional[Dict] = None):
        """Called at the end of each epoch."""
        if self.use_dynamic_weights and metrics:
            # Log current weights
            weights = self.loss_weight.get_weights_dict()
            print(f"  Dynamic weights: {weights}")

    def apply_loss_weights(
        self,
        loss: torch.Tensor,
        emotion_labels,
        predictions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply dynamic loss weights."""
        if self.use_dynamic_weights:
            return self.loss_weight(loss, emotion_labels, predictions)
        return loss

    def filter_batch_by_curriculum(
        self,
        batch: Dict,
        emotion_key: str = "emotion",
    ) -> Optional[Dict]:
        """
        Filter batch to only include allowed emotions.

        Returns None if batch is entirely filtered out.
        """
        if not self.use_curriculum:
            return batch

        emotions = batch.get(emotion_key, [])
        if isinstance(emotions, torch.Tensor):
            # Can't filter tensor batch easily
            return batch

        # Filter samples
        mask = [e in self.allowed_emotions for e in emotions]

        if not any(mask):
            return None

        # Filter each field in batch
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, (list, tuple)):
                filtered_batch[key] = [v for v, m in zip(value, mask) if m]
            elif isinstance(value, torch.Tensor) and value.size(0) == len(mask):
                filtered_batch[key] = value[torch.tensor(mask)]
            else:
                filtered_batch[key] = value

        return filtered_batch

    def get_sampler(self, dataset) -> Optional[Sampler]:
        """Get hard negative sampler if configured."""
        if self.hard_negative_ratio > 0:
            return HardNegativeSampler(
                dataset=dataset,
                hard_negative_ratio=self.hard_negative_ratio,
            )
        return None


# =============================================================================
# Utility Functions
# =============================================================================

def compute_emotion_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    emotion_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-emotion accuracy.

    Args:
        predictions: Predicted emotion indices (B,)
        labels: Ground truth emotion indices (B,)
        emotion_names: Optional list of emotion names for dict keys

    Returns:
        Dictionary mapping emotion to accuracy
    """
    accuracies = {}

    unique_labels = labels.unique()
    for label in unique_labels:
        mask = labels == label
        correct = (predictions[mask] == label).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0

        if emotion_names and label.item() < len(emotion_names):
            key = emotion_names[label.item()]
        else:
            key = f"emotion_{label.item()}"

        accuracies[key] = accuracy

    return accuracies


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted emotion indices (B,)
        labels: Ground truth emotion indices (B,)
        num_classes: Number of emotion classes

    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for pred, label in zip(predictions, labels):
        confusion[label.item(), pred.item()] += 1

    return confusion


def get_hard_negatives_from_confusion(
    confusion_matrix: torch.Tensor,
    threshold: float = 0.1,
) -> Dict[int, List[int]]:
    """
    Extract hard negatives from confusion matrix.

    Args:
        confusion_matrix: Confusion matrix (num_classes, num_classes)
        threshold: Minimum confusion rate to consider as hard negative

    Returns:
        Dictionary mapping class to list of hard negative classes
    """
    num_classes = confusion_matrix.size(0)
    hard_negatives = {}

    # Normalize confusion matrix by row
    row_sums = confusion_matrix.sum(dim=1, keepdim=True).float()
    row_sums = row_sums.clamp(min=1)
    normalized = confusion_matrix.float() / row_sums

    for i in range(num_classes):
        # Find classes that i is often confused with
        negatives = []
        for j in range(num_classes):
            if i != j and normalized[i, j] > threshold:
                negatives.append(j)
        hard_negatives[i] = negatives

    return hard_negatives
