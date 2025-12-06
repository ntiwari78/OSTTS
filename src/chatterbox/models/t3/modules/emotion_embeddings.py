"""
Emotion type embeddings for T3 model.

This module provides learnable 64-dimensional embeddings for different emotion types,
allowing the model to control emotional characteristics of generated speech.

Features:
- 64D embeddings with structured initialization (VAD + prosodic + fine-grained)
- Emotion intensity control (0.0 = neutral, 1.0 = full, >1.0 = exaggerated)
- Emotion interpolation/blending for complex emotional expressions
"""

from typing import Dict, List, Optional

import torch
from torch import nn


# 64-dimensional emotion embeddings with structured initialization:
# - Dims 0-2: VAD (Valence, Arousal, Dominance) - psychological model
# - Dims 3-15: Prosodic features (pitch, energy, rate patterns)
# - Dims 16-63: Fine-grained learned features (initialized near zero)
#
# VAD Reference:
#   Valence: positive (happy) to negative (sad)
#   Arousal: high energy (angry/excited) to low energy (calm/sad)
#   Dominance: strong/confident to weak/submissive

def _create_64d_embedding(vad: List[float], prosodic: List[float]) -> List[float]:
    """Helper to create 64D embedding from VAD and prosodic features."""
    # VAD: 3 dimensions
    # Prosodic: 13 dimensions (pitch_mean, pitch_range, pitch_contour, energy_mean,
    #           energy_range, speaking_rate, rhythm, voice_quality, breathiness,
    #           tension, nasality, jitter, shimmer)
    # Fine-grained: 48 dimensions (initialized to small random-like values)
    fine_grained = [0.0] * 48  # Will be learned during training
    return vad + prosodic + fine_grained


EMOTION_INIT_EMBEDDINGS_64D = {
    # Neutral: baseline, all zeros
    "neutral": _create_64d_embedding(
        vad=[0.0, 0.0, 0.0],
        prosodic=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ),

    # Happy: positive valence, medium-high arousal, medium dominance
    "happy": _create_64d_embedding(
        vad=[0.8, 0.6, 0.4],
        prosodic=[0.5, 0.4, 0.3, 0.4, 0.3, 0.2, 0.3, 0.2, -0.2, -0.1, 0.0, 0.1, 0.1]
    ),

    # Sad: negative valence, low arousal, low dominance
    "sad": _create_64d_embedding(
        vad=[-0.7, -0.5, -0.4],
        prosodic=[-0.4, -0.3, -0.2, -0.5, -0.2, -0.4, -0.3, -0.2, 0.3, -0.2, 0.0, 0.0, 0.1]
    ),

    # Angry: negative valence, high arousal, high dominance
    "angry": _create_64d_embedding(
        vad=[-0.5, 0.9, 0.7],
        prosodic=[0.3, 0.6, 0.4, 0.8, 0.5, 0.3, 0.4, -0.3, -0.3, 0.6, 0.1, 0.2, 0.2]
    ),

    # Excited: very positive valence, very high arousal, high dominance
    "excited": _create_64d_embedding(
        vad=[0.9, 0.9, 0.6],
        prosodic=[0.6, 0.7, 0.5, 0.7, 0.5, 0.5, 0.5, 0.3, -0.2, 0.2, 0.0, 0.2, 0.1]
    ),

    # Calm: slightly positive valence, very low arousal, medium dominance
    "calm": _create_64d_embedding(
        vad=[0.3, -0.7, 0.2],
        prosodic=[-0.3, -0.4, -0.3, -0.4, -0.3, -0.5, -0.4, 0.3, 0.2, -0.4, 0.0, -0.1, -0.1]
    ),

    # Surprised: positive valence, high arousal, low dominance
    "surprised": _create_64d_embedding(
        vad=[0.4, 0.8, -0.2],
        prosodic=[0.7, 0.8, 0.6, 0.5, 0.4, 0.2, 0.3, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1]
    ),

    # Fearful: negative valence, high arousal, very low dominance
    "fearful": _create_64d_embedding(
        vad=[-0.6, 0.7, -0.6],
        prosodic=[0.4, 0.5, 0.3, 0.3, 0.4, 0.3, 0.2, -0.2, 0.2, 0.4, 0.1, 0.3, 0.2]
    ),

    # Disgusted: negative valence, medium arousal, medium dominance
    "disgusted": _create_64d_embedding(
        vad=[-0.6, 0.3, 0.3],
        prosodic=[0.0, 0.2, 0.1, 0.2, 0.2, -0.1, 0.1, -0.3, 0.0, 0.3, 0.2, 0.1, 0.1]
    ),

    # Whisper: neutral valence, very low arousal, low dominance
    "whisper": _create_64d_embedding(
        vad=[0.0, -0.8, -0.5],
        prosodic=[-0.5, -0.4, -0.3, -0.8, -0.4, -0.3, -0.3, -0.4, 0.6, -0.5, 0.0, 0.0, 0.0]
    ),

    # Shout: positive valence, very high arousal, very high dominance
    "shout": _create_64d_embedding(
        vad=[0.3, 1.0, 0.9],
        prosodic=[0.4, 0.5, 0.3, 1.0, 0.6, 0.2, 0.4, -0.2, -0.4, 0.7, 0.1, 0.3, 0.3]
    ),
}


class EmotionEmbeddings(nn.Module):
    """
    Learnable 64-dimensional emotion type embeddings with intensity and interpolation support.

    Features:
    - 64D embeddings for rich emotion representation
    - Intensity control: scale emotion effect from neutral (0.0) to full (1.0) to exaggerated (>1.0)
    - Emotion interpolation: blend multiple emotions with weights

    Example:
        >>> embeddings = EmotionEmbeddings()
        >>> # Get happy embedding with full intensity
        >>> happy = embeddings.get_emotion_embedding("happy", intensity=1.0)
        >>> # Get subtle happiness
        >>> subtle_happy = embeddings.get_emotion_embedding("happy", intensity=0.3)
        >>> # Blend emotions
        >>> bittersweet = embeddings.interpolate_emotions({"happy": 0.4, "sad": 0.6})
    """

    def __init__(self, emotion_embed_dim: int = 64, emotion_types: Optional[Dict] = None):
        """
        Args:
            emotion_embed_dim: Dimension of emotion embeddings (default: 64)
            emotion_types: Dictionary mapping emotion names to initial embedding values.
                          If None, uses EMOTION_INIT_EMBEDDINGS_64D.
        """
        super().__init__()
        self.emotion_embed_dim = emotion_embed_dim

        if emotion_types is None:
            emotion_types = EMOTION_INIT_EMBEDDINGS_64D

        # Create embedding table
        num_emotions = len(emotion_types)
        self.embedding = nn.Embedding(num_emotions, emotion_embed_dim)

        # Build bidirectional mapping
        emotion_list = list(emotion_types.keys())
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_list)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

        # Store neutral index for intensity calculations
        self.neutral_idx = self.emotion_to_idx.get("neutral", 0)

        # Initialize embeddings with provided values
        init_weights = torch.zeros(num_emotions, emotion_embed_dim)
        for emotion, values in emotion_types.items():
            idx = self.emotion_to_idx[emotion]
            values_len = len(values)

            if values_len == emotion_embed_dim:
                init_weights[idx] = torch.tensor(values, dtype=torch.float32)
            elif values_len < emotion_embed_dim:
                # Pad with zeros if provided values are shorter
                values_tensor = torch.tensor(values, dtype=torch.float32)
                padding = torch.zeros(emotion_embed_dim - values_len)
                init_weights[idx] = torch.cat([values_tensor, padding])
            else:
                # Truncate if provided values are longer
                init_weights[idx] = torch.tensor(values[:emotion_embed_dim], dtype=torch.float32)

        self.embedding.weight.data = init_weights

    def forward(self, emotion_indices: torch.LongTensor) -> torch.Tensor:
        """
        Get emotion embeddings for given emotion indices.

        Args:
            emotion_indices: Tensor of emotion indices (B,) or (B, 1)

        Returns:
            Emotion embeddings (B, emotion_embed_dim)
        """
        if emotion_indices.dim() == 2:
            emotion_indices = emotion_indices.squeeze(-1)
        return self.embedding(emotion_indices)

    def get_emotion_embedding(
        self,
        emotion_name: str,
        intensity: float = 1.0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get embedding for a specific emotion with intensity control.

        Intensity interpolates between neutral and the target emotion:
        - intensity=0.0: returns neutral embedding
        - intensity=1.0: returns full emotion embedding
        - intensity>1.0: extrapolates beyond the emotion (exaggerated)

        Args:
            emotion_name: Name of the emotion (e.g., "happy", "sad")
            intensity: Emotion intensity (default: 1.0). Range typically [0.0, 1.5]
            device: Target device for the embedding tensor

        Returns:
            Emotion embedding tensor (1, emotion_embed_dim)
        """
        if emotion_name not in self.emotion_to_idx:
            raise ValueError(
                f"Unknown emotion '{emotion_name}'. "
                f"Available emotions: {list(self.emotion_to_idx.keys())}"
            )

        idx = self.emotion_to_idx[emotion_name]
        target_device = device if device is not None else self.embedding.weight.device

        # Get target emotion embedding
        idx_tensor = torch.tensor([idx], dtype=torch.long, device=target_device)
        target_embed = self.embedding(idx_tensor)  # (1, embed_dim)

        # If intensity is 1.0 or emotion is neutral, return directly
        if intensity == 1.0 or emotion_name == "neutral":
            return target_embed

        # Get neutral embedding for interpolation
        neutral_tensor = torch.tensor([self.neutral_idx], dtype=torch.long, device=target_device)
        neutral_embed = self.embedding(neutral_tensor)  # (1, embed_dim)

        # Interpolate: neutral + intensity * (target - neutral)
        result = neutral_embed + intensity * (target_embed - neutral_embed)
        return result

    def interpolate_emotions(
        self,
        emotions: Dict[str, float],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Blend multiple emotions with given weights.

        The weights are normalized to sum to 1.0.

        Args:
            emotions: Dictionary mapping emotion names to weights.
                     Example: {"happy": 0.7, "excited": 0.3}
            device: Target device for the embedding tensor

        Returns:
            Blended emotion embedding tensor (1, emotion_embed_dim)
        """
        if not emotions:
            raise ValueError("emotions dict cannot be empty")

        # Validate all emotion names
        for emotion_name in emotions.keys():
            if emotion_name not in self.emotion_to_idx:
                raise ValueError(
                    f"Unknown emotion '{emotion_name}'. "
                    f"Available emotions: {list(self.emotion_to_idx.keys())}"
                )

        target_device = device if device is not None else self.embedding.weight.device

        # Compute weighted sum
        total_weight = sum(emotions.values())
        if total_weight <= 0:
            raise ValueError("Sum of emotion weights must be positive")

        result = torch.zeros(1, self.emotion_embed_dim, device=target_device)

        for emotion_name, weight in emotions.items():
            idx = self.emotion_to_idx[emotion_name]
            idx_tensor = torch.tensor([idx], dtype=torch.long, device=target_device)
            embed = self.embedding(idx_tensor)  # (1, embed_dim)
            result += (weight / total_weight) * embed

        return result

    def get_supported_emotions(self) -> List[str]:
        """Return list of supported emotion names."""
        return list(self.emotion_to_idx.keys())

    def get_emotion_index(self, emotion_name: str) -> int:
        """Get the index for an emotion name."""
        if emotion_name not in self.emotion_to_idx:
            raise ValueError(
                f"Unknown emotion '{emotion_name}'. "
                f"Available emotions: {list(self.emotion_to_idx.keys())}"
            )
        return self.emotion_to_idx[emotion_name]


def create_emotion_embeddings(emotion_embed_dim: int = 64) -> EmotionEmbeddings:
    """Factory function to create emotion embeddings with default 64D dimension."""
    return EmotionEmbeddings(emotion_embed_dim=emotion_embed_dim)
