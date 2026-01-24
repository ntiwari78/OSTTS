"""
Emotion type embeddings for T3 model.

This module provides learnable 64-dimensional embeddings for different emotion types,
allowing the model to control emotional characteristics of generated speech.

Features:
- 64D embeddings with structured initialization (VAD + prosodic + fine-grained)
- Emotion intensity control (0.0 = neutral, 1.0 = full, >1.0 = exaggerated)
- Nonlinear intensity transformation via learned MLP
- Emotion interpolation/blending for complex emotional expressions
"""

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# Intensity Transform Module
# =============================================================================

class IntensityTransform(nn.Module):
    """
    Nonlinear intensity mapping for emotion embeddings.

    Instead of linear interpolation in Euclidean space, this learns
    a curved manifold that better captures perceptual intensity scaling.

    The transform blends between linear interpolation and a learned nonlinear
    mapping, allowing the model to capture nuanced intensity relationships.

    Architecture:
    - Input: [emotion_embed (64D), intensity (1D)] -> 65D
    - Hidden: 128D with GELU activation
    - Output: 64D transformed embedding

    Example:
        >>> transform = IntensityTransform(emotion_dim=64)
        >>> neutral = torch.randn(1, 64)
        >>> target = torch.randn(1, 64)
        >>> intensity = torch.tensor([[0.5]])
        >>> result = transform(neutral, target, intensity)
    """

    def __init__(self, emotion_dim: int = 64, hidden_dim: int = 128):
        """
        Args:
            emotion_dim: Dimension of emotion embeddings (default: 64)
            hidden_dim: Hidden dimension for MLP (default: 128)
        """
        super().__init__()
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim

        # MLP for nonlinear intensity mapping
        # Input: target emotion + intensity scalar
        self.transform = nn.Sequential(
            nn.Linear(emotion_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # Learnable blend weight between linear and nonlinear (starts at 0.5)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        # Initialize to approximate identity + small nonlinearity
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to start close to linear interpolation behavior."""
        for layer in self.transform:
            if isinstance(layer, nn.Linear):
                # Small initialization for gentle deviation from linear
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        neutral_embed: torch.Tensor,  # (B, emotion_dim)
        target_embed: torch.Tensor,   # (B, emotion_dim)
        intensity: torch.Tensor,      # (B, 1) or scalar
    ) -> torch.Tensor:
        """
        Apply nonlinear intensity transformation.

        Args:
            neutral_embed: Neutral emotion embedding (B, emotion_dim)
            target_embed: Target emotion embedding (B, emotion_dim)
            intensity: Intensity value (B, 1) or scalar

        Returns:
            Transformed emotion embedding (B, emotion_dim)
        """
        # Handle scalar and 1D intensity
        if intensity.dim() == 0:
            intensity = intensity.unsqueeze(0).unsqueeze(0)
        elif intensity.dim() == 1:
            intensity = intensity.unsqueeze(-1)

        # Ensure batch dimension matches
        if neutral_embed.dim() == 1:
            neutral_embed = neutral_embed.unsqueeze(0)
        if target_embed.dim() == 1:
            target_embed = target_embed.unsqueeze(0)

        # Expand intensity to match batch size
        if intensity.size(0) != target_embed.size(0):
            intensity = intensity.expand(target_embed.size(0), -1)

        # Compute linear baseline for residual connection
        linear_result = neutral_embed + intensity * (target_embed - neutral_embed)

        # Concatenate target embedding with intensity for MLP input
        mlp_input = torch.cat([target_embed, intensity], dim=-1)

        # Nonlinear transform gives delta from neutral
        nonlinear_delta = self.transform(mlp_input)

        # Blend linear and nonlinear with learnable sigmoid weight
        alpha = torch.sigmoid(self.residual_weight)
        result = alpha * linear_result + (1 - alpha) * (neutral_embed + nonlinear_delta * intensity)

        return result


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


# =============================================================================
# Phase 2: Enhanced Prosodic Parameters for Better SER Recognition
# =============================================================================
# Prosodic dimensions (indices 3-15):
#   0: pitch_mean      - Overall pitch level
#   1: pitch_range     - Pitch variation range
#   2: pitch_contour   - Intonation pattern
#   3: energy_mean     - Overall loudness/energy
#   4: energy_range    - Energy variation
#   5: speaking_rate   - Speech tempo
#   6: rhythm          - Rhythmic patterns
#   7: voice_quality   - Voice timbre
#   8: breathiness     - Breathy quality
#   9: tension         - Vocal tension
#   10: nasality       - Nasal quality
#   11: jitter         - Pitch perturbation
#   12: shimmer        - Amplitude perturbation
# =============================================================================

EMOTION_INIT_EMBEDDINGS_64D = {
    # Neutral: baseline, all zeros
    "neutral": _create_64d_embedding(
        vad=[0.0, 0.0, 0.0],
        prosodic=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ),

    # Happy: positive valence, high arousal, medium dominance
    # ENHANCED: Increased pitch, energy, and speaking rate for better recognition
    "happy": _create_64d_embedding(
        vad=[0.9, 0.8, 0.5],  # Increased from [0.8, 0.6, 0.4]
        prosodic=[0.7, 0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.3, -0.3, -0.2, 0.0, 0.15, 0.15]
        # Was:   [0.5, 0.4, 0.3, 0.4, 0.3, 0.2, 0.3, 0.2, -0.2, -0.1, 0.0, 0.1, 0.1]
    ),

    # Sad: negative valence, low arousal, low dominance
    # ENHANCED: Lower pitch, less energy, slower rate for clearer sadness
    "sad": _create_64d_embedding(
        vad=[-0.8, -0.7, -0.5],  # Increased from [-0.7, -0.5, -0.4]
        prosodic=[-0.6, -0.5, -0.4, -0.7, -0.4, -0.6, -0.5, -0.3, 0.4, -0.3, 0.0, 0.05, 0.15]
        # Was:   [-0.4, -0.3, -0.2, -0.5, -0.2, -0.4, -0.3, -0.2, 0.3, -0.2, 0.0, 0.0, 0.1]
    ),

    # Angry: negative valence, very high arousal, high dominance
    # ENHANCED: Maximum energy, tension, faster rate for unmistakable anger
    "angry": _create_64d_embedding(
        vad=[-0.6, 1.0, 0.9],  # Increased from [-0.5, 0.9, 0.7]
        prosodic=[0.5, 0.8, 0.6, 1.0, 0.7, 0.5, 0.6, -0.4, -0.4, 0.9, 0.15, 0.3, 0.3]
        # Was:   [0.3, 0.6, 0.4, 0.8, 0.5, 0.3, 0.4, -0.3, -0.3, 0.6, 0.1, 0.2, 0.2]
    ),

    # Excited: very positive valence, very high arousal, high dominance
    # ENHANCED: High energy and fast rate
    "excited": _create_64d_embedding(
        vad=[1.0, 1.0, 0.7],  # Increased from [0.9, 0.9, 0.6]
        prosodic=[0.8, 0.9, 0.7, 0.9, 0.7, 0.7, 0.7, 0.4, -0.3, 0.3, 0.0, 0.25, 0.15]
        # Was:   [0.6, 0.7, 0.5, 0.7, 0.5, 0.5, 0.5, 0.3, -0.2, 0.2, 0.0, 0.2, 0.1]
    ),

    # Calm: slightly positive valence, very low arousal, medium dominance
    # ENHANCED: Lower energy and slower rate
    "calm": _create_64d_embedding(
        vad=[0.4, -0.9, 0.3],  # Adjusted from [0.3, -0.7, 0.2]
        prosodic=[-0.5, -0.6, -0.5, -0.6, -0.5, -0.7, -0.6, 0.4, 0.3, -0.6, 0.0, -0.15, -0.15]
        # Was:   [-0.3, -0.4, -0.3, -0.4, -0.3, -0.5, -0.4, 0.3, 0.2, -0.4, 0.0, -0.1, -0.1]
    ),

    # Surprised: positive valence, high arousal, low dominance
    # ENHANCED: Very high pitch range for unmistakable surprise
    "surprised": _create_64d_embedding(
        vad=[0.5, 1.0, -0.3],  # Increased from [0.4, 0.8, -0.2]
        prosodic=[0.9, 1.0, 0.8, 0.7, 0.6, 0.3, 0.4, 0.2, 0.15, 0.2, 0.0, 0.15, 0.15]
        # Was:   [0.7, 0.8, 0.6, 0.5, 0.4, 0.2, 0.3, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1]
    ),

    # Fearful: negative valence, high arousal, very low dominance
    # ENHANCED: Higher tension and tremor for clearer fear
    "fearful": _create_64d_embedding(
        vad=[-0.7, 0.9, -0.8],  # Adjusted from [-0.6, 0.7, -0.6]
        prosodic=[0.6, 0.7, 0.5, 0.4, 0.6, 0.5, 0.3, -0.3, 0.3, 0.6, 0.15, 0.4, 0.3]
        # Was:   [0.4, 0.5, 0.3, 0.3, 0.4, 0.3, 0.2, -0.2, 0.2, 0.4, 0.1, 0.3, 0.2]
    ),

    # Disgusted: negative valence, medium arousal, medium dominance
    # ENHANCED: More nasal quality and tension
    "disgusted": _create_64d_embedding(
        vad=[-0.7, 0.4, 0.4],  # Adjusted from [-0.6, 0.3, 0.3]
        prosodic=[0.1, 0.3, 0.2, 0.3, 0.3, -0.2, 0.15, -0.4, 0.0, 0.5, 0.3, 0.15, 0.15]
        # Was:   [0.0, 0.2, 0.1, 0.2, 0.2, -0.1, 0.1, -0.3, 0.0, 0.3, 0.2, 0.1, 0.1]
    ),

    # Whisper: neutral valence, very low arousal, low dominance
    # ENHANCED: Maximum breathiness and minimum energy
    "whisper": _create_64d_embedding(
        vad=[0.0, -1.0, -0.7],  # Adjusted from [0.0, -0.8, -0.5]
        prosodic=[-0.7, -0.6, -0.5, -1.0, -0.6, -0.4, -0.4, -0.5, 0.8, -0.7, 0.0, 0.0, 0.0]
        # Was:   [-0.5, -0.4, -0.3, -0.8, -0.4, -0.3, -0.3, -0.4, 0.6, -0.5, 0.0, 0.0, 0.0]
    ),

    # Shout: positive valence, maximum arousal, maximum dominance
    # ENHANCED: Maximum energy and tension
    "shout": _create_64d_embedding(
        vad=[0.4, 1.0, 1.0],  # Adjusted from [0.3, 1.0, 0.9]
        prosodic=[0.6, 0.7, 0.5, 1.0, 0.8, 0.3, 0.5, -0.3, -0.5, 0.9, 0.15, 0.4, 0.4]
        # Was:   [0.4, 0.5, 0.3, 1.0, 0.6, 0.2, 0.4, -0.2, -0.4, 0.7, 0.1, 0.3, 0.3]
    ),

    # =========================================================================
    # New emotions (v0.2.1) - 5 additional emotions (ENHANCED)
    # =========================================================================

    # Sarcastic: slightly negative valence, medium arousal, medium-high dominance
    # ENHANCED: More exaggerated intonation patterns
    "sarcastic": _create_64d_embedding(
        vad=[-0.3, 0.4, 0.5],  # Adjusted from [-0.2, 0.3, 0.4]
        prosodic=[0.3, 0.7, 0.6, 0.3, 0.4, -0.15, 0.4, -0.3, 0.0, 0.15, 0.15, 0.15, 0.0]
        # Was:   [0.2, 0.5, 0.4, 0.2, 0.3, -0.1, 0.3, -0.2, 0.0, 0.1, 0.1, 0.1, 0.0]
    ),

    # Bored: negative valence, very low arousal, low dominance
    # ENHANCED: Flatter intonation, slower rate
    "bored": _create_64d_embedding(
        vad=[-0.4, -0.8, -0.3],  # Adjusted from [-0.3, -0.6, -0.2]
        prosodic=[-0.5, -0.7, -0.6, -0.6, -0.5, -0.7, -0.6, -0.15, 0.15, -0.4, 0.0, 0.0, 0.0]
        # Was:   [-0.3, -0.5, -0.4, -0.4, -0.3, -0.5, -0.4, -0.1, 0.1, -0.3, 0.0, 0.0, 0.0]
    ),

    # Affectionate: very positive valence, medium-low arousal, medium dominance
    # ENHANCED: Warmer tone, softer voice
    "affectionate": _create_64d_embedding(
        vad=[1.0, 0.4, 0.4],  # Adjusted from [0.9, 0.3, 0.3]
        prosodic=[0.3, 0.4, 0.3, 0.15, 0.3, -0.3, 0.3, 0.5, 0.4, -0.4, 0.0, 0.0, 0.0]
        # Was:   [0.2, 0.3, 0.2, 0.1, 0.2, -0.2, 0.2, 0.4, 0.3, -0.3, 0.0, 0.0, 0.0]
    ),

    # Contemptuous: negative valence, low-medium arousal, high dominance
    # ENHANCED: More dismissive tone
    "contemptuous": _create_64d_embedding(
        vad=[-0.6, 0.2, 0.7],  # Adjusted from [-0.5, 0.1, 0.6]
        prosodic=[0.1, 0.3, 0.15, 0.15, 0.3, -0.3, 0.15, -0.4, 0.0, 0.3, 0.3, 0.15, 0.0]
        # Was:   [0.0, 0.2, 0.1, 0.1, 0.2, -0.2, 0.1, -0.3, 0.0, 0.2, 0.2, 0.1, 0.0]
    ),

    # Awed: positive valence, medium-high arousal, low dominance
    # ENHANCED: More breathy, wider pitch range
    "awed": _create_64d_embedding(
        vad=[0.7, 0.6, -0.4],  # Adjusted from [0.6, 0.5, -0.3]
        prosodic=[0.4, 0.7, 0.6, 0.4, 0.4, -0.3, 0.3, 0.3, 0.5, -0.15, 0.0, 0.15, 0.0]
        # Was:   [0.3, 0.5, 0.4, 0.3, 0.3, -0.2, 0.2, 0.2, 0.3, -0.1, 0.0, 0.1, 0.0]
    ),
}


class EmotionEmbeddings(nn.Module):
    """
    Learnable 64-dimensional emotion type embeddings with intensity and interpolation support.

    Features:
    - 64D embeddings for rich emotion representation (16 emotions)
    - Intensity control: scale emotion effect from neutral (0.0) to full (1.0) to exaggerated (>1.0)
    - Nonlinear intensity transform: learned MLP for perceptually accurate intensity scaling
    - Emotion interpolation: blend multiple emotions with weights
    - Expressiveness scale: global multiplier for prosodic features (Phase 2)

    Supported emotions (16 total):
    - Basic: neutral, happy, sad, angry, fearful, disgusted, surprised
    - Extended: excited, calm, whisper, shout
    - New (v0.2.1): sarcastic, bored, affectionate, contemptuous, awed

    Example:
        >>> embeddings = EmotionEmbeddings()
        >>> # Get happy embedding with full intensity
        >>> happy = embeddings.get_emotion_embedding("happy", intensity=1.0)
        >>> # Get subtle happiness with nonlinear mapping
        >>> subtle_happy = embeddings.get_emotion_embedding("happy", intensity=0.3)
        >>> # Blend emotions
        >>> bittersweet = embeddings.interpolate_emotions({"happy": 0.4, "sad": 0.6})
    """

    def __init__(
        self,
        emotion_embed_dim: int = 64,
        emotion_types: Optional[Dict] = None,
        use_nonlinear_intensity: bool = True,
        expressiveness_scale: float = 1.0,
    ):
        """
        Args:
            emotion_embed_dim: Dimension of emotion embeddings (default: 64)
            emotion_types: Dictionary mapping emotion names to initial embedding values.
                          If None, uses EMOTION_INIT_EMBEDDINGS_64D.
            use_nonlinear_intensity: Whether to use learned nonlinear intensity transform
                                    (default: True for v0.2+, set False for backward compatibility)
            expressiveness_scale: Global multiplier for prosodic features (default: 1.0)
                                 Values > 1.0 make emotions more pronounced for SER recognition.
                                 Recommended: 1.0-1.5 for natural speech, 1.5-2.0 for expressive.
        """
        super().__init__()
        self.emotion_embed_dim = emotion_embed_dim
        self.use_nonlinear_intensity = use_nonlinear_intensity
        self.expressiveness_scale = expressiveness_scale

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

        # Nonlinear intensity transform (v0.2+)
        if use_nonlinear_intensity:
            self.intensity_transform = IntensityTransform(
                emotion_dim=emotion_embed_dim,
                hidden_dim=128,
            )

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
        device: Optional[torch.device] = None,
        use_linear: bool = False,
    ) -> torch.Tensor:
        """
        Get embedding for a specific emotion with intensity control.

        Intensity interpolates between neutral and the target emotion:
        - intensity=0.0: returns neutral embedding
        - intensity=1.0: returns full emotion embedding
        - intensity>1.0: extrapolates beyond the emotion (exaggerated)

        By default, uses nonlinear intensity transform for perceptually
        accurate intensity scaling. Set use_linear=True for backward
        compatibility with linear interpolation.

        Args:
            emotion_name: Name of the emotion (e.g., "happy", "sad")
            intensity: Emotion intensity (default: 1.0). Range typically [0.0, 1.5]
            device: Target device for the embedding tensor
            use_linear: Force linear interpolation even if nonlinear is enabled

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

        # Apply expressiveness scale to prosodic dimensions (3-15)
        # This makes emotions more pronounced for better SER recognition
        if self.expressiveness_scale != 1.0 and emotion_name != "neutral":
            target_embed = target_embed.clone()
            # Scale prosodic features (dims 3-15) by expressiveness_scale
            target_embed[:, 3:16] = target_embed[:, 3:16] * self.expressiveness_scale

        # If intensity is 1.0 or emotion is neutral, return directly
        if intensity == 1.0 or emotion_name == "neutral":
            return target_embed

        # Get neutral embedding for interpolation
        neutral_tensor = torch.tensor([self.neutral_idx], dtype=torch.long, device=target_device)
        neutral_embed = self.embedding(neutral_tensor)  # (1, embed_dim)

        # Use nonlinear transform if enabled and available
        if self.use_nonlinear_intensity and hasattr(self, 'intensity_transform') and not use_linear:
            intensity_tensor = torch.tensor([[intensity]], device=target_device)
            result = self.intensity_transform(
                neutral_embed,  # (1, embed_dim)
                target_embed,   # (1, embed_dim)
                intensity_tensor,
            )
            return result

        # Fallback to linear interpolation
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


def create_emotion_embeddings(
    emotion_embed_dim: int = 64,
    expressiveness_scale: float = 1.0,
) -> EmotionEmbeddings:
    """
    Factory function to create emotion embeddings.

    Args:
        emotion_embed_dim: Dimension of emotion embeddings (default: 64)
        expressiveness_scale: Global multiplier for prosodic features (default: 1.0)
                             Use 1.5 for better SER recognition.

    Returns:
        EmotionEmbeddings instance
    """
    return EmotionEmbeddings(
        emotion_embed_dim=emotion_embed_dim,
        expressiveness_scale=expressiveness_scale,
    )
