"""
Emotion Prosody Prediction Module (Phase 2)

This module provides:
1. EmotionProsodyPredictor: Predicts explicit prosodic features from emotion embeddings
2. ProsodyMatchingLoss: Compares predicted prosody with actual audio prosody
3. ProsodyExtractor: Extracts prosodic features from audio

These components enable the model to learn more recognizable emotional prosody
by providing explicit supervision on prosodic features.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# =============================================================================
# Prosody Feature Definitions
# =============================================================================

# Target ranges for each prosodic feature (used for normalization)
PROSODY_FEATURE_RANGES = {
    "pitch_mean": (80.0, 400.0),     # Hz - fundamental frequency
    "pitch_std": (0.0, 100.0),       # Hz - pitch variation
    "pitch_range": (0.0, 200.0),     # Hz - pitch span
    "energy_mean": (0.01, 0.20),     # RMS energy
    "energy_std": (0.0, 0.10),       # RMS variation
    "tempo": (0.5, 2.0),             # Relative speaking rate
    "duration": (0.5, 2.0),          # Relative duration
}

# Expected prosody profiles for each emotion (normalized 0-1)
EMOTION_PROSODY_TARGETS = {
    "neutral": {
        "pitch_mean": 0.4,
        "pitch_std": 0.3,
        "pitch_range": 0.3,
        "energy_mean": 0.4,
        "energy_std": 0.3,
        "tempo": 0.5,
        "duration": 0.5,
    },
    "happy": {
        "pitch_mean": 0.7,
        "pitch_std": 0.6,
        "pitch_range": 0.6,
        "energy_mean": 0.6,
        "energy_std": 0.5,
        "tempo": 0.6,
        "duration": 0.4,
    },
    "sad": {
        "pitch_mean": 0.3,
        "pitch_std": 0.2,
        "pitch_range": 0.3,
        "energy_mean": 0.3,
        "energy_std": 0.2,
        "tempo": 0.3,
        "duration": 0.7,
    },
    "angry": {
        "pitch_mean": 0.6,
        "pitch_std": 0.8,
        "pitch_range": 0.7,
        "energy_mean": 0.9,
        "energy_std": 0.7,
        "tempo": 0.7,
        "duration": 0.4,
    },
    "fearful": {
        "pitch_mean": 0.6,
        "pitch_std": 0.7,
        "pitch_range": 0.6,
        "energy_mean": 0.4,
        "energy_std": 0.6,
        "tempo": 0.7,
        "duration": 0.4,
    },
    "surprised": {
        "pitch_mean": 0.8,
        "pitch_std": 0.9,
        "pitch_range": 0.9,
        "energy_mean": 0.7,
        "energy_std": 0.6,
        "tempo": 0.5,
        "duration": 0.5,
    },
    "disgusted": {
        "pitch_mean": 0.4,
        "pitch_std": 0.4,
        "pitch_range": 0.4,
        "energy_mean": 0.4,
        "energy_std": 0.4,
        "tempo": 0.4,
        "duration": 0.6,
    },
    "calm": {
        "pitch_mean": 0.35,
        "pitch_std": 0.2,
        "pitch_range": 0.25,
        "energy_mean": 0.3,
        "energy_std": 0.2,
        "tempo": 0.35,
        "duration": 0.65,
    },
    "excited": {
        "pitch_mean": 0.8,
        "pitch_std": 0.8,
        "pitch_range": 0.8,
        "energy_mean": 0.85,
        "energy_std": 0.7,
        "tempo": 0.8,
        "duration": 0.35,
    },
}


# =============================================================================
# Prosody Prediction Head
# =============================================================================

class EmotionProsodyPredictor(nn.Module):
    """
    Predicts explicit prosodic features from emotion embedding.

    This module takes a 64D emotion embedding and predicts 7 prosodic
    features that characterize the emotional expression in speech:
    - pitch_mean: Average fundamental frequency
    - pitch_std: Pitch variation/stability
    - pitch_range: Range of pitch values
    - energy_mean: Average loudness
    - energy_std: Energy variation
    - tempo: Speaking rate (relative)
    - duration: Utterance duration (relative)

    These predictions guide the acoustic model and provide explicit
    supervision during training.

    Example:
        >>> predictor = EmotionProsodyPredictor(emotion_dim=64)
        >>> emotion_embed = torch.randn(1, 64)
        >>> prosody = predictor(emotion_embed)
        >>> print(prosody["pitch_mean"])  # Predicted pitch in Hz
    """

    def __init__(
        self,
        emotion_dim: int = 64,
        hidden_dim: int = 128,
        num_prosody_features: int = 7,
    ):
        """
        Args:
            emotion_dim: Dimension of input emotion embedding (default: 64)
            hidden_dim: Hidden dimension for MLP (default: 128)
            num_prosody_features: Number of prosodic features to predict (default: 7)
        """
        super().__init__()
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_prosody_features

        # MLP for prosody prediction
        self.prosody_head = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_prosody_features),
        )

        # Feature names in output order
        self.feature_names = [
            "pitch_mean",
            "pitch_std",
            "pitch_range",
            "energy_mean",
            "energy_std",
            "tempo",
            "duration",
        ]

        # Register ranges as buffers
        ranges = torch.tensor([
            PROSODY_FEATURE_RANGES[name] for name in self.feature_names
        ])
        self.register_buffer("feature_min", ranges[:, 0])
        self.register_buffer("feature_max", ranges[:, 1])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for layer in self.prosody_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, emotion_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict prosodic features from emotion embedding.

        Args:
            emotion_embed: Emotion embedding (B, emotion_dim)

        Returns:
            Dictionary of predicted prosodic features:
            - Each key is a feature name
            - Each value is a tensor of shape (B,) with values in physical units
        """
        # Handle sequence dimension if present
        if emotion_embed.dim() == 3:
            emotion_embed = emotion_embed.mean(dim=1)  # Average over sequence

        # Predict raw values (pre-sigmoid)
        raw = self.prosody_head(emotion_embed)  # (B, num_features)

        # Apply sigmoid and scale to target ranges
        normalized = torch.sigmoid(raw)  # (B, num_features) in [0, 1]

        # Scale to physical ranges
        scaled = normalized * (self.feature_max - self.feature_min) + self.feature_min

        # Build output dictionary
        features = {}
        for i, name in enumerate(self.feature_names):
            features[name] = scaled[:, i]

        return features

    def get_normalized_features(self, emotion_embed: torch.Tensor) -> torch.Tensor:
        """
        Get normalized features (0-1 range) for loss computation.

        Args:
            emotion_embed: Emotion embedding (B, emotion_dim)

        Returns:
            Normalized features tensor (B, num_features)
        """
        if emotion_embed.dim() == 3:
            emotion_embed = emotion_embed.mean(dim=1)

        raw = self.prosody_head(emotion_embed)
        return torch.sigmoid(raw)


# =============================================================================
# Prosody Extraction
# =============================================================================

class ProsodyExtractor:
    """
    Extracts prosodic features from audio waveforms.

    Uses librosa for pitch and energy extraction.
    Falls back to simple energy-based features if librosa is unavailable.

    Example:
        >>> extractor = ProsodyExtractor(sr=24000)
        >>> audio = torch.randn(1, 24000)  # 1 second of audio
        >>> features = extractor.extract(audio)
        >>> print(features["pitch_mean"])
    """

    def __init__(self, sr: int = 24000):
        """
        Args:
            sr: Audio sample rate (default: 24000 for Chatterbox)
        """
        self.sr = sr

        if not HAS_LIBROSA:
            import warnings
            warnings.warn(
                "librosa not available. ProsodyExtractor will use simplified features."
            )

    def extract(
        self,
        audio: torch.Tensor,
        target_sr: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract prosodic features from audio.

        Args:
            audio: Audio waveform (B, T) or (T,)
            target_sr: Sample rate of input audio (if different from self.sr)

        Returns:
            Dictionary of prosodic features (values are scalars or (B,) tensors)
        """
        # Handle input format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = audio.device
        batch_size = audio.shape[0]

        # Initialize output
        features = {name: [] for name in PROSODY_FEATURE_RANGES.keys()}

        for i in range(batch_size):
            waveform = audio[i].cpu().numpy()

            if HAS_LIBROSA:
                feat = self._extract_with_librosa(waveform, target_sr or self.sr)
            else:
                feat = self._extract_simple(waveform)

            for name in features:
                features[name].append(feat.get(name, 0.0))

        # Convert to tensors
        for name in features:
            features[name] = torch.tensor(features[name], device=device)

        return features

    def _extract_with_librosa(
        self,
        waveform: "np.ndarray",
        sr: int,
    ) -> Dict[str, float]:
        """Extract features using librosa."""
        import librosa
        import numpy as np

        features = {}

        # Resample if needed
        if sr != self.sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr)
            sr = self.sr

        # Pitch extraction (using pyin for better accuracy)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
            )
            f0_valid = f0[~np.isnan(f0)]

            if len(f0_valid) > 0:
                features["pitch_mean"] = float(np.mean(f0_valid))
                features["pitch_std"] = float(np.std(f0_valid))
                features["pitch_range"] = float(np.max(f0_valid) - np.min(f0_valid))
            else:
                features["pitch_mean"] = 150.0
                features["pitch_std"] = 20.0
                features["pitch_range"] = 50.0
        except Exception:
            features["pitch_mean"] = 150.0
            features["pitch_std"] = 20.0
            features["pitch_range"] = 50.0

        # Energy extraction
        rms = librosa.feature.rms(y=waveform)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))

        # Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
            # Normalize to 0.5-2.0 range (assuming 120 BPM is baseline)
            features["tempo"] = float(np.clip(tempo / 120.0, 0.5, 2.0))
        except Exception:
            features["tempo"] = 1.0

        # Duration (relative to 1 second baseline)
        duration_sec = len(waveform) / sr
        features["duration"] = float(np.clip(duration_sec, 0.5, 2.0))

        return features

    def _extract_simple(self, waveform: "np.ndarray") -> Dict[str, float]:
        """Simple feature extraction without librosa."""
        import numpy as np

        features = {}

        # Energy-based features
        rms = np.sqrt(np.mean(waveform ** 2))
        features["energy_mean"] = float(rms)

        # Estimate energy variation using windowed RMS
        window_size = len(waveform) // 10
        if window_size > 0:
            rms_values = []
            for i in range(0, len(waveform) - window_size, window_size):
                window = waveform[i:i + window_size]
                rms_values.append(np.sqrt(np.mean(window ** 2)))
            features["energy_std"] = float(np.std(rms_values)) if rms_values else 0.0
        else:
            features["energy_std"] = 0.0

        # Placeholder values for pitch (requires pitch detection)
        features["pitch_mean"] = 150.0
        features["pitch_std"] = 20.0
        features["pitch_range"] = 50.0

        # Default tempo and duration
        features["tempo"] = 1.0
        features["duration"] = len(waveform) / self.sr

        return features

    def normalize(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Normalize features to 0-1 range.

        Args:
            features: Dictionary of prosodic features

        Returns:
            Normalized features dictionary
        """
        normalized = {}
        for name, value in features.items():
            min_val, max_val = PROSODY_FEATURE_RANGES.get(name, (0.0, 1.0))
            normalized[name] = (value - min_val) / (max_val - min_val + 1e-8)
            normalized[name] = torch.clamp(normalized[name], 0.0, 1.0)
        return normalized


# =============================================================================
# Prosody Matching Loss
# =============================================================================

class ProsodyMatchingLoss(nn.Module):
    """
    Loss that compares predicted prosody with actual audio prosody.

    This loss encourages the model to generate audio with prosodic
    features that match the emotion-predicted targets.

    Example:
        >>> loss_fn = ProsodyMatchingLoss()
        >>> predicted = {"pitch_mean": torch.tensor([200.0]), ...}
        >>> audio = torch.randn(1, 24000)
        >>> loss = loss_fn(predicted, audio)
    """

    def __init__(
        self,
        sr: int = 24000,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            sr: Audio sample rate
            feature_weights: Optional weights for each feature (default: uniform)
        """
        super().__init__()
        self.extractor = ProsodyExtractor(sr=sr)

        # Default weights emphasize pitch and energy
        self.feature_weights = feature_weights or {
            "pitch_mean": 1.5,
            "pitch_std": 1.0,
            "pitch_range": 1.0,
            "energy_mean": 1.5,
            "energy_std": 1.0,
            "tempo": 0.5,
            "duration": 0.5,
        }

    def forward(
        self,
        predicted_prosody: Dict[str, torch.Tensor],
        audio: torch.Tensor,
        audio_sr: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute prosody matching loss.

        Args:
            predicted_prosody: Predicted prosodic features from EmotionProsodyPredictor
            audio: Audio waveform (B, T)
            audio_sr: Sample rate of audio (if different from extractor's sr)

        Returns:
            Scalar loss tensor
        """
        # Extract actual prosody from audio
        actual_prosody = self.extractor.extract(audio, target_sr=audio_sr)

        # Normalize both for comparison
        pred_norm = self._normalize(predicted_prosody)
        actual_norm = self.extractor.normalize(actual_prosody)

        # Compute weighted MSE loss
        total_loss = 0.0
        total_weight = 0.0

        for name, weight in self.feature_weights.items():
            if name in pred_norm and name in actual_norm:
                loss = F.mse_loss(pred_norm[name], actual_norm[name])
                total_loss = total_loss + weight * loss
                total_weight += weight

        if total_weight > 0:
            return total_loss / total_weight
        return torch.tensor(0.0, device=audio.device)

    def _normalize(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Normalize predicted features to 0-1 range."""
        normalized = {}
        for name, value in features.items():
            min_val, max_val = PROSODY_FEATURE_RANGES.get(name, (0.0, 1.0))
            normalized[name] = (value - min_val) / (max_val - min_val + 1e-8)
            normalized[name] = torch.clamp(normalized[name], 0.0, 1.0)
        return normalized


# =============================================================================
# Emotion-Prosody Target Loss
# =============================================================================

class EmotionProsodyTargetLoss(nn.Module):
    """
    Loss that encourages predicted prosody to match emotion-specific targets.

    This loss uses predefined prosody profiles for each emotion and
    penalizes deviations from the expected prosodic pattern.

    Example:
        >>> loss_fn = EmotionProsodyTargetLoss()
        >>> predictor = EmotionProsodyPredictor()
        >>> emotion_embed = torch.randn(1, 64)
        >>> loss = loss_fn(predictor, emotion_embed, ["happy"])
    """

    def __init__(self):
        super().__init__()

        # Build target tensor for each emotion
        emotion_names = list(EMOTION_PROSODY_TARGETS.keys())
        feature_names = list(PROSODY_FEATURE_RANGES.keys())

        targets = torch.zeros(len(emotion_names), len(feature_names))
        for i, emotion in enumerate(emotion_names):
            profile = EMOTION_PROSODY_TARGETS.get(emotion, EMOTION_PROSODY_TARGETS["neutral"])
            for j, feat in enumerate(feature_names):
                targets[i, j] = profile.get(feat, 0.5)

        self.register_buffer("targets", targets)
        self.emotion_to_idx = {e: i for i, e in enumerate(emotion_names)}
        self.feature_names = feature_names

    def forward(
        self,
        predictor: EmotionProsodyPredictor,
        emotion_embed: torch.Tensor,
        emotion_names: list,
    ) -> torch.Tensor:
        """
        Compute loss between predicted prosody and emotion targets.

        Args:
            predictor: EmotionProsodyPredictor module
            emotion_embed: Emotion embeddings (B, emotion_dim)
            emotion_names: List of emotion names for the batch

        Returns:
            Scalar loss tensor
        """
        # Get normalized predictions
        predicted = predictor.get_normalized_features(emotion_embed)  # (B, num_features)

        # Get targets for each emotion in batch
        batch_targets = []
        for emotion in emotion_names:
            emotion_lower = emotion.lower()
            idx = self.emotion_to_idx.get(emotion_lower, self.emotion_to_idx["neutral"])
            batch_targets.append(self.targets[idx])

        targets = torch.stack(batch_targets).to(predicted.device)  # (B, num_features)

        # Compute MSE loss
        return F.mse_loss(predicted, targets)


# =============================================================================
# Factory Functions
# =============================================================================

def create_prosody_predictor(
    emotion_dim: int = 64,
    hidden_dim: int = 128,
) -> EmotionProsodyPredictor:
    """Create a prosody predictor with default settings."""
    return EmotionProsodyPredictor(
        emotion_dim=emotion_dim,
        hidden_dim=hidden_dim,
    )


def create_prosody_loss(
    sr: int = 24000,
    use_audio_matching: bool = True,
    use_target_matching: bool = True,
    audio_weight: float = 0.5,
    target_weight: float = 0.5,
) -> nn.Module:
    """
    Create a combined prosody loss module.

    Args:
        sr: Audio sample rate
        use_audio_matching: Whether to use audio prosody matching
        use_target_matching: Whether to use emotion target matching
        audio_weight: Weight for audio matching loss
        target_weight: Weight for target matching loss

    Returns:
        Loss module (ProsodyMatchingLoss or combined)
    """
    losses = {}
    weights = {}

    if use_audio_matching:
        losses["audio"] = ProsodyMatchingLoss(sr=sr)
        weights["audio"] = audio_weight

    if use_target_matching:
        losses["target"] = EmotionProsodyTargetLoss()
        weights["target"] = target_weight

    if len(losses) == 1:
        return list(losses.values())[0]

    # Combined loss wrapper
    class CombinedProsodyLoss(nn.Module):
        def __init__(self, losses, weights):
            super().__init__()
            self.losses = nn.ModuleDict(losses)
            self.weights = weights

        def forward(self, **kwargs):
            total = 0.0
            for name, loss_fn in self.losses.items():
                if name == "audio":
                    loss = loss_fn(
                        kwargs.get("predicted_prosody"),
                        kwargs.get("audio"),
                    )
                elif name == "target":
                    loss = loss_fn(
                        kwargs.get("predictor"),
                        kwargs.get("emotion_embed"),
                        kwargs.get("emotion_names"),
                    )
                else:
                    continue
                total = total + self.weights[name] * loss
            return total

    return CombinedProsodyLoss(losses, weights)
