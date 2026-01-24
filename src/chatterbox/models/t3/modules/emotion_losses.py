"""
Emotion-specific loss functions for training.

This module provides losses to improve emotion-audio alignment during training:
- Emotion consistency loss: Ensure generated audio matches intended emotion
- Contrastive loss: Better discrimination between different emotions
- Optional SER integration: Verify emotion using pretrained Speech Emotion Recognition

These losses complement the standard TTS cross-entropy loss to produce
more emotionally accurate speech synthesis.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionConsistencyLoss(nn.Module):
    """
    Ensure generated audio features match intended emotion embedding.

    This loss projects audio features to emotion space and computes both
    MSE (for alignment) and contrastive loss (for discrimination).

    The audio features should come from an encoder that extracts prosodic
    information from the generated/reconstructed audio.

    Example:
        >>> loss_fn = EmotionConsistencyLoss(emotion_dim=64)
        >>> target_emotion = torch.randn(4, 64)  # Intended emotion
        >>> audio_features = torch.randn(4, 256)  # From audio encoder
        >>> losses = loss_fn(target_emotion, audio_features)
        >>> print(losses["total_loss"])
    """

    def __init__(
        self,
        emotion_dim: int = 64,
        audio_feature_dim: int = 256,
        hidden_dim: int = 128,
        temperature: float = 0.07,
    ):
        """
        Args:
            emotion_dim: Dimension of emotion embeddings (default: 64)
            audio_feature_dim: Dimension of audio features (default: 256)
            hidden_dim: Hidden dimension for projection (default: 128)
            temperature: Temperature for contrastive loss (default: 0.07)
        """
        super().__init__()
        self.emotion_dim = emotion_dim
        self.audio_feature_dim = audio_feature_dim

        # Project audio features to emotion space
        self.audio_to_emotion = nn.Sequential(
            nn.Linear(audio_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # Learnable temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        target_emotion_embed: torch.Tensor,  # (B, emotion_dim) - intended emotion
        audio_features: torch.Tensor,        # (B, audio_feature_dim) - from audio
        labels: Optional[torch.Tensor] = None,  # (B,) - emotion class indices
    ) -> Dict[str, torch.Tensor]:
        """
        Compute consistency loss between emotion embedding and audio.

        Args:
            target_emotion_embed: Intended emotion embedding (B, emotion_dim)
            audio_features: Features extracted from audio (B, audio_feature_dim)
            labels: Optional emotion class indices for additional supervision

        Returns:
            Dictionary with:
            - consistency_loss: MSE between projected audio and target emotion
            - contrastive_loss: InfoNCE loss for better discrimination
            - total_loss: Weighted combination
        """
        B = target_emotion_embed.size(0)
        device = target_emotion_embed.device

        # Project audio to emotion space
        predicted_emotion = self.audio_to_emotion(audio_features)  # (B, emotion_dim)

        # L2 consistency loss (MSE)
        consistency_loss = F.mse_loss(predicted_emotion, target_emotion_embed)

        # Contrastive loss for better emotion separation
        # Normalize embeddings for cosine similarity
        pred_norm = F.normalize(predicted_emotion, dim=-1)
        target_norm = F.normalize(target_emotion_embed, dim=-1)

        # Compute similarity matrix
        # Each prediction should be most similar to its corresponding target
        sim_matrix = torch.mm(pred_norm, target_norm.t()) / self.temperature.clamp(min=0.01)

        # InfoNCE loss: predict which target each prediction matches
        contrastive_loss = F.cross_entropy(
            sim_matrix,
            torch.arange(B, device=device)
        )

        # Combined loss
        total_loss = consistency_loss + 0.5 * contrastive_loss

        return {
            "consistency_loss": consistency_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": total_loss,
        }


class EmotionDiscriminatorLoss(nn.Module):
    """
    Discriminator loss for adversarial emotion training.

    This can be used to ensure the emotion embedding pathway carries
    actual emotion information rather than speaker-specific patterns.

    The discriminator tries to predict the emotion class from the
    conditioning, while the generator tries to fool it.
    """

    def __init__(
        self,
        emotion_dim: int = 64,
        num_emotions: int = 16,
        hidden_dim: int = 128,
    ):
        """
        Args:
            emotion_dim: Dimension of emotion embeddings (default: 64)
            num_emotions: Number of emotion classes (default: 16)
            hidden_dim: Hidden dimension for discriminator (default: 128)
        """
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_emotions),
        )

    def forward(
        self,
        emotion_embed: torch.Tensor,  # (B, emotion_dim)
        emotion_labels: torch.Tensor,  # (B,) - emotion class indices
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss.

        Args:
            emotion_embed: Emotion embedding (B, emotion_dim)
            emotion_labels: Ground truth emotion indices (B,)

        Returns:
            Dictionary with discriminator_loss
        """
        logits = self.discriminator(emotion_embed)
        loss = F.cross_entropy(logits, emotion_labels)

        # Compute accuracy for monitoring
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == emotion_labels).float().mean()

        return {
            "discriminator_loss": loss,
            "discriminator_accuracy": accuracy,
        }


class SERIntegrationLoss(nn.Module):
    """
    Optional integration with pretrained Speech Emotion Recognition model.

    Uses a frozen SER model to verify that generated audio actually
    sounds like the intended emotion. This provides external validation
    beyond the internal consistency loss.

    Supports models like:
    - wav2vec2-emotion
    - HuBERT-emotion
    - Custom SER models

    Note: This requires the transformers library and a compatible SER model.
    """

    def __init__(
        self,
        ser_model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        emotion_mapping: Optional[Dict[str, int]] = None,
        freeze_ser: bool = True,
    ):
        """
        Args:
            ser_model_name: HuggingFace model name for SER
            emotion_mapping: Mapping from our emotion names to SER model classes
            freeze_ser: Whether to freeze SER model weights (default: True)
        """
        super().__init__()
        self.ser_model_name = ser_model_name
        self.freeze_ser = freeze_ser

        # Lazy loading to avoid import issues if transformers not installed
        self._ser_model = None
        self._ser_processor = None
        self._model_loaded = False
        self._debug_printed = False

        # Optional mapping from our emotions to SER model classes.
        # If not provided, we will derive it from the SER model config.
        self.emotion_mapping = emotion_mapping

    def _load_ser_model(self, device: torch.device) -> bool:
        """Lazy load SER model."""
        if self._model_loaded:
            return self._ser_model is not None

        try:
            from transformers import (
                AutoModelForAudioClassification,
                AutoProcessor,
                AutoFeatureExtractor,
            )

            print(f"Loading SER model: {self.ser_model_name}")
            try:
                self._ser_processor = AutoProcessor.from_pretrained(self.ser_model_name)
            except Exception:
                # Some audio classification models don't ship a tokenizer; use feature extractor
                self._ser_processor = AutoFeatureExtractor.from_pretrained(self.ser_model_name)

            self._ser_model = AutoModelForAudioClassification.from_pretrained(
                self.ser_model_name
            ).to(device)

            if self.freeze_ser:
                self._ser_model.eval()
                for param in self._ser_model.parameters():
                    param.requires_grad = False

            # Build emotion mapping from model config if not provided
            if self.emotion_mapping is None:
                label2id = getattr(self._ser_model.config, "label2id", {}) or {}
                if label2id:
                    self.emotion_mapping = {
                        str(label).lower(): int(idx) for label, idx in label2id.items()
                    }
                else:
                    # Fallback to common ordering when config doesn't provide labels
                    self.emotion_mapping = {
                        "angry": 0,
                        "calm": 1,
                        "disgusted": 2,
                        "fearful": 3,
                        "happy": 4,
                        "neutral": 5,
                        "sad": 6,
                        "surprised": 7,
                    }
            else:
                # If provided, fill missing labels from model config when possible
                label2id = getattr(self._ser_model.config, "label2id", {}) or {}
                for label, idx in label2id.items():
                    self.emotion_mapping.setdefault(str(label).lower(), int(idx))

            print(f"SER label mapping: {self.emotion_mapping}")

            self._model_loaded = True
            print("SER model loaded successfully")
            return True

        except ImportError:
            print("Warning: transformers library not installed. SER loss disabled.")
            self._model_loaded = True
            return False

        except Exception as e:
            print(f"Warning: Could not load SER model: {e}")
            self._model_loaded = True
            return False

    def forward(
        self,
        audio: torch.Tensor,         # (B, num_samples) at 16kHz
        target_emotions: List[str],  # List of emotion names
        sample_rate: int = 16000,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SER-based emotion verification loss.

        Args:
            audio: Audio waveform (B, num_samples) at 16kHz
            target_emotions: List of target emotion names
            sample_rate: Audio sample rate (default: 16000)

        Returns:
            Dictionary with ser_loss and ser_accuracy
        """
        device = audio.device

        if not self._load_ser_model(device):
            return {
                "ser_loss": torch.tensor(0.0, device=device),
                "ser_accuracy": torch.tensor(0.0, device=device),
            }

        if not self.emotion_mapping:
            print("Warning: SER emotion mapping is empty. SER loss disabled.")
            return {
                "ser_loss": torch.tensor(0.0, device=device),
                "ser_accuracy": torch.tensor(0.0, device=device),
            }

        # Process audio for SER
        context = torch.no_grad() if self.freeze_ser else torch.enable_grad()

        with context:
            # Convert to numpy for processor
            audio_np = audio.cpu().numpy()

            # Process through SER model
            inputs = self._ser_processor(
                audio_np,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            ).to(device)

            logits = self._ser_model(**inputs).logits  # (B, num_classes)

        # Convert target emotions to indices
        target_indices = []
        for emotion in target_emotions:
            emotion_lower = emotion.lower()
            if emotion_lower in self.emotion_mapping:
                target_indices.append(self.emotion_mapping[emotion_lower])
            else:
                # Default to neutral if unknown, else first available label
                fallback = self.emotion_mapping.get("neutral")
                if fallback is None:
                    fallback = next(iter(self.emotion_mapping.values()))
                target_indices.append(fallback)

        target_tensor = torch.tensor(target_indices, device=device)

        # Cross-entropy loss
        ser_loss = F.cross_entropy(logits, target_tensor)

        # Compute accuracy for monitoring
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == target_tensor).float().mean()

        if not self._debug_printed:
            self._debug_printed = True
            # Log a small sample of targets vs predictions for sanity check
            target_names = []
            pred_names = []
            inv_mapping = {v: k for k, v in self.emotion_mapping.items()}
            for idx in target_tensor[:8].tolist():
                target_names.append(inv_mapping.get(idx, str(idx)))
            for idx in predictions[:8].tolist():
                pred_names.append(inv_mapping.get(idx, str(idx)))
            print(
                "SER debug - targets:",
                target_names,
                "predictions:",
                pred_names,
            )

        return {
            "ser_loss": ser_loss,
            "ser_accuracy": accuracy,
        }


class CombinedEmotionLoss(nn.Module):
    """
    Combined loss function for emotion-aware TTS training.

    Combines:
    - Standard TTS loss (text + speech cross-entropy)
    - Emotion consistency loss
    - Optional SER verification loss
    - Optional discriminator loss

    Usage:
        >>> loss_fn = CombinedEmotionLoss(emotion_dim=64, use_ser=False)
        >>> losses = loss_fn(
        ...     tts_loss=tts_loss,
        ...     emotion_embed=emotion_embed,
        ...     audio_features=audio_features,
        ... )
        >>> losses["total_loss"].backward()
    """

    def __init__(
        self,
        emotion_dim: int = 64,
        audio_feature_dim: int = 256,
        use_ser: bool = False,
        use_discriminator: bool = False,
        num_emotions: int = 16,
        consistency_weight: float = 0.5,
        ser_weight: float = 0.3,
        discriminator_weight: float = 0.1,
    ):
        """
        Args:
            emotion_dim: Dimension of emotion embeddings
            audio_feature_dim: Dimension of audio features
            use_ser: Whether to use SER verification loss
            use_discriminator: Whether to use discriminator loss
            num_emotions: Number of emotion classes
            consistency_weight: Weight for consistency loss
            ser_weight: Weight for SER loss
            discriminator_weight: Weight for discriminator loss
        """
        super().__init__()

        self.consistency_loss_fn = EmotionConsistencyLoss(
            emotion_dim=emotion_dim,
            audio_feature_dim=audio_feature_dim,
        )

        self.use_ser = use_ser
        if use_ser:
            self.ser_loss_fn = SERIntegrationLoss()

        self.use_discriminator = use_discriminator
        if use_discriminator:
            self.discriminator_loss_fn = EmotionDiscriminatorLoss(
                emotion_dim=emotion_dim,
                num_emotions=num_emotions,
            )

        self.consistency_weight = consistency_weight
        self.ser_weight = ser_weight
        self.discriminator_weight = discriminator_weight

    def forward(
        self,
        tts_loss: torch.Tensor,
        emotion_embed: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        target_emotions: Optional[List[str]] = None,
        emotion_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            tts_loss: Base TTS loss (text + speech cross-entropy)
            emotion_embed: Emotion embedding used for conditioning
            audio_features: Features extracted from audio (for consistency loss)
            audio: Raw audio waveform (for SER loss)
            target_emotions: List of emotion names (for SER loss)
            emotion_labels: Emotion class indices (for discriminator loss)

        Returns:
            Dictionary with all loss components and total_loss
        """
        losses = {"tts_loss": tts_loss}
        total_loss = tts_loss

        # Consistency loss (if audio features provided)
        if audio_features is not None:
            consistency_out = self.consistency_loss_fn(emotion_embed, audio_features)
            losses["emotion_consistency_loss"] = consistency_out["consistency_loss"]
            losses["emotion_contrastive_loss"] = consistency_out["contrastive_loss"]
            total_loss = total_loss + self.consistency_weight * consistency_out["total_loss"]

        # SER loss (if enabled and audio provided)
        if self.use_ser and audio is not None and target_emotions is not None:
            ser_out = self.ser_loss_fn(audio, target_emotions)
            losses["ser_loss"] = ser_out["ser_loss"]
            losses["ser_accuracy"] = ser_out["ser_accuracy"]
            total_loss = total_loss + self.ser_weight * ser_out["ser_loss"]

        # Discriminator loss (if enabled and labels provided)
        if self.use_discriminator and emotion_labels is not None:
            disc_out = self.discriminator_loss_fn(emotion_embed, emotion_labels)
            losses["discriminator_loss"] = disc_out["discriminator_loss"]
            losses["discriminator_accuracy"] = disc_out["discriminator_accuracy"]
            total_loss = total_loss + self.discriminator_weight * disc_out["discriminator_loss"]

        losses["total_loss"] = total_loss
        return losses
