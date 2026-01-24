"""
Emotion Contrastive Pre-training (Phase 3)

This module provides contrastive learning for emotion embeddings,
making different emotions maximally separable in the embedding space.

Key components:
1. EmotionContrastiveLoss: InfoNCE-style contrastive loss
2. EmotionContrastivePretrainer: Pre-training pipeline
3. AdversarialSERLoss: Adversarial loss to fool SER models

The contrastive loss ensures that:
- Same emotion embeddings are close together
- Different emotion embeddings are far apart
- Emotion boundaries are clear for SER models
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# Contrastive Loss
# =============================================================================

class EmotionContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for emotion embeddings.

    Given a batch of emotion embeddings with labels, this loss:
    - Pulls together embeddings of the same emotion
    - Pushes apart embeddings of different emotions

    Example:
        >>> loss_fn = EmotionContrastiveLoss(temperature=0.07)
        >>> embeddings = torch.randn(8, 64)  # 8 samples, 64-dim
        >>> labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # 4 emotions
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        """
        Args:
            temperature: Temperature for softmax scaling (lower = sharper)
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Emotion embeddings (B, embed_dim)
            labels: Emotion labels (B,) as integers
            mask: Optional mask for valid samples (B,)

        Returns:
            Scalar loss tensor
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask (1 where labels match, excluding diagonal)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        positive_mask = positive_mask - torch.eye(batch_size, device=device)

        # Remove diagonal from similarity (self-similarity)
        logits_mask = 1 - torch.eye(batch_size, device=device)
        similarity = similarity * logits_mask

        # For numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        num_positives = positive_mask.sum(dim=1)
        # Avoid division by zero
        num_positives = torch.clamp(num_positives, min=1)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / num_positives

        # Loss is negative log-likelihood
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss with hard negative mining.

    This variant focuses on hard negatives (similar but different emotions)
    to improve discrimination.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        hard_negative_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss with hard negatives.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Masks
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        neg_mask = 1 - pos_mask

        # Remove diagonal
        diag_mask = 1 - torch.eye(batch_size, device=device)
        pos_mask = pos_mask * diag_mask
        neg_mask = neg_mask * diag_mask

        # Hard negative mining: find closest negative for each sample
        neg_sim = sim * neg_mask + (-1e9) * (1 - neg_mask)
        hardest_neg, _ = neg_sim.max(dim=1, keepdim=True)

        # Positive similarities
        pos_sim = sim * pos_mask + (-1e9) * (1 - pos_mask)

        # Contrastive loss: log(exp(pos) / (exp(pos) + exp(hard_neg)))
        exp_pos = torch.exp(pos_sim) * pos_mask
        exp_hard_neg = torch.exp(hardest_neg) * self.hard_negative_weight

        # Sum positive pairs
        pos_sum = exp_pos.sum(dim=1)
        num_pos = pos_mask.sum(dim=1).clamp(min=1)

        # Log probability
        log_prob = torch.log(pos_sum / (pos_sum + exp_hard_neg.squeeze() + 1e-8) + 1e-8)
        log_prob = log_prob / num_pos

        return -log_prob.mean()


# =============================================================================
# Contrastive Pre-trainer
# =============================================================================

class EmotionContrastivePretrainer:
    """
    Pre-trains emotion embeddings using contrastive learning.

    This aligns emotion embeddings to be maximally separable,
    improving SER model recognition.

    Example:
        >>> from chatterbox.models.t3.modules.emotion_embeddings import EmotionEmbeddings
        >>> embeddings = EmotionEmbeddings()
        >>> pretrainer = EmotionContrastivePretrainer(embeddings)
        >>> pretrainer.pretrain(num_epochs=10)
    """

    def __init__(
        self,
        emotion_embeddings: nn.Module,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 0.07,
        device: str = "cuda",
    ):
        """
        Args:
            emotion_embeddings: EmotionEmbeddings module to pretrain
            projection_dim: Dimension of projection head output
            hidden_dim: Hidden dimension for projection head
            temperature: Contrastive loss temperature
            device: Device for training
        """
        self.embeddings = emotion_embeddings
        self.device = device
        self.temperature = temperature

        # Get embedding dimension
        if hasattr(emotion_embeddings, 'emotion_embed_dim'):
            embed_dim = emotion_embeddings.emotion_embed_dim
        else:
            embed_dim = 64

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        ).to(device)

        # Contrastive loss
        self.contrastive_loss = EmotionContrastiveLoss(temperature=temperature)

        # Move embeddings to device
        self.embeddings.to(device)

    def create_batch(
        self,
        emotions: List[str],
        batch_size: int = 32,
        with_augmentation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a batch of emotion embeddings with labels.

        Args:
            emotions: List of emotion names to sample from
            batch_size: Number of samples per batch
            with_augmentation: Whether to add noise augmentation

        Returns:
            Tuple of (embeddings, labels)
        """
        emotion_to_idx = {e: i for i, e in enumerate(emotions)}

        embeddings = []
        labels = []

        for _ in range(batch_size):
            # Sample random emotion
            emotion = emotions[torch.randint(len(emotions), (1,)).item()]
            label = emotion_to_idx[emotion]

            # Get embedding with random intensity
            intensity = 0.5 + torch.rand(1).item()  # 0.5 to 1.5
            embed = self.embeddings.get_emotion_embedding(
                emotion,
                intensity=intensity,
                device=self.device,
            )

            # Augmentation: add small noise
            if with_augmentation:
                noise = torch.randn_like(embed) * 0.1
                embed = embed + noise

            embeddings.append(embed)
            labels.append(label)

        embeddings = torch.cat(embeddings, dim=0)  # (B, embed_dim)
        labels = torch.tensor(labels, device=self.device)  # (B,)

        return embeddings, labels

    def pretrain(
        self,
        emotions: Optional[List[str]] = None,
        num_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Pre-train emotion embeddings contrastively.

        Args:
            emotions: List of emotions to train on (default: all supported)
            num_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            log_interval: Steps between logging

        Returns:
            Dictionary with training history
        """
        if emotions is None:
            emotions = self.embeddings.get_supported_emotions()

        print(f"Pre-training on {len(emotions)} emotions: {emotions}")

        # Optimizer for embeddings and projection
        params = list(self.embeddings.parameters()) + list(self.projection.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        history = {"loss": [], "epoch_loss": []}
        steps_per_epoch = 100  # Synthetic batches per epoch

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for step in range(steps_per_epoch):
                # Create batch
                embeddings, labels = self.create_batch(
                    emotions,
                    batch_size=batch_size,
                    with_augmentation=True,
                )

                # Project embeddings
                projected = self.projection(embeddings)

                # Compute contrastive loss
                loss = self.contrastive_loss(projected, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                history["loss"].append(loss.item())

                if step % log_interval == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step}: Loss = {loss.item():.4f}")

            avg_loss = epoch_loss / steps_per_epoch
            history["epoch_loss"].append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss = {avg_loss:.4f}")

        return history

    def compute_embedding_distances(
        self,
        emotions: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise distances between emotion embeddings.

        Returns:
            Nested dict of distances[emotion1][emotion2]
        """
        if emotions is None:
            emotions = self.embeddings.get_supported_emotions()

        distances = {}

        for e1 in emotions:
            distances[e1] = {}
            embed1 = self.embeddings.get_emotion_embedding(e1, device=self.device)
            embed1 = F.normalize(embed1, dim=-1)

            for e2 in emotions:
                embed2 = self.embeddings.get_emotion_embedding(e2, device=self.device)
                embed2 = F.normalize(embed2, dim=-1)

                # Cosine distance (1 - similarity)
                similarity = (embed1 * embed2).sum().item()
                distances[e1][e2] = 1 - similarity

        return distances


# =============================================================================
# Adversarial SER Loss
# =============================================================================

class AdversarialSERLoss(nn.Module):
    """
    Adversarial loss that encourages TTS output to be recognized
    by SER models as the target emotion.

    This creates a "generator" that tries to fool the SER model
    into predicting the correct emotion.

    Note: This requires the SER model to be differentiable or uses
    REINFORCE-style gradients.
    """

    def __init__(
        self,
        ser_model: Optional[nn.Module] = None,
        num_emotions: int = 8,
        use_soft_labels: bool = True,
    ):
        """
        Args:
            ser_model: Pre-trained SER model (frozen)
            num_emotions: Number of emotion classes
            use_soft_labels: Use soft labels for smoother gradients
        """
        super().__init__()
        self.ser_model = ser_model
        self.num_emotions = num_emotions
        self.use_soft_labels = use_soft_labels

        # Emotion name to index mapping
        self.emotion_to_idx = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "fearful": 4,
            "disgusted": 5,
            "surprised": 6,
            "calm": 7,
        }

        # Freeze SER model if provided
        if self.ser_model is not None:
            for param in self.ser_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        audio_features: torch.Tensor,
        target_emotions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adversarial SER loss.

        Args:
            audio_features: Audio features from TTS (B, T, D) or (B, D)
            target_emotions: List of target emotion names

        Returns:
            Dictionary with loss and metrics
        """
        device = audio_features.device
        batch_size = audio_features.shape[0]

        if self.ser_model is None:
            return {
                "loss": torch.tensor(0.0, device=device),
                "accuracy": torch.tensor(0.0, device=device),
            }

        # Get target indices
        target_indices = torch.tensor([
            self.emotion_to_idx.get(e.lower(), 0)
            for e in target_emotions
        ], device=device)

        # Get SER predictions
        with torch.no_grad():
            ser_logits = self.ser_model(audio_features)

        # Compute cross-entropy loss (we want low loss = correct prediction)
        if self.use_soft_labels:
            # Soft labels with label smoothing
            soft_targets = torch.zeros(batch_size, self.num_emotions, device=device)
            soft_targets.scatter_(1, target_indices.unsqueeze(1), 0.9)
            soft_targets += 0.1 / self.num_emotions  # Label smoothing

            loss = F.cross_entropy(ser_logits, soft_targets)
        else:
            loss = F.cross_entropy(ser_logits, target_indices)

        # Compute accuracy
        predictions = ser_logits.argmax(dim=-1)
        accuracy = (predictions == target_indices).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "predictions": predictions,
        }


# =============================================================================
# Combined Contrastive + Adversarial Training
# =============================================================================

class CombinedEmotionTrainer:
    """
    Combined training with contrastive and adversarial losses.

    This provides a unified interface for:
    1. Contrastive pre-training of embeddings
    2. Adversarial training against SER models
    3. Standard TTS training
    """

    def __init__(
        self,
        emotion_embeddings: nn.Module,
        contrastive_weight: float = 0.3,
        adversarial_weight: float = 0.2,
        device: str = "cuda",
    ):
        self.embeddings = emotion_embeddings
        self.contrastive_weight = contrastive_weight
        self.adversarial_weight = adversarial_weight
        self.device = device

        # Initialize loss functions
        self.contrastive_loss = EmotionContrastiveLoss()
        self.adversarial_loss = AdversarialSERLoss()

    def compute_loss(
        self,
        emotion_embeds: torch.Tensor,
        emotion_labels: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        target_emotions: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            emotion_embeds: Emotion embeddings (B, D)
            emotion_labels: Emotion label indices (B,)
            audio_features: Optional audio features for adversarial loss
            target_emotions: Optional target emotion names

        Returns:
            Dictionary with losses and total
        """
        losses = {}

        # Contrastive loss
        if self.contrastive_weight > 0:
            contrastive = self.contrastive_loss(emotion_embeds, emotion_labels)
            losses["contrastive"] = contrastive * self.contrastive_weight

        # Adversarial loss
        if self.adversarial_weight > 0 and audio_features is not None:
            adv_result = self.adversarial_loss(audio_features, target_emotions or [])
            losses["adversarial"] = adv_result["loss"] * self.adversarial_weight
            losses["ser_accuracy"] = adv_result["accuracy"]

        # Total
        losses["total"] = sum(v for k, v in losses.items() if k != "ser_accuracy")

        return losses


# =============================================================================
# Factory Functions
# =============================================================================

def create_contrastive_loss(temperature: float = 0.07) -> EmotionContrastiveLoss:
    """Create a contrastive loss with default settings."""
    return EmotionContrastiveLoss(temperature=temperature)


def create_pretrainer(
    emotion_embeddings: nn.Module,
    device: str = "cuda",
) -> EmotionContrastivePretrainer:
    """Create a contrastive pretrainer."""
    return EmotionContrastivePretrainer(
        emotion_embeddings=emotion_embeddings,
        device=device,
    )
