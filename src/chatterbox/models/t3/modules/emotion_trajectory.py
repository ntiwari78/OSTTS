"""
EmotionTrajectory module for temporal emotion dynamics.

This module allows emotions to evolve over the duration of an utterance,
supporting patterns like:
- Gradual intensification
- Emotion transitions (sad -> hopeful)
- Emotional arcs (calm -> excited -> calm)

Features:
- Static mode: Same emotion throughout (backward compatible)
- Linear transition: Smooth interpolation between start/end emotions
- Keyframe mode: Multiple emotion keyframes with learned interpolation
- Text-aware transitions: Cross-attention to text for context-aware dynamics
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmotionKeyframe:
    """A single emotion state at a relative position in the utterance."""
    position: float  # 0.0 = start, 1.0 = end
    emotion: str
    intensity: float = 1.0


class EmotionTrajectory(nn.Module):
    """
    Generate time-varying emotion embeddings for an utterance.

    Supports three modes:
    1. Static: Same emotion throughout (backward compatible)
    2. Linear transition: Smooth interpolation between start/end emotions
    3. Keyframe: Multiple emotion keyframes with learned interpolation

    The module uses learned interpolation for smooth, natural-sounding
    emotion transitions that avoid abrupt changes.

    Example:
        >>> trajectory = EmotionTrajectory(emotion_dim=64)
        >>> start = torch.randn(1, 64)  # Happy embedding
        >>> end = torch.randn(1, 64)    # Sad embedding
        >>> # Generate trajectory over 100 time steps
        >>> emotions = trajectory.forward_transition(start, end, seq_len=100)
        >>> print(emotions.shape)  # (1, 100, 64)
    """

    def __init__(
        self,
        emotion_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_keyframes: int = 5,
        text_hidden_size: int = 1024,
    ):
        """
        Args:
            emotion_dim: Dimension of emotion embeddings (default: 64)
            hidden_dim: Hidden dimension for interpolation network (default: 128)
            num_heads: Number of attention heads for text cross-attention (default: 4)
            max_keyframes: Maximum number of keyframes supported (default: 5)
            text_hidden_size: Dimension of text context from T3 (default: 1024)
        """
        super().__init__()
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        self.max_keyframes = max_keyframes

        # Position encoding for relative time (0-1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # Interpolation network for smooth transitions
        # Input: start embedding + end embedding + position = 2*emotion_dim + 1
        self.interpolation_net = nn.Sequential(
            nn.Linear(emotion_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # Cross-attention for text-aware trajectory (optional)
        self.text_proj = nn.Linear(text_hidden_size, emotion_dim)
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=emotion_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.text_attn_norm = nn.LayerNorm(emotion_dim)

        # Learnable transition smoothness parameter
        self.transition_sharpness = nn.Parameter(torch.tensor(2.0))

        # Layer norm for output
        self.output_norm = nn.LayerNorm(emotion_dim)

    def forward_static(
        self,
        emotion_embed: torch.Tensor,  # (B, emotion_dim)
        seq_len: int,
    ) -> torch.Tensor:
        """
        Static mode: repeat same embedding for all positions.

        This maintains backward compatibility with non-trajectory usage.

        Args:
            emotion_embed: Single emotion embedding (B, emotion_dim)
            seq_len: Number of time steps

        Returns:
            Repeated embedding (B, seq_len, emotion_dim)
        """
        if emotion_embed.dim() == 1:
            emotion_embed = emotion_embed.unsqueeze(0)
        return emotion_embed.unsqueeze(1).expand(-1, seq_len, -1)

    def forward_transition(
        self,
        start_embed: torch.Tensor,  # (B, emotion_dim)
        end_embed: torch.Tensor,    # (B, emotion_dim)
        seq_len: int,
        text_context: Optional[torch.Tensor] = None,  # (B, text_len, text_hidden_size)
    ) -> torch.Tensor:
        """
        Transition mode: learned interpolation between start and end emotions.

        The interpolation uses a learned network to produce smooth, natural
        transitions that may follow non-linear paths through emotion space.

        Args:
            start_embed: Start emotion embedding (B, emotion_dim)
            end_embed: End emotion embedding (B, emotion_dim)
            seq_len: Number of time steps
            text_context: Optional text embeddings for context-aware transitions

        Returns:
            Emotion embeddings for each position (B, seq_len, emotion_dim)
        """
        # Handle dimension
        if start_embed.dim() == 1:
            start_embed = start_embed.unsqueeze(0)
        if end_embed.dim() == 1:
            end_embed = end_embed.unsqueeze(0)

        B = start_embed.size(0)
        device = start_embed.device

        # Create relative positions [0, 1]
        positions = torch.linspace(0, 1, seq_len, device=device)  # (seq_len,)
        positions = positions.unsqueeze(0).expand(B, -1)  # (B, seq_len)

        # Expand embeddings for each position
        start_expanded = start_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, emotion_dim)
        end_expanded = end_embed.unsqueeze(1).expand(-1, seq_len, -1)      # (B, seq_len, emotion_dim)
        positions_expanded = positions.unsqueeze(-1)  # (B, seq_len, 1)

        # Concatenate for interpolation network
        interp_input = torch.cat([
            start_expanded,
            end_expanded,
            positions_expanded,
        ], dim=-1)  # (B, seq_len, emotion_dim*2 + 1)

        # Get interpolated embeddings via learned network
        trajectory = self.interpolation_net(interp_input)  # (B, seq_len, emotion_dim)

        # Add time-based modulation
        time_mod = self.time_embed(positions_expanded)  # (B, seq_len, emotion_dim)
        trajectory = trajectory + 0.1 * time_mod  # Small contribution

        # Optional: modulate based on text context
        if text_context is not None:
            trajectory = self._apply_text_attention(trajectory, text_context)

        # Normalize output
        trajectory = self.output_norm(trajectory)

        return trajectory

    def forward_keyframes(
        self,
        keyframe_embeds: List[torch.Tensor],  # List of (B, emotion_dim)
        keyframe_positions: List[float],       # List of positions [0, 1]
        seq_len: int,
        text_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Keyframe mode: interpolate through multiple keyframes.

        Supports complex emotional arcs like:
        - calm (0.0) -> excited (0.5) -> calm (1.0)
        - neutral (0.0) -> angry (0.3) -> sad (0.7) -> hopeful (1.0)

        Args:
            keyframe_embeds: List of emotion embeddings for each keyframe
            keyframe_positions: List of relative positions [0, 1] for each keyframe
            seq_len: Number of time steps
            text_context: Optional text embeddings for context-aware transitions

        Returns:
            Emotion embeddings for each position (B, seq_len, emotion_dim)
        """
        if len(keyframe_embeds) != len(keyframe_positions):
            raise ValueError("Number of keyframe embeddings must match number of positions")

        if len(keyframe_embeds) < 2:
            raise ValueError("At least 2 keyframes required")

        if len(keyframe_embeds) > self.max_keyframes:
            raise ValueError(f"Maximum {self.max_keyframes} keyframes supported")

        # Ensure keyframes have batch dimension
        keyframe_embeds = [
            e.unsqueeze(0) if e.dim() == 1 else e
            for e in keyframe_embeds
        ]

        B = keyframe_embeds[0].size(0)
        device = keyframe_embeds[0].device

        # Ensure keyframes are sorted by position
        sorted_pairs = sorted(zip(keyframe_positions, keyframe_embeds), key=lambda x: x[0])
        positions = [p for p, _ in sorted_pairs]
        embeds = [e for _, e in sorted_pairs]

        # Create output trajectory
        output_positions = torch.linspace(0, 1, seq_len, device=device)
        trajectory = torch.zeros(B, seq_len, self.emotion_dim, device=device)

        # For each output position, find surrounding keyframes and interpolate
        for i, pos in enumerate(output_positions):
            pos_val = pos.item()

            # Find surrounding keyframes
            for j in range(len(positions) - 1):
                if positions[j] <= pos_val <= positions[j + 1]:
                    # Compute interpolation weight with smooth step
                    segment_length = positions[j + 1] - positions[j]
                    if segment_length > 0:
                        t = (pos_val - positions[j]) / segment_length
                    else:
                        t = 0.0

                    # Smooth step (Hermite interpolation) for natural transitions
                    # t_smooth = 3t^2 - 2t^3 (S-curve)
                    t_smooth = t * t * (3 - 2 * t)

                    # Interpolate between keyframes
                    trajectory[:, i, :] = (1 - t_smooth) * embeds[j] + t_smooth * embeds[j + 1]
                    break
            else:
                # Position beyond last keyframe - use last keyframe
                trajectory[:, i, :] = embeds[-1]

        # Optional: apply text attention
        if text_context is not None:
            trajectory = self._apply_text_attention(trajectory, text_context)

        # Normalize
        trajectory = self.output_norm(trajectory)

        return trajectory

    def _apply_text_attention(
        self,
        trajectory: torch.Tensor,  # (B, seq_len, emotion_dim)
        text_context: torch.Tensor,  # (B, text_len, text_hidden_size)
    ) -> torch.Tensor:
        """
        Apply cross-attention to text context for context-aware trajectories.

        This allows the emotion trajectory to adapt based on text content,
        e.g., emphasizing emotion at exclamation marks or questions.
        """
        # Project text to emotion dimension
        text_proj = self.text_proj(text_context)  # (B, text_len, emotion_dim)

        # Cross-attention: trajectory attends to text
        trajectory_attended, _ = self.text_cross_attn(
            trajectory,   # query: emotion trajectory
            text_proj,    # key: text context
            text_proj,    # value: text context
        )

        # Residual connection with normalization
        trajectory = self.text_attn_norm(trajectory + 0.2 * trajectory_attended)

        return trajectory

    def forward(
        self,
        start_embed: torch.Tensor,
        end_embed: Optional[torch.Tensor] = None,
        seq_len: int = 1,
        text_context: Optional[torch.Tensor] = None,
        keyframe_embeds: Optional[List[torch.Tensor]] = None,
        keyframe_positions: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Unified forward method that selects appropriate mode.

        Args:
            start_embed: Start/only emotion embedding
            end_embed: End emotion for transition mode (optional)
            seq_len: Number of time steps
            text_context: Text embeddings for context-aware transitions
            keyframe_embeds: List of keyframe embeddings for keyframe mode
            keyframe_positions: List of keyframe positions for keyframe mode

        Returns:
            Emotion trajectory (B, seq_len, emotion_dim)
        """
        # Keyframe mode
        if keyframe_embeds is not None and keyframe_positions is not None:
            return self.forward_keyframes(
                keyframe_embeds,
                keyframe_positions,
                seq_len,
                text_context,
            )

        # Transition mode
        if end_embed is not None:
            return self.forward_transition(
                start_embed,
                end_embed,
                seq_len,
                text_context,
            )

        # Static mode
        return self.forward_static(start_embed, seq_len)
