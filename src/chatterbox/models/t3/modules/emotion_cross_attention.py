"""
Emotion Cross-Attention module for T3 model.

This module implements cross-attention between emotion embeddings and text/speech context,
allowing the emotion to modulate based on the input content. This provides richer
emotion conditioning compared to simple concatenation.

Architecture:
- Projects 64D emotion embeddings to model dimension (1024)
- Uses 4 learnable query tokens to capture different aspects of emotion
- Cross-attention: emotion queries attend to text context
- Self-attention: refines the emotion representation
- FFN: final transformation

The output is a (B, 4, 1024) tensor that is concatenated with other conditioning.
"""

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class EmotionCrossAttention(nn.Module):
    """
    Cross-attention module for emotion conditioning.

    Emotion embeddings attend to text/speech context to create
    context-aware emotion representations.

    Args:
        hidden_size: Model hidden dimension (default: 1024)
        emotion_dim: Input emotion embedding dimension (default: 64)
        num_heads: Number of attention heads (default: 8)
        num_query_tokens: Number of learnable query tokens (default: 4)
        dropout: Dropout rate (default: 0.1)
        use_flash_attention: Whether to use flash attention when available (default: True)

    Example:
        >>> cross_attn = EmotionCrossAttention(hidden_size=1024, emotion_dim=64)
        >>> emotion_embed = torch.randn(2, 64)  # (B, emotion_dim)
        >>> text_context = torch.randn(2, 50, 1024)  # (B, seq_len, hidden)
        >>> output = cross_attn(emotion_embed, context=text_context)  # (B, 4, 1024)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        emotion_dim: int = 64,
        num_heads: int = 8,
        num_query_tokens: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.emotion_dim = emotion_dim
        self.num_heads = num_heads
        self.num_query_tokens = num_query_tokens
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # Project emotion embedding to hidden size
        self.emotion_proj = nn.Linear(emotion_dim, hidden_size)

        # Learnable query tokens for capturing different emotion aspects
        # These represent: pitch, energy, rate, voice quality (approximately)
        self.query_tokens = nn.Parameter(
            torch.empty(1, num_query_tokens, hidden_size)
        )
        # Initialize with scaled uniform distribution
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (num_query_tokens + hidden_size))
        self.query_tokens.data.uniform_(-query_variance, query_variance)

        # Cross-attention layers (queries attend to context)
        self.cross_q_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_k_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_v_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_out_proj = nn.Linear(hidden_size, hidden_size)

        # Self-attention layers (for query refinement)
        self.self_q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_out_proj = nn.Linear(hidden_size, hidden_size)

        # Layer normalization
        self.norm_pre = nn.LayerNorm(hidden_size)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.norm_self = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into attention heads: (B, L, D) -> (B, H, L, head_dim)"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine attention heads: (B, H, L, head_dim) -> (B, L, D)"""
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor (B, H, Lq, head_dim)
            k: Key tensor (B, H, Lk, head_dim)
            v: Value tensor (B, H, Lk, head_dim)
            mask: Optional attention mask

        Returns:
            Attention output (B, H, Lq, head_dim)
        """
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's efficient attention implementation
            try:
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    scale=self.scale,
                )
            except RuntimeError:
                # Fall back to manual implementation if flash attention fails
                pass

        # Manual attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _cross_attention(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention: queries attend to context.

        Args:
            queries: Query tensor (B, Lq, D)
            context: Context tensor (B, Lc, D)
            mask: Optional attention mask

        Returns:
            Attended queries (B, Lq, D)
        """
        # Normalize
        queries_norm = self.norm_cross(queries)
        context_norm = self.norm_cross(context)

        # Project to Q, K, V
        q = self._split_heads(self.cross_q_proj(queries_norm))
        k = self._split_heads(self.cross_k_proj(context_norm))
        v = self._split_heads(self.cross_v_proj(context_norm))

        # Attention
        attn_output = self._attention(q, k, v, mask)
        attn_output = self._combine_heads(attn_output)
        attn_output = self.cross_out_proj(attn_output)

        # Residual connection
        return queries + self.dropout(attn_output)

    def _self_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Self-attention for query refinement.

        Args:
            x: Input tensor (B, L, D)
            mask: Optional attention mask

        Returns:
            Refined tensor (B, L, D)
        """
        # Normalize
        x_norm = self.norm_self(x)

        # Project to Q, K, V
        q = self._split_heads(self.self_q_proj(x_norm))
        k = self._split_heads(self.self_k_proj(x_norm))
        v = self._split_heads(self.self_v_proj(x_norm))

        # Attention
        attn_output = self._attention(q, k, v, mask)
        attn_output = self._combine_heads(attn_output)
        attn_output = self.self_out_proj(attn_output)

        # Residual connection
        return x + self.dropout(attn_output)

    def forward(
        self,
        emotion_embed: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for emotion cross-attention.

        Args:
            emotion_embed: Emotion embedding tensor (B, emotion_dim) or (B, 1, emotion_dim)
            context: Optional context tensor (B, L, hidden_size) from text embeddings.
                    If None, only self-attention is applied.

        Returns:
            Emotion conditioning tensor (B, num_query_tokens, hidden_size)
        """
        # Handle input shape
        if emotion_embed.dim() == 3:
            # (B, 1, emotion_dim) -> (B, emotion_dim)
            emotion_embed = emotion_embed.squeeze(1)

        batch_size = emotion_embed.shape[0]

        # Project emotion to hidden size
        emotion_proj = self.emotion_proj(emotion_embed)  # (B, hidden_size)
        emotion_proj = self.norm_pre(emotion_proj)
        emotion_proj = emotion_proj.unsqueeze(1)  # (B, 1, hidden_size)

        # Expand query tokens for batch and add emotion information
        queries = self.query_tokens.expand(batch_size, -1, -1)  # (B, num_query, hidden_size)
        queries = queries + emotion_proj  # Broadcast emotion to all queries

        # Cross-attention to context if provided
        if context is not None and context.shape[1] > 0:
            queries = self._cross_attention(queries, context)

        # Self-attention for refinement
        queries = self._self_attention(queries)

        # Feed-forward network
        queries = queries + self.ffn(self.norm_ffn(queries))

        return queries  # (B, num_query_tokens, hidden_size)


def create_emotion_cross_attention(
    hidden_size: int = 1024,
    emotion_dim: int = 64,
    num_heads: int = 8,
    num_query_tokens: int = 4,
) -> EmotionCrossAttention:
    """Factory function to create emotion cross-attention module."""
    return EmotionCrossAttention(
        hidden_size=hidden_size,
        emotion_dim=emotion_dim,
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
    )
