"""
Emotion Cross-Attention module for T3 model (V0.4).

This module implements cross-attention between emotion embeddings and text/speech context,
allowing the emotion to modulate based on the input content. This provides richer
emotion conditioning compared to simple concatenation.

V0.4 Improvements:
- GatedEmotionProjection: Multi-layer gated projection instead of single Linear
- EmotionQueryFusion: FiLM-style modulation (scale * queries + shift)
- EmotionAttentionBias: Emotion-specific attention biases for text
- Expanded to 8 query tokens with semantic grouping
- Better initialization for near-identity start

Architecture:
- Projects 64D emotion embeddings to model dimension (1024) via gated projection
- Uses 8 learnable query tokens (pitch, energy, timing, voice quality)
- FiLM modulation for stronger emotion-query coupling
- Cross-attention: emotion queries attend to text context with emotion-specific bias
- Self-attention: refines the emotion representation
- FFN: final transformation

The output is a (B, 8, 1024) tensor that is concatenated with other conditioning.
"""

import math
from typing import Optional, Dict

import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# V0.4: Gated Emotion Projection
# =============================================================================

class GatedEmotionProjection(nn.Module):
    """
    Multi-layer gated projection for emotion embeddings (V0.4).

    The gate branch learns which dimensions to emphasize,
    while the value branch transforms the representation.
    This preserves fine-grained emotion information better than
    a single linear layer.

    Architecture:
        gate = Sigmoid(Linear(emotion_dim → hidden_size))
        value = GELU(Linear(emotion_dim → hidden_size/2)) → Linear(→ hidden_size)
        output = LayerNorm(gate * value)
    """

    def __init__(self, emotion_dim: int = 64, hidden_size: int = 1024):
        super().__init__()
        self.emotion_dim = emotion_dim
        self.hidden_size = hidden_size

        # Gate branch: learns importance weights
        self.gate_proj = nn.Sequential(
            nn.Linear(emotion_dim, hidden_size),
            nn.Sigmoid()
        )

        # Value branch: transforms representation
        self.value_proj = nn.Sequential(
            nn.Linear(emotion_dim, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

        # Output normalization
        self.norm = nn.LayerNorm(hidden_size)

        # Initialize for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Gate initialized to ~0.5 output (sigmoid(0) = 0.5)
        nn.init.zeros_(self.gate_proj[0].bias)
        nn.init.xavier_uniform_(self.gate_proj[0].weight, gain=0.5)

        # Value branch with small initialization
        for module in self.value_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, emotion_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emotion_embed: (B, emotion_dim) emotion embedding

        Returns:
            (B, hidden_size) projected emotion
        """
        gate = self.gate_proj(emotion_embed)    # (B, hidden_size)
        value = self.value_proj(emotion_embed)  # (B, hidden_size)

        return self.norm(gate * value)


# =============================================================================
# V0.4: FiLM-Style Query Fusion
# =============================================================================

class EmotionQueryFusion(nn.Module):
    """
    FiLM-style modulation for emotion-query coupling (V0.4).

    Feature-wise Linear Modulation: queries = (1 + scale) * queries + shift

    This provides stronger emotion-query coupling than simple addition.
    The scale allows the emotion to amplify/dampen query features,
    while shift allows additive modification.

    Initialized to near-identity (scale≈0, shift≈0) for stable training.
    """

    def __init__(self, hidden_size: int = 1024, num_queries: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries

        # Generate per-query scale from emotion
        self.scale_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_queries * hidden_size),
        )

        # Generate per-query shift from emotion
        self.shift_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_queries * hidden_size),
        )

        # Initialize to near-identity transformation
        self._init_weights()

    def _init_weights(self):
        """Initialize to near-identity (scale≈0, shift≈0)."""
        # Zero out the final layer for identity-like initialization
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)
        nn.init.zeros_(self.shift_net[-1].weight)
        nn.init.zeros_(self.shift_net[-1].bias)

    def forward(
        self,
        queries: torch.Tensor,
        emotion_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to queries.

        Args:
            queries: (B, N, D) learnable query tokens
            emotion_proj: (B, D) projected emotion embedding

        Returns:
            (B, N, D) modulated queries
        """
        B, N, D = queries.shape

        # Generate scale and shift
        scale = self.scale_net(emotion_proj).view(B, N, D)  # (B, N, D)
        shift = self.shift_net(emotion_proj).view(B, N, D)  # (B, N, D)

        # FiLM modulation: (1 + scale) for residual-style
        return queries * (1 + scale) + shift


# =============================================================================
# V0.4: Emotion-Specific Attention Bias
# =============================================================================

class EmotionAttentionBias(nn.Module):
    """
    Learn emotion-specific attention patterns over text (V0.4).

    Different emotions naturally attend to different parts of text:
    - Angry emotions might focus on exclamations and strong words
    - Sad emotions might focus on slower-paced segments
    - Happy emotions might focus on upbeat markers and rhythm

    The bias is added to cross-attention scores before softmax.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_emotions: int = 16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions

        # Each emotion has a learned bias vector
        self.emotion_bias = nn.Embedding(num_emotions, hidden_size)

        # Project context for bias computation
        self.context_proj = nn.Linear(hidden_size, hidden_size)

        # Scale factor
        self.scale = hidden_size ** -0.5

        # Emotion name to index mapping
        self.emotion_to_idx: Dict[str, int] = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'excited': 4, 'calm': 5, 'surprised': 6, 'fearful': 7,
            'disgusted': 8, 'whisper': 9, 'shout': 10,
            'sarcastic': 11, 'bored': 12, 'affectionate': 13,
            'contemptuous': 14, 'awed': 15,
        }

        self._init_weights()

    def _init_weights(self):
        """Initialize with small values for gradual learning."""
        nn.init.normal_(self.emotion_bias.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.context_proj.weight, gain=0.5)
        nn.init.zeros_(self.context_proj.bias)

    def get_emotion_idx(self, emotion_name: str) -> int:
        """Get emotion index from name."""
        return self.emotion_to_idx.get(emotion_name.lower(), 0)

    def forward(
        self,
        emotion_idx: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute emotion-specific attention bias.

        Args:
            emotion_idx: (B,) emotion class indices
            context: (B, L, D) text context

        Returns:
            bias: (B, 1, 1, L) attention bias to add to cross-attention scores
                  Shape is compatible with (B, num_heads, num_queries, L)
        """
        # Get emotion-specific bias vector (B, D)
        emotion_bias = self.emotion_bias(emotion_idx)

        # Project context (B, L, D)
        context_proj = self.context_proj(context)

        # Compute per-position bias via dot product
        # (B, D) @ (B, L, D).T -> (B, L)
        bias = torch.einsum('bd,bld->bl', emotion_bias, context_proj)

        # Scale and reshape for attention (B, 1, 1, L) for broadcasting
        # across num_heads and num_queries
        return (bias * self.scale).unsqueeze(1).unsqueeze(2)


# =============================================================================
# V0.4: Enhanced Emotion Cross-Attention
# =============================================================================

class EmotionCrossAttention(nn.Module):
    """
    Cross-attention module for emotion conditioning (V0.4).

    Emotion embeddings attend to text/speech context to create
    context-aware emotion representations.

    V0.4 Enhancements:
    - GatedEmotionProjection instead of single Linear
    - FiLM-style query fusion for stronger coupling
    - Emotion-specific attention biases
    - 8 query tokens (expanded from 4) with semantic grouping

    Args:
        hidden_size: Model hidden dimension (default: 1024)
        emotion_dim: Input emotion embedding dimension (default: 64)
        num_heads: Number of attention heads (default: 8)
        num_query_tokens: Number of learnable query tokens (default: 8 for V0.4)
        dropout: Dropout rate (default: 0.1)
        use_flash_attention: Whether to use flash attention when available
        use_gated_projection: Whether to use gated projection (V0.4)
        use_film_fusion: Whether to use FiLM fusion (V0.4)
        use_attention_bias: Whether to use emotion attention bias (V0.4)

    Example:
        >>> cross_attn = EmotionCrossAttention(hidden_size=1024, emotion_dim=64)
        >>> emotion_embed = torch.randn(2, 64)  # (B, emotion_dim)
        >>> text_context = torch.randn(2, 50, 1024)  # (B, seq_len, hidden)
        >>> output = cross_attn(emotion_embed, context=text_context)  # (B, 8, 1024)
    """

    # Semantic grouping for 8 query tokens
    QUERY_SEMANTICS = {
        0: 'pitch_mean',      # Pitch baseline
        1: 'pitch_contour',   # Pitch variation
        2: 'energy_mean',     # Energy/loudness
        3: 'energy_dynamics', # Energy variation
        4: 'speaking_rate',   # Tempo
        5: 'rhythm_pattern',  # Rhythm/pauses
        6: 'voice_quality',   # Breathiness/tension
        7: 'micro_prosody',   # Fine-grained timing
    }

    def __init__(
        self,
        hidden_size: int = 1024,
        emotion_dim: int = 64,
        num_heads: int = 8,
        num_query_tokens: int = 8,  # V0.4: expanded from 4
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        use_gated_projection: bool = True,   # V0.4
        use_film_fusion: bool = True,        # V0.4
        use_attention_bias: bool = True,     # V0.4
        num_emotions: int = 16,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.emotion_dim = emotion_dim
        self.num_heads = num_heads
        self.num_query_tokens = num_query_tokens
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        self.use_gated_projection = use_gated_projection
        self.use_film_fusion = use_film_fusion
        self.use_attention_bias = use_attention_bias

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # V0.4: Gated projection or simple linear
        if use_gated_projection:
            self.emotion_proj = GatedEmotionProjection(emotion_dim, hidden_size)
        else:
            self.emotion_proj = nn.Linear(emotion_dim, hidden_size)

        # Learnable query tokens for capturing different emotion aspects
        # V0.4: 8 tokens with semantic grouping
        self.query_tokens = nn.Parameter(
            torch.empty(1, num_query_tokens, hidden_size)
        )
        self._init_query_tokens()

        # V0.4: FiLM-style query fusion
        if use_film_fusion:
            self.query_fusion = EmotionQueryFusion(hidden_size, num_query_tokens)

        # V0.4: Emotion-specific attention bias
        if use_attention_bias:
            self.attention_bias = EmotionAttentionBias(hidden_size, num_emotions)

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

        # Store current emotion index for attention bias
        self._current_emotion_idx: Optional[torch.Tensor] = None

    def _init_query_tokens(self):
        """Initialize query tokens with semantic structure (V0.4)."""
        # Different initialization for different semantic groups
        with torch.no_grad():
            # Use scaled uniform distribution overall
            query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.num_query_tokens + self.hidden_size))

            # Initialize all tokens
            self.query_tokens.data.uniform_(-query_variance, query_variance)

            # Add slight differentiation for semantic groups (pairs)
            # Pitch queries (0-1): slightly higher values
            self.query_tokens.data[0, 0:2] *= 1.1
            # Energy queries (2-3): medium values
            self.query_tokens.data[0, 2:4] *= 1.0
            # Timing queries (4-5): slightly lower values
            self.query_tokens.data[0, 4:6] *= 0.9
            # Voice quality queries (6-7): medium values
            self.query_tokens.data[0, 6:8] *= 1.0

    def set_emotion_idx(self, emotion_idx: torch.Tensor):
        """Set current emotion index for attention bias."""
        self._current_emotion_idx = emotion_idx

    def set_emotion_name(self, emotion_name: str, batch_size: int, device: torch.device):
        """Set emotion by name for attention bias."""
        if self.use_attention_bias:
            idx = self.attention_bias.get_emotion_idx(emotion_name)
            self._current_emotion_idx = torch.tensor([idx] * batch_size, device=device)

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
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor (B, H, Lq, head_dim)
            k: Key tensor (B, H, Lk, head_dim)
            v: Value tensor (B, H, Lk, head_dim)
            mask: Optional attention mask
            attn_bias: Optional attention bias (B, 1, 1, Lk) for V0.4

        Returns:
            Attention output (B, H, Lq, head_dim)
        """
        # Check if we can use flash attention (no custom bias)
        if (self.use_flash_attention and
            hasattr(F, 'scaled_dot_product_attention') and
            attn_bias is None):
            try:
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    scale=self.scale,
                )
            except RuntimeError:
                pass

        # Manual attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # V0.4: Add emotion-specific attention bias
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

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
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention: queries attend to context.

        Args:
            queries: Query tensor (B, Lq, D)
            context: Context tensor (B, Lc, D)
            mask: Optional attention mask
            attn_bias: Optional emotion attention bias (V0.4)

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

        # Attention with optional bias
        attn_output = self._attention(q, k, v, mask, attn_bias)
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
        emotion_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass for emotion cross-attention.

        Args:
            emotion_embed: Emotion embedding tensor (B, emotion_dim) or (B, 1, emotion_dim)
            context: Optional context tensor (B, L, hidden_size) from text embeddings.
                    If None, only self-attention is applied.
            emotion_name: Optional emotion name for attention bias (V0.4)

        Returns:
            Emotion conditioning tensor (B, num_query_tokens, hidden_size)
        """
        # Handle input shape
        if emotion_embed.dim() == 3:
            # (B, 1, emotion_dim) -> (B, emotion_dim)
            emotion_embed = emotion_embed.squeeze(1)

        batch_size = emotion_embed.shape[0]
        device = emotion_embed.device

        # Project emotion to hidden size (V0.4: gated projection)
        emotion_proj = self.emotion_proj(emotion_embed)  # (B, hidden_size)
        if isinstance(self.emotion_proj, nn.Linear):
            emotion_proj = self.norm_pre(emotion_proj)
        emotion_proj_expanded = emotion_proj.unsqueeze(1)  # (B, 1, hidden_size)

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # (B, num_query, hidden_size)

        # V0.4: FiLM-style query fusion (instead of simple addition)
        if self.use_film_fusion:
            queries = self.query_fusion(queries, emotion_proj)
        else:
            # Fallback to simple addition for backward compatibility
            queries = queries + emotion_proj_expanded

        # V0.4: Compute emotion-specific attention bias
        attn_bias = None
        if self.use_attention_bias and context is not None:
            # Set emotion index if provided by name
            if emotion_name is not None:
                self.set_emotion_name(emotion_name, batch_size, device)

            # Use stored emotion index
            if self._current_emotion_idx is not None:
                attn_bias = self.attention_bias(self._current_emotion_idx, context)

        # Cross-attention to context if provided
        if context is not None and context.shape[1] > 0:
            queries = self._cross_attention(queries, context, attn_bias=attn_bias)

        # Self-attention for refinement
        queries = self._self_attention(queries)

        # Feed-forward network
        queries = queries + self.ffn(self.norm_ffn(queries))

        return queries  # (B, num_query_tokens, hidden_size)


# =============================================================================
# Factory Functions
# =============================================================================

def create_emotion_cross_attention(
    hidden_size: int = 1024,
    emotion_dim: int = 64,
    num_heads: int = 8,
    num_query_tokens: int = 8,
    use_v04_features: bool = True,
) -> EmotionCrossAttention:
    """
    Factory function to create emotion cross-attention module.

    Args:
        hidden_size: Model hidden dimension
        emotion_dim: Input emotion embedding dimension
        num_heads: Number of attention heads
        num_query_tokens: Number of query tokens (8 for V0.4)
        use_v04_features: Whether to enable V0.4 enhancements

    Returns:
        EmotionCrossAttention module
    """
    return EmotionCrossAttention(
        hidden_size=hidden_size,
        emotion_dim=emotion_dim,
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
        use_gated_projection=use_v04_features,
        use_film_fusion=use_v04_features,
        use_attention_bias=use_v04_features,
    )


def create_emotion_cross_attention_v03(
    hidden_size: int = 1024,
    emotion_dim: int = 64,
    num_heads: int = 8,
    num_query_tokens: int = 4,
) -> EmotionCrossAttention:
    """
    Factory function to create V0.3 compatible emotion cross-attention.

    For backward compatibility with V0.3 checkpoints.
    """
    return EmotionCrossAttention(
        hidden_size=hidden_size,
        emotion_dim=emotion_dim,
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
        use_gated_projection=False,
        use_film_fusion=False,
        use_attention_bias=False,
    )
