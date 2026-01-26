# Emotion Architecture V0.4 - Improvement Specification

**Version**: 0.4
**Date**: 2026-01-25
**Status**: Proposed
**Previous Version**: V0.3 (64D embeddings + cross-attention + LoRA)

---

## 1. Executive Summary

This document specifies architectural improvements to the Chatterbox emotion TTS system to increase SER (Speech Emotion Recognition) accuracy from the current ~66% (emotion2vec) to **85%+**.

### Current Performance (V0.3)

| Model | CREMA-D | RAVDESS | Average |
|-------|---------|---------|---------|
| emotion2vec | 70.4% | 62.1% | 66.3% |
| dpngtm | 29.6% | 34.5% | 32.1% |
| ehcalabres | 18.5% | 13.8% | 16.2% |

### Target Performance (V0.4)

| Model | Target | Improvement |
|-------|--------|-------------|
| emotion2vec | 85%+ | +19% |
| dpngtm | 70%+ | +38% |
| Ensemble | 80%+ | +42% |

---

## 2. Architecture Overview

### 2.1 Current Architecture (V0.3)

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMOTION CONDITIONING FLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Emotion Name + Intensity                                        │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │ EmotionEmbeddings │  64D = VAD(3) + Prosodic(13) + Fine(48)  │
│  │   + Intensity     │                                           │
│  │   Transform       │                                           │
│  └────────┬─────────┘                                           │
│           │ 64D                                                  │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  emotion_proj    │  Linear(64 → 1024)  ← BOTTLENECK          │
│  └────────┬─────────┘                                           │
│           │ 1024D                                                │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Query Tokens    │  4 learnable tokens (1024D each)          │
│  │  + emotion_proj  │  queries = queries + emotion  ← WEAK      │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐     ┌─────────────┐                       │
│  │ Cross-Attention  │◄────│ Text Context │                      │
│  └────────┬─────────┘     └─────────────┘                       │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Self-Attention   │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │      FFN         │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│     Output: (B, 4, 1024)                                        │
│           │                                                      │
│           ▼                                                      │
│  Concatenate with [speaker_emb | prompt_emb]                    │
│           │                                                      │
│           ▼                                                      │
│     T3 Transformer → Speech Tokens → Vocoder                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Identified Bottlenecks

| Component | Issue | Impact |
|-----------|-------|--------|
| `emotion_proj` | Single linear layer loses fine-grained info | High |
| Query injection | Additive only (`q + e`) is weak coupling | High |
| Query tokens | Only 4 tokens limits prosodic control | Medium |
| Fine-grained dims | 48D initialized to 0, never utilized | Medium |
| Cross-attention | No emotion-specific text attention | Medium |
| Intensity | Not calibrated per emotion | High |

---

## 3. V0.4 Architecture Improvements

### 3.1 Gated Emotion Projection

**Problem**: Single `Linear(64 → 1024)` loses fine-grained emotion information.

**Solution**: Multi-layer gated projection with residual connection.

```
┌─────────────────────────────────────────────────────────────┐
│              GATED EMOTION PROJECTION (V0.4)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: emotion_embed (B, 64)                               │
│              │                                              │
│              ├───────────────────┐                          │
│              │                   │                          │
│              ▼                   ▼                          │
│  ┌─────────────────┐   ┌─────────────────┐                 │
│  │   Gate Branch   │   │   Value Branch  │                 │
│  │ Linear(64→1024) │   │ Linear(64→512)  │                 │
│  │    Sigmoid()    │   │    GELU()       │                 │
│  └────────┬────────┘   │ Linear(512→1024)│                 │
│           │            └────────┬────────┘                 │
│           │                     │                          │
│           │    gate             │    value                 │
│           └──────────┬──────────┘                          │
│                      │                                      │
│                      ▼                                      │
│               gate * value                                  │
│                      │                                      │
│                      ▼                                      │
│              LayerNorm(1024)                                │
│                      │                                      │
│                      ▼                                      │
│  Output: emotion_proj (B, 1024)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
class GatedEmotionProjection(nn.Module):
    """
    Multi-layer gated projection for emotion embeddings.

    The gate branch learns which dimensions to emphasize,
    while the value branch transforms the representation.
    """

    def __init__(self, emotion_dim: int = 64, hidden_size: int = 1024):
        super().__init__()

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
```

**Expected Impact**: +7% accuracy

---

### 3.2 FiLM-Style Query Fusion

**Problem**: Additive injection `queries = queries + emotion_proj` is too weak.

**Solution**: Feature-wise Linear Modulation (FiLM) with learned scale and shift.

```
┌─────────────────────────────────────────────────────────────┐
│              FiLM QUERY FUSION (V0.4)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  emotion_proj (B, 1024)      query_tokens (B, N, 1024)     │
│         │                           │                       │
│         │                           │                       │
│         ├─────────────┬─────────────┤                       │
│         │             │             │                       │
│         ▼             ▼             │                       │
│  ┌────────────┐ ┌────────────┐     │                       │
│  │ Scale Net  │ │ Shift Net  │     │                       │
│  │Linear→     │ │Linear→     │     │                       │
│  │(N*1024)    │ │(N*1024)    │     │                       │
│  └─────┬──────┘ └─────┬──────┘     │                       │
│        │              │             │                       │
│        ▼              ▼             │                       │
│   scale (B,N,D)  shift (B,N,D)     │                       │
│        │              │             │                       │
│        └──────┬───────┘             │                       │
│               │                     │                       │
│               ▼                     ▼                       │
│    queries * (1 + scale) + shift ← queries                 │
│               │                                             │
│               ▼                                             │
│      fused_queries (B, N, 1024)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
class EmotionQueryFusion(nn.Module):
    """
    FiLM-style modulation: queries = scale * queries + shift

    This provides stronger emotion-query coupling than simple addition.
    The scale allows the emotion to amplify/dampen query features,
    while shift allows additive modification.
    """

    def __init__(self, hidden_size: int = 1024, num_queries: int = 4):
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

        # Initialize to near-identity (scale≈0, shift≈0)
        self._init_weights()

    def _init_weights(self):
        """Initialize to near-identity transformation."""
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
```

**Expected Impact**: +6% accuracy

---

### 3.3 Expanded Query Tokens (4 → 8)

**Problem**: 4 query tokens have limited capacity for nuanced prosodic control.

**Solution**: Expand to 8 tokens with semantic grouping.

```
┌─────────────────────────────────────────────────────────────┐
│              EXPANDED QUERY TOKENS (V0.4)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query Token Semantic Assignment:                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Query 0-1: PITCH CONTROL                            │   │
│  │   • Q0: Pitch mean/baseline                         │   │
│  │   • Q1: Pitch contour/variation                     │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ Query 2-3: ENERGY CONTROL                           │   │
│  │   • Q2: Energy mean/loudness                        │   │
│  │   • Q3: Energy dynamics/variation                   │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ Query 4-5: TIMING CONTROL                           │   │
│  │   • Q4: Speaking rate                               │   │
│  │   • Q5: Rhythm/pause patterns                       │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ Query 6-7: VOICE QUALITY CONTROL                    │   │
│  │   • Q6: Breathiness/tension                         │   │
│  │   • Q7: Tremolo/vibrato                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
# In EmotionCrossAttention.__init__():

# Change from:
# self.num_query_tokens = 4

# To:
self.num_query_tokens = 8

# Semantic initialization
def _init_query_tokens(self):
    """Initialize query tokens with semantic structure."""
    with torch.no_grad():
        # Use different initialization scales for different groups
        # Pitch queries (0-1): higher frequency sensitivity
        self.query_tokens.data[0, 0:2] = torch.randn(2, self.hidden_size) * 0.02

        # Energy queries (2-3): amplitude patterns
        self.query_tokens.data[0, 2:4] = torch.randn(2, self.hidden_size) * 0.02

        # Timing queries (4-5): temporal patterns
        self.query_tokens.data[0, 4:6] = torch.randn(2, self.hidden_size) * 0.02

        # Voice quality queries (6-7): spectral characteristics
        self.query_tokens.data[0, 6:8] = torch.randn(2, self.hidden_size) * 0.02
```

**Expected Impact**: +4% accuracy

---

### 3.4 Emotion-Specific Attention Bias

**Problem**: Cross-attention treats all text positions equally regardless of emotion.

**Solution**: Learn emotion-specific attention biases.

```
┌─────────────────────────────────────────────────────────────┐
│          EMOTION ATTENTION BIAS (V0.4)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Different emotions should attend to text differently:      │
│                                                             │
│  • ANGRY: Focus on exclamations, strong words              │
│  • SAD: Focus on pauses, elongated words                   │
│  • HAPPY: Focus on positive markers, rhythm                │
│  • SURPRISED: Focus on question marks, unexpected words    │
│                                                             │
│  ┌──────────────┐     ┌─────────────────────────┐          │
│  │ emotion_idx  │────▶│ Emotion Bias Embedding  │          │
│  └──────────────┘     │    (16 × 1024)          │          │
│                       └───────────┬─────────────┘          │
│                                   │                         │
│                                   ▼                         │
│                       emotion_bias (B, 1024)               │
│                                   │                         │
│                                   ▼                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Cross-Attention Scores                              │    │
│  │                                                     │    │
│  │ attn = Q @ K^T / sqrt(d) + emotion_text_bias       │    │
│  │                            ↑                        │    │
│  │                     dot(emotion_bias, context)      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
class EmotionAttentionBias(nn.Module):
    """
    Learn emotion-specific attention patterns over text.

    Different emotions naturally attend to different parts of text:
    - Angry emotions might focus on exclamations
    - Sad emotions might focus on slower-paced segments
    - Happy emotions might focus on upbeat markers
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_emotions: int = 16
    ):
        super().__init__()

        # Each emotion has a learned bias vector
        self.emotion_bias = nn.Embedding(num_emotions, hidden_size)

        # Project context for bias computation
        self.context_proj = nn.Linear(hidden_size, hidden_size)

        # Scale factor
        self.scale = hidden_size ** -0.5

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
            bias: (B, 1, L) attention bias to add to cross-attention scores
        """
        # Get emotion-specific bias vector (B, D)
        emotion_bias = self.emotion_bias(emotion_idx)

        # Project context (B, L, D)
        context_proj = self.context_proj(context)

        # Compute per-position bias via dot product
        # (B, D) @ (B, D, L) -> (B, L)
        bias = torch.einsum('bd,bld->bl', emotion_bias, context_proj)

        # Scale and reshape for attention (B, 1, L) for broadcasting
        return (bias * self.scale).unsqueeze(1)
```

**Expected Impact**: +6% accuracy

---

### 3.5 Fine-Grained Dimension Initialization

**Problem**: 48 fine-grained dimensions initialized to 0.0, never utilized.

**Solution**: Initialize with emotion-specific patterns.

```
┌─────────────────────────────────────────────────────────────┐
│          FINE-GRAINED INITIALIZATION (V0.4)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  64D Emotion Embedding Structure:                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Dims 0-2: VAD (Valence, Arousal, Dominance)         │   │
│  │   → Psychological emotion model                      │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ Dims 3-15: Prosodic Features (13D)                  │   │
│  │   → pitch_mean, pitch_range, pitch_contour          │   │
│  │   → energy_mean, energy_range                       │   │
│  │   → speaking_rate, rhythm, voice_quality            │   │
│  │   → breathiness, tension, nasality                  │   │
│  │   → jitter, shimmer                                 │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ Dims 16-63: Fine-Grained Features (48D)             │   │
│  │                                                      │   │
│  │   ┌────────────────────────────────────────────┐    │   │
│  │   │ 16-23: Formant modifications (8D)          │    │   │
│  │   │   → F1/F2 shifts for vowel coloring        │    │   │
│  │   ├────────────────────────────────────────────┤    │   │
│  │   │ 24-31: Spectral tilt (8D)                  │    │   │
│  │   │   → Brightness/warmth characteristics      │    │   │
│  │   ├────────────────────────────────────────────┤    │   │
│  │   │ 32-39: Harmonics-to-noise (8D)             │    │   │
│  │   │   → Voice clarity/breathiness details      │    │   │
│  │   ├────────────────────────────────────────────┤    │   │
│  │   │ 40-47: Micro-prosody (8D)                  │    │   │
│  │   │   → Sub-phoneme timing variations          │    │   │
│  │   ├────────────────────────────────────────────┤    │   │
│  │   │ 48-55: Articulation (8D)                   │    │   │
│  │   │   → Consonant strength, vowel reduction    │    │   │
│  │   ├────────────────────────────────────────────┤    │   │
│  │   │ 56-63: Reserved/learnable (8D)             │    │   │
│  │   │   → Additional learned features            │    │   │
│  │   └────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
def _create_64d_embedding_v04(
    vad: List[float],
    prosodic: List[float],
    emotion_name: str = "neutral"
) -> List[float]:
    """
    Create 64D embedding with emotion-specific fine-grained initialization.

    Args:
        vad: 3D valence-arousal-dominance values
        prosodic: 13D prosodic feature values
        emotion_name: Name for emotion-specific initialization

    Returns:
        64D embedding list
    """
    # Emotion-specific fine-grained patterns
    FINE_GRAINED_PATTERNS = {
        'angry': {
            'formant': [0.1, 0.15, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1],      # Tense formants
            'spectral': [0.2, 0.15, 0.1, 0.15, 0.2, 0.1, 0.15, 0.1],     # Bright
            'hnr': [-0.1, -0.15, -0.1, -0.1, -0.05, -0.1, -0.1, -0.05],  # More noise
            'micro': [0.1, 0.05, 0.1, 0.15, 0.1, 0.05, 0.1, 0.1],        # Sharp timing
            'articulation': [0.15, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1], # Strong consonants
            'reserved': [0.05] * 8,
        },
        'sad': {
            'formant': [-0.1, -0.05, -0.1, -0.1, -0.05, -0.1, -0.05, -0.1],  # Lowered
            'spectral': [-0.15, -0.1, -0.15, -0.1, -0.15, -0.1, -0.1, -0.15], # Dark
            'hnr': [0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1],            # Clearer
            'micro': [-0.1, -0.15, -0.1, -0.1, -0.15, -0.1, -0.1, -0.15],   # Slow
            'articulation': [-0.1, -0.05, -0.1, -0.1, -0.05, -0.1, -0.05, -0.1],
            'reserved': [0.0] * 8,
        },
        'happy': {
            'formant': [0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05],
            'spectral': [0.15, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1],
            'hnr': [0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1],
            'micro': [0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.05],
            'articulation': [0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1],
            'reserved': [0.05] * 8,
        },
        'fearful': {
            'formant': [0.05, 0.1, 0.15, 0.1, 0.05, 0.1, 0.15, 0.1],
            'spectral': [0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.15],
            'hnr': [-0.15, -0.1, -0.15, -0.1, -0.15, -0.1, -0.1, -0.15],  # Tremor
            'micro': [0.15, 0.1, 0.15, 0.2, 0.15, 0.1, 0.15, 0.2],        # Irregular
            'articulation': [-0.05, 0.0, -0.05, 0.0, -0.05, 0.0, -0.05, 0.0],
            'reserved': [0.1] * 8,
        },
        'disgusted': {
            'formant': [0.0, 0.1, 0.0, 0.15, 0.0, 0.1, 0.15, 0.0],       # Nasal shift
            'spectral': [-0.05, 0.0, -0.05, 0.0, -0.05, 0.0, 0.0, -0.05],
            'hnr': [-0.05, -0.1, -0.05, -0.1, -0.05, -0.1, -0.05, -0.1],
            'micro': [-0.05, 0.0, -0.05, 0.0, -0.05, 0.0, -0.05, 0.0],
            'articulation': [0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.15], # Tight articulation
            'reserved': [0.0] * 8,
        },
        'neutral': {
            'formant': [0.0] * 8,
            'spectral': [0.0] * 8,
            'hnr': [0.0] * 8,
            'micro': [0.0] * 8,
            'articulation': [0.0] * 8,
            'reserved': [0.0] * 8,
        },
    }

    # Get pattern for emotion (default to neutral)
    pattern = FINE_GRAINED_PATTERNS.get(emotion_name.lower(), FINE_GRAINED_PATTERNS['neutral'])

    # Build fine-grained features
    fine_grained = (
        pattern['formant'] +
        pattern['spectral'] +
        pattern['hnr'] +
        pattern['micro'] +
        pattern['articulation'] +
        pattern['reserved']
    )

    return vad + prosodic + fine_grained
```

**Expected Impact**: +4% accuracy

---

### 3.6 Calibrated Emotion Intensities

**Problem**: Default intensity 1.0 doesn't produce recognizable emotions for all types.

**Solution**: Pre-calibrated intensity multipliers per emotion.

```
┌─────────────────────────────────────────────────────────────┐
│          CALIBRATED INTENSITIES (V0.4)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Per-emotion calibration based on SER analysis:             │
│                                                             │
│  ┌────────────┬────────────┬──────────────────────────┐    │
│  │  Emotion   │ Multiplier │ Rationale                │    │
│  ├────────────┼────────────┼──────────────────────────┤    │
│  │ angry      │    1.5     │ Needs energy boost       │    │
│  │ disgusted  │    1.4     │ Needs clearer markers    │    │
│  │ excited    │    1.3     │ Needs energy boost       │    │
│  │ happy      │    1.2     │ Slight pitch boost       │    │
│  │ fearful    │    1.1     │ Minor enhancement        │    │
│  │ surprised  │    1.0     │ Works well               │    │
│  │ sad        │    1.0     │ Works well               │    │
│  │ neutral    │    1.0     │ Baseline                 │    │
│  │ calm       │    0.8     │ Reduce to avoid neutral  │    │
│  └────────────┴────────────┴──────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
# File: emotion_intensity_calibration.py

CALIBRATED_INTENSITIES = {
    # High-energy emotions need boost
    'angry': 1.5,
    'disgusted': 1.4,
    'excited': 1.3,
    'happy': 1.2,
    'fearful': 1.1,
    'shout': 1.3,

    # Well-performing emotions
    'surprised': 1.0,
    'sad': 1.0,
    'neutral': 1.0,

    # Low-arousal emotions need reduction to differentiate
    'calm': 0.8,
    'whisper': 0.9,
    'bored': 0.9,

    # New emotions (conservative)
    'sarcastic': 1.1,
    'affectionate': 1.0,
    'contemptuous': 1.2,
    'awed': 1.1,
}


def get_calibrated_intensity(
    emotion: str,
    user_intensity: float = 1.0,
    use_calibration: bool = True
) -> float:
    """
    Apply calibration to user-specified intensity.

    Args:
        emotion: Emotion name
        user_intensity: User-specified intensity (0.0 - 2.0)
        use_calibration: Whether to apply calibration

    Returns:
        Calibrated intensity value
    """
    if not use_calibration:
        return user_intensity

    calibration = CALIBRATED_INTENSITIES.get(emotion.lower(), 1.0)
    calibrated = user_intensity * calibration

    # Clamp to reasonable range
    return max(0.1, min(2.5, calibrated))
```

**Expected Impact**: +7% accuracy (immediate)

---

## 4. Training Improvements

### 4.1 Dynamic Emotion Loss Weighting

**Problem**: All emotions trained equally despite different difficulty levels.

**Solution**: Increase loss weight for poorly-performing emotions.

```python
class DynamicEmotionLossWeight:
    """
    Dynamically adjust loss weights based on per-emotion accuracy.

    Emotions with lower SER accuracy get higher loss weights
    to focus training on difficult cases.
    """

    def __init__(self, emotions: List[str], min_weight: float = 0.5, max_weight: float = 3.0):
        self.emotions = emotions
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.accuracy_ema = {e: 0.5 for e in emotions}  # Start at 50%
        self.ema_decay = 0.9

    def update(self, emotion: str, accuracy: float):
        """Update EMA of accuracy for emotion."""
        if emotion in self.accuracy_ema:
            self.accuracy_ema[emotion] = (
                self.ema_decay * self.accuracy_ema[emotion] +
                (1 - self.ema_decay) * accuracy
            )

    def get_weight(self, emotion: str) -> float:
        """Get loss weight for emotion (inverse of accuracy)."""
        acc = self.accuracy_ema.get(emotion, 0.5)

        # Inverse relationship: lower accuracy = higher weight
        weight = 1.0 / max(0.1, acc)

        # Clamp to range
        return max(self.min_weight, min(self.max_weight, weight))
```

### 4.2 Hard Negative Mining

**Problem**: Commonly confused emotion pairs (angry/neutral, disgusted/surprised).

**Solution**: Explicitly train on hard negative pairs.

```python
HARD_NEGATIVE_PAIRS = [
    ('angry', 'neutral'),      # Default intensity confusion
    ('disgusted', 'surprised'), # Prosodic similarity
    ('calm', 'neutral'),       # Low arousal confusion
    ('calm', 'angry'),         # Unexpected confusion
    ('happy', 'angry'),        # Low intensity confusion
]


class HardNegativeSampler:
    """
    Sample batches that include hard negative pairs.

    Ensures each batch has examples of commonly confused emotions
    to provide strong gradient signals for discrimination.
    """

    def __init__(self, dataset, hard_pairs: List[Tuple[str, str]]):
        self.dataset = dataset
        self.hard_pairs = hard_pairs
        self.emotion_indices = self._build_emotion_indices()

    def _build_emotion_indices(self):
        """Build mapping from emotion to dataset indices."""
        indices = {}
        for i, item in enumerate(self.dataset):
            emotion = item['emotion']
            if emotion not in indices:
                indices[emotion] = []
            indices[emotion].append(i)
        return indices

    def sample_batch(self, batch_size: int) -> List[int]:
        """
        Sample batch ensuring hard negative pairs are included.
        """
        indices = []

        # Include at least one hard pair
        if self.hard_pairs and batch_size >= 4:
            pair = random.choice(self.hard_pairs)
            e1, e2 = pair

            if e1 in self.emotion_indices and e2 in self.emotion_indices:
                indices.append(random.choice(self.emotion_indices[e1]))
                indices.append(random.choice(self.emotion_indices[e2]))

        # Fill rest randomly
        all_indices = list(range(len(self.dataset)))
        remaining = batch_size - len(indices)
        indices.extend(random.sample(all_indices, remaining))

        return indices
```

### 4.3 Curriculum Learning

**Problem**: Training all emotions simultaneously can cause interference.

**Solution**: Progressively introduce emotions during training.

```python
class EmotionCurriculum:
    """
    Progressive emotion introduction during training.

    Start with easily recognizable emotions, gradually add difficult ones.
    """

    # Phases ordered by difficulty (easiest first)
    PHASES = [
        # Phase 1 (Epochs 0-4): High accuracy emotions
        {'emotions': ['neutral', 'happy', 'sad'], 'name': 'core'},

        # Phase 2 (Epochs 5-9): Standard emotions
        {'emotions': ['angry', 'fearful', 'surprised'], 'name': 'standard'},

        # Phase 3 (Epochs 10-14): Subtle emotions
        {'emotions': ['disgusted', 'calm', 'excited'], 'name': 'subtle'},

        # Phase 4 (Epochs 15+): All emotions
        {'emotions': ['whisper', 'shout', 'sarcastic', 'bored',
                     'affectionate', 'contemptuous', 'awed'], 'name': 'extended'},
    ]

    def __init__(self, epochs_per_phase: int = 5):
        self.epochs_per_phase = epochs_per_phase

    def get_active_emotions(self, epoch: int) -> List[str]:
        """Get emotions active at current epoch."""
        phase_idx = min(epoch // self.epochs_per_phase, len(self.PHASES) - 1)

        active = []
        for i in range(phase_idx + 1):
            active.extend(self.PHASES[i]['emotions'])

        return active

    def get_phase_name(self, epoch: int) -> str:
        """Get current phase name."""
        phase_idx = min(epoch // self.epochs_per_phase, len(self.PHASES) - 1)
        return self.PHASES[phase_idx]['name']
```

---

## 5. Updated Architecture Diagram (V0.4)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EMOTION CONDITIONING FLOW (V0.4)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Emotion Name + Intensity                                            │
│         │                                                            │
│         ▼                                                            │
│  ┌────────────────────────┐                                         │
│  │   Intensity Calibration │  ← NEW: Per-emotion multiplier          │
│  └───────────┬────────────┘                                         │
│              │                                                       │
│              ▼                                                       │
│  ┌────────────────────────┐                                         │
│  │  EmotionEmbeddings     │  64D with initialized fine-grained      │
│  │  + IntensityTransform  │  ← ENHANCED: Emotion-specific init      │
│  └───────────┬────────────┘                                         │
│              │ 64D                                                   │
│              ▼                                                       │
│  ┌────────────────────────┐                                         │
│  │ GatedEmotionProjection │  ← NEW: gate * value + LayerNorm        │
│  │   (replaces Linear)    │                                         │
│  └───────────┬────────────┘                                         │
│              │ 1024D                                                 │
│              ▼                                                       │
│  ┌────────────────────────┐                                         │
│  │   8 Query Tokens       │  ← EXPANDED: 4 → 8 semantic tokens      │
│  └───────────┬────────────┘                                         │
│              │                                                       │
│              ▼                                                       │
│  ┌────────────────────────┐                                         │
│  │  EmotionQueryFusion    │  ← NEW: FiLM modulation                 │
│  │  (scale * q + shift)   │                                         │
│  └───────────┬────────────┘                                         │
│              │ (B, 8, 1024)                                          │
│              ▼                                                       │
│  ┌────────────────────────┐     ┌─────────────────────────┐         │
│  │    Cross-Attention     │◄────│     Text Context        │         │
│  │  + EmotionAttnBias     │     └─────────────────────────┘         │
│  │       ← NEW            │                                          │
│  └───────────┬────────────┘                                         │
│              │                                                       │
│              ▼                                                       │
│  ┌────────────────────────┐                                         │
│  │    Self-Attention      │                                         │
│  └───────────┬────────────┘                                         │
│              │                                                       │
│              ▼                                                       │
│  ┌────────────────────────┐                                         │
│  │         FFN            │                                         │
│  └───────────┬────────────┘                                         │
│              │                                                       │
│              ▼                                                       │
│       Output: (B, 8, 1024)                                          │
│              │                                                       │
│              ▼                                                       │
│  Concatenate with [speaker_emb | prompt_emb]                        │
│              │                                                       │
│              ▼                                                       │
│       T3 Transformer → Speech Tokens → Vocoder                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Plan

### Phase 1: Quick Wins (1-2 days)

| Task | File | Priority |
|------|------|----------|
| Add intensity calibration | `emotion_intensity_calibration.py` | P0 |
| Implement GatedEmotionProjection | `emotion_cross_attention.py` | P0 |
| Implement EmotionQueryFusion | `emotion_cross_attention.py` | P0 |

### Phase 2: Architecture (3-5 days)

| Task | File | Priority |
|------|------|----------|
| Expand to 8 query tokens | `emotion_cross_attention.py` | P1 |
| Add EmotionAttentionBias | `emotion_cross_attention.py` | P1 |
| Initialize fine-grained dims | `emotion_embeddings.py` | P1 |

### Phase 3: Training (3-5 days)

| Task | File | Priority |
|------|------|----------|
| Dynamic loss weighting | `train_emotion_lora.py` | P1 |
| Hard negative sampling | `train_emotion_lora.py` | P1 |
| Curriculum learning | `train_emotion_lora.py` | P2 |

### Phase 4: Validation (2-3 days)

| Task | File | Priority |
|------|------|----------|
| Benchmark V0.4 architecture | `benchmark_v03.py` | P1 |
| Compare with V0.3 baseline | `benchmark_llm_emotions.py` | P1 |
| Document results | `BENCHMARK_RESULT.md` | P1 |

---

## 7. Expected Results

### Accuracy Improvements

| Component | Improvement | Cumulative |
|-----------|-------------|------------|
| Baseline (V0.3) | - | 66% |
| + Calibrated intensities | +7% | 73% |
| + Gated projection | +5% | 78% |
| + FiLM fusion | +4% | 82% |
| + 8 query tokens | +2% | 84% |
| + Attention bias | +3% | 87% |
| + Fine-grained init | +2% | **89%** |

### Per-Emotion Targets

| Emotion | V0.3 | V0.4 Target | Key Fix |
|---------|------|-------------|---------|
| Angry | 40% | 85% | Intensity calibration (1.5x) |
| Disgusted | 0% | 70% | Attention bias + fine-grained |
| Calm | 0% | 75% | Intensity reduction (0.8x) |
| Happy | 57% | 90% | FiLM fusion |
| Excited | 50% | 85% | Query expansion |
| Sad | 100% | 100% | Maintain |
| Fearful | 100% | 100% | Maintain |
| Neutral | 100% | 100% | Maintain |
| Surprised | 100% | 100% | Maintain |

---

## 8. Backward Compatibility

V0.4 maintains backward compatibility with V0.3:

1. **Checkpoint Loading**: V0.3 checkpoints can be loaded; new modules initialize to identity-like transformations.

2. **API Compatibility**: `generate()` function signature unchanged; calibration is optional.

3. **Configuration**: New parameters have sensible defaults matching V0.3 behavior.

```python
# V0.3 compatible call
model.generate(text="Hello", emotion="happy", emotion_intensity=1.0)

# V0.4 with calibration
model.generate(text="Hello", emotion="happy", emotion_intensity=1.0, use_calibration=True)
```

---

## 9. Files to Modify

| File | Changes |
|------|---------|
| `emotion_cross_attention.py` | Add GatedProjection, FiLM, AttentionBias, expand queries |
| `emotion_embeddings.py` | Add fine-grained initialization |
| `emotion_intensity_calibration.py` | NEW: Calibration module |
| `train_emotion_lora.py` | Add curriculum, dynamic weights, hard negatives |
| `mtl_tts.py` | Integrate calibration in inference |
| `benchmark_v03.py` | Add calibration flag |

---

## 10. References

1. FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)
2. Attention Is All You Need (Vaswani et al., 2017)
3. emotion2vec: Self-Supervised Pre-Training for Speech Emotion Recognition
4. Speech Emotion Recognition: A Review (Akçay & Oğuz, 2020)

---

**Document Status**: Ready for Implementation
**Next Steps**: Begin Phase 1 implementation with intensity calibration
