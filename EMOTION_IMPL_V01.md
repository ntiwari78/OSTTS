# Emotion System v0.1 Implementation

This document describes the v0.1 implementation of the improved emotion conditioning system for Chatterbox TTS.

## Overview

The emotion system has been upgraded from an 8-dimensional heuristic vector approach to a 64-dimensional learnable embedding system with cross-attention conditioning, intensity control, and emotion blending capabilities.

### Key Improvements

| Feature | Before (v0) | After (v0.1) |
|---------|-------------|--------------|
| Embedding dimension | 8D | 64D |
| Conditioning mechanism | Single-token concatenation | 4-token cross-attention |
| Intensity control | Scalar `exaggeration` (separate from emotion) | Integrated `emotion_intensity` parameter |
| Emotion blending | Not supported | Full interpolation support |
| Text awareness | None | Cross-attention to text context |

---

## Rationale for Changes

### 1. Increased Embedding Dimensionality (8D → 64D)

**Problem**: 8 dimensions were insufficient to capture the full prosodic space of emotional speech, which includes:
- Pitch patterns (mean, range, contour)
- Energy dynamics (mean, variance, burst patterns)
- Speaking rate and rhythm
- Voice quality characteristics (breathiness, tension)

**Solution**: 64-dimensional embeddings with structured initialization:
- **Dimensions 0-2**: VAD (Valence, Arousal, Dominance) from psychological emotion models
- **Dimensions 3-15**: Prosodic features (pitch, energy, rate, voice quality)
- **Dimensions 16-63**: Fine-grained learned features (initialized to zero, learned during training)

**Benefit**: Richer emotion representation while maintaining interpretability in the first 16 dimensions.

### 2. Cross-Attention Conditioning

**Problem**: The previous implementation concatenated a single 768D emotion token to the conditioning sequence. This weak signal could be "washed out" by longer text/speech sequences during transformer attention.

**Solution**: `EmotionCrossAttention` module that:
1. Projects 64D emotion to 1024D hidden space
2. Uses 4 learnable query tokens (representing pitch, energy, rate, quality aspects)
3. Applies cross-attention: emotion queries attend to text context
4. Applies self-attention for refinement
5. Outputs 4 conditioning tokens instead of 1

**Benefit**:
- 4x more conditioning tokens provide stronger signal
- Cross-attention allows emotion to modulate based on text content
- Text-aware emotion conditioning (e.g., questions vs. statements)

### 3. Integrated Intensity Control

**Problem**: The old system had separate `exaggeration` (scalar 0-1) and `emotion` (categorical) parameters that were mutually exclusive and confusing.

**Solution**: Single unified API with `emotion` + `emotion_intensity`:
- `intensity=0.0`: Returns neutral embedding
- `intensity=1.0`: Returns full emotion embedding
- `intensity>1.0`: Extrapolates beyond the emotion (exaggerated)

**Implementation**: Linear interpolation between neutral and target emotion:
```python
result = neutral_embed + intensity * (target_embed - neutral_embed)
```

**Benefit**: Intuitive control over emotion strength without separate parameters.

### 4. Emotion Blending/Interpolation

**Problem**: Real emotions are often complex mixtures (e.g., "bittersweet" = happy + sad). The old system only supported discrete categories.

**Solution**: `interpolate_emotions()` method that blends multiple emotions:
```python
embed = model.emotion_embeddings.interpolate_emotions({
    "happy": 0.4,
    "sad": 0.6
})
```

**Benefit**: More natural and nuanced emotional expressions.

### 5. Removal of Backward Compatibility

**Decision**: Remove the old `emotion_adv` scalar system entirely rather than maintaining dual APIs.

**Rationale**:
- Cleaner codebase without legacy code paths
- No confusion about which parameter to use
- The new `emotion_intensity` provides equivalent functionality
- Breaking change is acceptable for a v0.1 implementation

---

## File Changes

### 1. `src/chatterbox/models/t3/modules/emotion_embeddings.py`

**Changes**:
- Updated `EMOTION_INIT_EMBEDDINGS_64D` with 64-dimensional vectors
- Added structured initialization helper `_create_64d_embedding(vad, prosodic)`
- Added `intensity` parameter to `get_emotion_embedding()`
- Added `interpolate_emotions()` method for blending
- Added `get_emotion_index()` helper method
- Default dimension changed from 8 to 64

**Key Code**:
```python
def get_emotion_embedding(self, emotion_name: str, intensity: float = 1.0, device=None):
    """Get embedding with intensity interpolation from neutral."""
    target_embed = self.embedding(idx)
    if intensity == 1.0:
        return target_embed
    neutral_embed = self.embedding(neutral_idx)
    return neutral_embed + intensity * (target_embed - neutral_embed)

def interpolate_emotions(self, emotions: Dict[str, float], device=None):
    """Blend multiple emotions with normalized weights."""
    total_weight = sum(emotions.values())
    result = sum((w/total_weight) * self.embedding(idx) for em, w in emotions.items())
    return result
```

### 2. `src/chatterbox/models/t3/modules/emotion_cross_attention.py` (NEW)

**Purpose**: Cross-attention module for emotion conditioning.

**Architecture**:
```
emotion_embed (B, 64)
    ↓
emotion_proj: Linear(64, 1024)
    ↓
+ query_tokens (4, 1024)  ← learnable parameters
    ↓
cross_attention(queries, text_context)  ← emotion attends to text
    ↓
self_attention(queries, queries)  ← refinement
    ↓
FFN
    ↓
output (B, 4, 1024)  ← 4 conditioning tokens
```

**Key Parameters**:
- `hidden_size`: 1024 (model dimension)
- `emotion_dim`: 64 (input emotion dimension)
- `num_heads`: 8 (attention heads)
- `num_query_tokens`: 4 (output conditioning tokens)

### 3. `src/chatterbox/models/t3/modules/t3_config.py`

**Changes**:
```python
# Before
self.emotion_adv = True
self.emotion_embed_dim = 8

# After
self.emotion_embed_dim = 64
self.emotion_num_query_tokens = 4
self.emotion_cross_attn_heads = 8
```

### 4. `src/chatterbox/models/t3/modules/cond_enc.py`

**Changes to T3Cond**:
- Removed `emotion_adv` field
- `emotion_embed` now expects (B, 64) tensors
- Added legacy checkpoint handling in `load()`

**Changes to T3CondEnc**:
- Removed `emotion_adv_fc` and `emotion_embed_fc` linear layers
- Added `emotion_cross_attn` module
- Updated `forward()` to accept optional `text_context` parameter
- Cross-attention output is (B, 4, 1024) instead of (B, 1, 1024)

**Key Code**:
```python
class T3CondEnc(nn.Module):
    def __init__(self, hp: T3Config):
        # Removed: self.emotion_adv_fc = nn.Linear(1, hp.n_channels)
        # Removed: self.emotion_embed_fc = nn.Linear(8, hp.n_channels)

        # Added:
        self.emotion_cross_attn = EmotionCrossAttention(
            hidden_size=hp.n_channels,
            emotion_dim=hp.emotion_embed_dim,
            num_heads=hp.emotion_cross_attn_heads,
            num_query_tokens=hp.emotion_num_query_tokens,
        )

    def forward(self, cond: T3Cond, text_context: Optional[Tensor] = None):
        # ...
        if cond.emotion_embed is not None:
            cond_emotion = self.emotion_cross_attn(
                cond.emotion_embed,
                context=text_context,
            )  # (B, 4, n_channels)
```

### 5. `src/chatterbox/models/t3/t3.py`

**Changes**:
- Updated `prepare_conditioning()` to accept `text_tokens` parameter
- Text embeddings are computed and passed to `cond_enc` for cross-attention
- Updated `prepare_input_embeds()` to pass text tokens

**Key Code**:
```python
def prepare_conditioning(self, t3_cond: T3Cond, text_tokens: Optional[Tensor] = None):
    # Get text context for emotion cross-attention
    text_context = None
    if text_tokens is not None:
        text_context = self.text_emb(text_tokens)
        if self.hp.input_pos_emb == "learned":
            text_context = text_context + self.text_pos_emb(text_tokens)

    return self.cond_enc(t3_cond, text_context=text_context)
```

### 6. `src/chatterbox/mtl_tts.py`

**API Changes**:
```python
# Before
def generate(self, text, language_id, audio_prompt_path=None,
             exaggeration=0.5, emotion=None, ...):

# After
def generate(self, text, language_id, audio_prompt_path=None,
             emotion="neutral", emotion_intensity=1.0,
             emotion_blend=None, ...):
```

**New Features**:
- `emotion`: Default is "neutral" instead of None
- `emotion_intensity`: Controls strength (0.0-1.5 typical range)
- `emotion_blend`: Dict for mixing emotions, e.g., `{"happy": 0.7, "sad": 0.3}`

**Removed**:
- `exaggeration` parameter
- All `emotion_adv` handling code

### 7. `train_emotion_lora.py`

**Changes**:
- Updated to fine-tune `emotion_cross_attn` instead of `emotion_embed_fc`
- T3Cond creation no longer uses `emotion_adv`
- Comments updated for 64D embeddings

### 8. `src/chatterbox/models/t3/modules/lora_adapter.py`

**New Functions**:
```python
def apply_lora_to_emotion_cross_attention(emotion_cross_attn, rank=8, alpha=16.0):
    """Apply LoRA to emotion cross-attention Q/K/V projections."""

def get_emotion_trainable_params(model):
    """Get all trainable parameters for emotion fine-tuning."""
```

---

## New API Usage

### Basic Usage
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Generate with emotion
audio = model.generate(
    text="Hello, how are you today?",
    language_id="en",
    audio_prompt_path="reference.wav",
    emotion="happy"
)
```

### Intensity Control
```python
# Subtle emotion (30% intensity)
audio = model.generate(text, language_id="en", emotion="happy", emotion_intensity=0.3)

# Full emotion (100% intensity) - default
audio = model.generate(text, language_id="en", emotion="happy", emotion_intensity=1.0)

# Exaggerated emotion (130% intensity)
audio = model.generate(text, language_id="en", emotion="excited", emotion_intensity=1.3)
```

### Emotion Blending
```python
# Bittersweet (mix of happy and sad)
audio = model.generate(
    text="I'm happy you're here, but sad you have to leave.",
    language_id="en",
    emotion_blend={"happy": 0.4, "sad": 0.6}
)

# Nervous excitement
audio = model.generate(
    text="I can't believe this is happening!",
    language_id="en",
    emotion_blend={"excited": 0.6, "fearful": 0.4}
)
```

### Supported Emotions
```python
emotions = model.get_supported_emotions()
# ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm',
#  'surprised', 'fearful', 'disgusted', 'whisper', 'shout']
```

---

## Migration Guide

### For Users

**Before (v0)**:
```python
# Scalar intensity only
audio = model.generate(text, language_id="en", exaggeration=0.7)

# Or emotion type (mutually exclusive with exaggeration)
audio = model.generate(text, language_id="en", emotion="happy")
```

**After (v0.1)**:
```python
# Emotion with intensity
audio = model.generate(text, language_id="en", emotion="happy", emotion_intensity=0.7)

# Or neutral with intensity (equivalent to old exaggeration)
audio = model.generate(text, language_id="en", emotion="neutral", emotion_intensity=0.7)
```

### For Developers

1. **T3Cond changes**: Remove any `emotion_adv` references
2. **T3CondEnc changes**: Access `emotion_cross_attn` instead of `emotion_embed_fc`
3. **Training**: Update to use 64D embeddings and new T3Cond format

---

## Model Checkpoint Compatibility

When loading pre-trained checkpoints with the new code:

```
Warning: Missing keys in T3 model (will use random initialization):
  - cond_enc.emotion_cross_attn.emotion_proj.weight
  - cond_enc.emotion_cross_attn.emotion_proj.bias
  - cond_enc.emotion_cross_attn.query_tokens
  - cond_enc.emotion_cross_attn.cross_q_proj.weight
  - ... (other cross-attention parameters)
```

This is expected. The new emotion cross-attention layers will be randomly initialized and require fine-tuning on emotion-labeled data for optimal performance.

---

## Technical Details

### Embedding Initialization Structure

The 64D emotion embeddings are structured as:

| Dimensions | Purpose | Initialization |
|------------|---------|----------------|
| 0-2 | VAD (Valence, Arousal, Dominance) | Based on psychological models |
| 3-15 | Prosodic features | Heuristic values |
| 16-63 | Fine-grained features | Zeros (learned) |

**VAD Reference**:
- **Valence**: Positive (happy) ↔ Negative (sad)
- **Arousal**: High energy (angry) ↔ Low energy (calm)
- **Dominance**: Strong/confident ↔ Weak/submissive

### Cross-Attention Architecture

```
Input: emotion_embed (B, 64)

1. emotion_proj: Linear(64, 1024) → (B, 1, 1024)
2. query_tokens: Parameter(1, 4, 1024) expanded to (B, 4, 1024)
3. queries = query_tokens + emotion_proj  # broadcast

4. If text_context provided:
   cross_attn: MultiheadAttention(1024, 8)
   queries = queries + cross_attn(Q=queries, K=context, V=context)
   queries = LayerNorm(queries)

5. self_attn: MultiheadAttention(1024, 8)
   queries = queries + self_attn(Q=queries, K=queries, V=queries)
   queries = LayerNorm(queries)

6. FFN: Linear(1024, 4096) → GELU → Linear(4096, 1024)
   queries = queries + FFN(LayerNorm(queries))

Output: (B, 4, 1024)
```

### Parameter Count

| Component | Parameters |
|-----------|------------|
| EmotionEmbeddings (11 × 64) | 704 |
| EmotionCrossAttention | ~17M |
| - emotion_proj | 65K |
| - query_tokens | 4K |
| - cross_attn (Q/K/V/O) | 4.2M |
| - self_attn (Q/K/V/O) | 4.2M |
| - FFN | 8.4M |

The cross-attention module adds ~17M parameters, which is ~3% of the T3 model's ~520M parameters.

---

## Future Improvements

1. **Pre-trained emotion embeddings**: Initialize from speech emotion recognition models
2. **Dynamic emotion trajectories**: Allow emotion to vary within an utterance
3. **Emotion-text alignment loss**: Ensure emotions match semantic content
4. **Larger embedding dimension**: Consider 128D for more expressive capacity
5. **FiLM conditioning**: Add feature-wise modulation throughout transformer layers

---

## Version History

- **v0.1** (Current): 64D embeddings, cross-attention, intensity control, emotion blending
- **v0**: 8D heuristic embeddings, single-token concatenation, scalar exaggeration
