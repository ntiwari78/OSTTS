# Emotion Architecture v0.2 - LoRA/Adapter Fine-tuning System

This document provides a comprehensive overview of the emotion control system implemented in ChatterboxMultilingualTTS v0.2, focusing on the LoRA/Adapter-based fine-tuning architecture.

## Table of Contents
- [Overview](#overview)
- [Architecture Evolution](#architecture-evolution)
- [Core Components](#core-components)
- [LoRA Architecture](#lora-architecture)
- [Cross-Attention Mechanism](#cross-attention-mechanism)
- [Training Pipeline](#training-pipeline)
- [Checkpoint Management](#checkpoint-management)
- [Parameter Analysis](#parameter-analysis)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)

---

## Overview

The v0.2 emotion system introduces a sophisticated fine-tuning architecture using LoRA (Low-Rank Adaptation) for efficient parameter-efficient training on emotion-labeled datasets. The system combines:

1. **64D Learnable Emotion Embeddings**: Rich emotion representations
2. **Cross-Attention Conditioning**: Text-aware emotion modulation
3. **LoRA Adapters**: Efficient fine-tuning of transformer layers
4. **Multi-Dataset Training**: Combined training on RAVDESS, CREMA-D, and IESC
5. **Checkpoint Merging**: Unified model from multiple trained checkpoints

### Key Metrics

| Component | Parameters | Trainable | Notes |
|-----------|------------|-----------|-------|
| Base T3 Transformer | ~520M | Frozen | LLaMA-based, 24 layers |
| LoRA on Transformer | ~4.5M | Yes | rank=8, applied to Q/K/V/O + MLP |
| EmotionCrossAttention | ~17M | Yes | Full module trained |
| EmotionEmbeddings | 704 | Yes | 11 emotions × 64D |
| **Total Trainable** | **~22.5M** | - | **~4.2% of total model** |

**Parameter Breakdown**:
- LoRA on 24 transformer layers: ~4.5M (rank=8 on 7 projections × 24 layers)
- EmotionCrossAttention (full): ~17M (emotion_proj + 2×attention + FFN)
- EmotionEmbeddings: 704 (11 × 64)

---

## Architecture Evolution

### v0.1 → v0.2 Changes

| Feature | v0.1 | v0.2 |
|---------|------|------|
| Embedding dimension | 8D | 64D |
| Conditioning | Single FC projection | 4-token cross-attention |
| Fine-tuning | Manual weight updates | LoRA with rank=8 |
| Datasets | Single dataset | Multi-dataset merging |
| Intensity control | Scalar `exaggeration` | `emotion_intensity` parameter |
| Emotion blending | Not supported | Full interpolation support |
| Checkpoint size | Full model (~2GB) | LoRA-only (~86MB merged) |

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ChatterboxMultilingualTTS v0.2                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    T3 Model (with LoRA)                         │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │  LLaMA Transformer (24 layers)                            │  │   │
│  │  │  ┌─────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  Self-Attention (with LoRA on Q, K, V, O projections)│  │  │   │
│  │  │  │  ┌───────────────────────────────────────────────┐  │  │  │   │
│  │  │  │  │  base_weight + (lora_A @ lora_B) * scaling    │  │  │  │   │
│  │  │  │  └───────────────────────────────────────────────┘  │  │  │   │
│  │  │  ├─────────────────────────────────────────────────────┤  │  │   │
│  │  │  │  MLP (with LoRA on gate, up, down projections)      │  │  │   │
│  │  │  │  ┌───────────────────────────────────────────────┐  │  │  │   │
│  │  │  │  │  base_weight + (lora_A @ lora_B) * scaling    │  │  │  │   │
│  │  │  │  └───────────────────────────────────────────────┘  │  │  │   │
│  │  │  └─────────────────────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │               T3CondEnc (Conditioning Encoder)            │  │   │
│  │  │  ┌──────────────┐  ┌────────────────────────────────────┐ │  │   │
│  │  │  │ Speaker Emb  │  │  EmotionCrossAttention (with LoRA) │ │  │   │
│  │  │  │  Projection  │  │  ┌──────────────────────────────┐  │ │  │   │
│  │  │  └──────────────┘  │  │ emotion_proj (64D → 1024D)   │  │ │  │   │
│  │  │                    │  │ query_tokens (4 × 1024D)     │  │ │  │   │
│  │  │  ┌──────────────┐  │  │ cross_attn (Q,K,V,O + LoRA)  │  │ │  │   │
│  │  │  │ Prompt Audio │  │  │ self_attn (Q,K,V,O + LoRA)   │  │ │  │   │
│  │  │  │  (Perceiver) │  │  │ FFN                          │  │ │  │   │
│  │  │  └──────────────┘  │  └──────────────────────────────┘  │ │  │   │
│  │  │                    └────────────────────────────────────┘ │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────┐  ┌────────────────────────────────────────────┐   │
│  │ EmotionEmbeddings│  │                S3Gen                       │   │
│  │  (11 × 64D)     │  │           (Audio Generator)                │   │
│  └─────────────────┘  └────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. EmotionEmbeddings (64D)

**Location**: `src/chatterbox/models/t3/modules/emotion_embeddings.py`

The emotion embedding module stores 64-dimensional learnable vectors for each emotion type.

```python
class EmotionEmbeddings(nn.Module):
    def __init__(self, emotion_embed_dim: int = 64, emotion_types: dict = None):
        self.embedding = nn.Embedding(num_emotions, emotion_embed_dim)
```

**64D Structure**:
| Dimensions | Purpose | Initialization |
|------------|---------|----------------|
| 0-2 | VAD (Valence, Arousal, Dominance) | Psychological models |
| 3-15 | Prosodic features (pitch, energy, rate, quality) | Heuristic values |
| 16-63 | Fine-grained learned features | Zeros (learned) |

**Supported Emotions** (11 types):
```
neutral, happy, sad, angry, excited, calm, surprised, fearful, disgusted, whisper, shout
```

**Key Methods**:
- `get_emotion_embedding(emotion_name, intensity=1.0)`: Get embedding with intensity scaling
- `interpolate_emotions(emotions_dict)`: Blend multiple emotions
- `get_supported_emotions()`: List all available emotions

### 2. EmotionCrossAttention

**Location**: `src/chatterbox/models/t3/modules/emotion_cross_attention.py`

Cross-attention module that allows emotion to modulate based on text context.

```python
class EmotionCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,      # Model dimension
        emotion_dim: int = 64,         # Input emotion dimension
        num_heads: int = 8,            # Attention heads
        num_query_tokens: int = 4,     # Output conditioning tokens
    ):
```

**Architecture Flow**:
```
emotion_embed (B, 64)
    │
    ▼
emotion_proj: Linear(64, 1024)
    │
    ▼ normalize
emotion_proj (B, 1, 1024)
    │
    ▼ broadcast + add
query_tokens (1, 4, 1024) ──► queries (B, 4, 1024)
    │
    ▼ (if text_context provided)
cross_attention(queries, text_context)
    │
    ▼
queries (B, 4, 1024)
    │
    ▼
self_attention(queries)
    │
    ▼
FFN(queries)
    │
    ▼
output (B, 4, 1024) ──► 4 conditioning tokens
```

**Why 4 Query Tokens?**
The 4 query tokens approximately capture:
1. **Pitch patterns** - Fundamental frequency characteristics
2. **Energy dynamics** - Volume and intensity patterns
3. **Speaking rate** - Temporal characteristics
4. **Voice quality** - Breathiness, tension, etc.

### 3. T3CondEnc (Conditioning Encoder)

**Location**: `src/chatterbox/models/t3/modules/cond_enc.py`

Processes all conditioning inputs and produces the final conditioning sequence.

```python
class T3CondEnc(nn.Module):
    def __init__(self, hp: T3Config):
        self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        self.emotion_cross_attn = EmotionCrossAttention(
            hidden_size=hp.n_channels,
            emotion_dim=hp.emotion_embed_dim,
            num_heads=hp.emotion_cross_attn_heads,
            num_query_tokens=hp.emotion_num_query_tokens,
        )
        self.perceiver = Perceiver()  # For prompt audio
```

**Conditioning Concatenation**:
```
cond_embeds = concat([
    cond_spkr,              # (B, 1, 1024) - Speaker embedding
    cond_clap,              # (B, 0, 1024) - CLAP (not implemented)
    cond_prompt_speech_emb, # (B, L, 1024) - Prompt audio
    cond_emotion,           # (B, 4, 1024) - Emotion cross-attention output
])
```

---

## LoRA Architecture

### LoRA Layer Design

**Location**: `src/chatterbox/models/t3/modules/lora_adapter.py`

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to linear layers without modifying the base weights.

```python
class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,          # LoRA rank (r)
        alpha: float = 16.0,    # Scaling factor
        dropout: float = 0.0,
    ):
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
```

**Forward Pass**:
```python
def forward(self, x):
    # output = base_output + (x @ A^T @ B^T) * scaling
    lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
    return lora_output
```

**LoRALinear Wrapper**:
```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        self.base_layer = base_layer  # Frozen
        self.lora = LoRALayer(...)    # Trainable

    def forward(self, x):
        return self.base_layer(x) + self.lora(x)
```

### LoRA Application Points

LoRA is applied to the following layers in the T3 transformer:

**1. Self-Attention (per layer × 24 layers)**:
```
tfmr.layers.{i}.self_attn.q_proj  ← LoRA(1024, 1024, rank=8)
tfmr.layers.{i}.self_attn.k_proj  ← LoRA(1024, 1024, rank=8)
tfmr.layers.{i}.self_attn.v_proj  ← LoRA(1024, 1024, rank=8)
tfmr.layers.{i}.self_attn.o_proj  ← LoRA(1024, 1024, rank=8)
```

**2. MLP (per layer × 24 layers)**:
```
tfmr.layers.{i}.mlp.gate_proj    ← LoRA(1024, 4096, rank=8)
tfmr.layers.{i}.mlp.up_proj      ← LoRA(1024, 4096, rank=8)
tfmr.layers.{i}.mlp.down_proj    ← LoRA(4096, 1024, rank=8)
```

**3. EmotionCrossAttention**:
```
cond_enc.emotion_cross_attn.emotion_proj      ← LoRA(64, 1024, rank=8)
cond_enc.emotion_cross_attn.cross_q_proj      ← LoRA(1024, 1024, rank=8)
cond_enc.emotion_cross_attn.cross_k_proj      ← LoRA(1024, 1024, rank=8)
cond_enc.emotion_cross_attn.cross_v_proj      ← LoRA(1024, 1024, rank=8)
cond_enc.emotion_cross_attn.cross_out_proj    ← LoRA(1024, 1024, rank=8)
cond_enc.emotion_cross_attn.self_q_proj       ← LoRA(1024, 1024, rank=8)
cond_enc.emotion_cross_attn.self_k_proj       ← LoRA(1024, 1024, rank=8)
cond_enc.emotion_cross_attn.self_v_proj       ← LoRA(1024, 1024, rank=8)
cond_enc.emotion_cross_attn.self_out_proj     ← LoRA(1024, 1024, rank=8)
```

### LoRA Parameter Count

For a linear layer `(in_features, out_features)` with rank `r`:
```
LoRA params = r × in_features + out_features × r
            = r × (in_features + out_features)
```

**Example: q_proj (1024 → 1024, rank=8)**:
```
params = 8 × (1024 + 1024) = 16,384
```

**Total LoRA Parameters (24 layers)**:
```
Per layer:
  - 4 attention projs: 4 × 8 × (1024 + 1024) = 65,536
  - 3 MLP projs:
    - gate/up: 2 × 8 × (1024 + 4096) = 81,920
    - down: 8 × (4096 + 1024) = 40,960
  Total per layer: 188,416

24 layers: 24 × 188,416 = 4,521,984

Emotion cross-attention:
  - emotion_proj: 8 × (64 + 1024) = 8,704
  - 8 attention projs: 8 × 8 × (1024 + 1024) = 131,072
  Total: 139,776

Grand Total LoRA: ~4.66M parameters
```

---

## Cross-Attention Mechanism

### Why Cross-Attention?

Previous implementations used simple concatenation of emotion embeddings, which had limitations:
1. **Weak signal**: Single 768D token gets "washed out" in long sequences
2. **No context awareness**: Same emotion regardless of text content
3. **Limited expressiveness**: Single embedding for all prosodic aspects

Cross-attention solves these issues:
1. **Stronger signal**: 4 conditioning tokens instead of 1
2. **Text awareness**: Emotion attends to text, adapting to content
3. **Multi-aspect**: Different query tokens for pitch, energy, rate, quality

### Cross-Attention Flow

```
Input:
  - emotion_embed: (B, 64) - from EmotionEmbeddings
  - text_context: (B, L, 1024) - from text embedding layer

Step 1: Project emotion
  emotion_proj = Linear(64→1024)(emotion_embed)  # (B, 1024)
  emotion_proj = LayerNorm(emotion_proj)
  emotion_proj = emotion_proj.unsqueeze(1)       # (B, 1, 1024)

Step 2: Initialize queries
  queries = query_tokens.expand(B, 4, 1024)      # (B, 4, 1024)
  queries = queries + emotion_proj               # Broadcast add

Step 3: Cross-attention to text
  Q = queries @ W_q                              # (B, 4, 1024)
  K = text_context @ W_k                         # (B, L, 1024)
  V = text_context @ W_v                         # (B, L, 1024)
  attn = softmax(Q @ K.T / sqrt(d)) @ V          # (B, 4, 1024)
  queries = queries + attn                       # Residual

Step 4: Self-attention refinement
  Q = K = V = queries                            # Self-attention
  attn = softmax(Q @ K.T / sqrt(d)) @ V
  queries = queries + attn

Step 5: FFN
  queries = queries + FFN(LayerNorm(queries))

Output: (B, 4, 1024) - 4 conditioning tokens
```

### Attention Visualization

```
Cross-Attention Pattern:

Query Tokens (4)          Text Context (L tokens)
    ┌─┐                   ┌─────────────────────┐
    │1│ ── attends to ──► │ Hello how are you   │
    │2│ ── attends to ──► │ ? I am feeling ...  │
    │3│ ── attends to ──► │                     │
    │4│ ── attends to ──► │                     │
    └─┘                   └─────────────────────┘
     ▼
  Emotion-modulated based on text content
  (e.g., "?" triggers question intonation)
```

---

## Training Pipeline

### Dataset Structure

```
data/
├── ravdess_emotions/           # 1,440 files (English)
│   ├── emotion_angry/
│   ├── emotion_calm/
│   ├── emotion_disgusted/
│   ├── emotion_fearful/
│   ├── emotion_happy/
│   ├── emotion_neutral/
│   ├── emotion_sad/
│   └── emotion_surprised/
├── cremad_emotions/            # 7,442 files (English)
│   ├── emotion_angry/
│   ├── emotion_disgusted/
│   ├── emotion_fearful/
│   ├── emotion_happy/
│   ├── emotion_neutral/
│   └── emotion_sad/
└── hindi_emotions/             # ~600 files (Hindi)
    ├── emotion_angry/
    ├── emotion_happy/
    ├── emotion_neutral/
    ├── emotion_sad/
    └── emotion_surprised/
```

### Training Script

**Location**: `train_emotion_lora.py`

```bash
# Fine-tune on a single dataset
python train_emotion_lora.py \
    --data_dir data/ravdess_emotions \
    --output_dir checkpoints/emotion_lora_ravdess \
    --epochs 3 \
    --batch_size 2 \
    --lr 5e-5 \
    --early_stop_loss 0.4 \
    --early_stop_patience 50

# Fine-tune on Hindi data
python train_emotion_lora.py \
    --data_dir data/hindi_emotions \
    --output_dir checkpoints/emotion_lora_iesc \
    --language hi \
    --epochs 3 \
    --batch_size 2 \
    --lr 5e-5
```

### Training Loop

```python
# 1. Freeze base model
for param in model.t3.parameters():
    param.requires_grad = False

# 2. Apply LoRA to transformer layers
apply_lora_to_linear(model.t3.tfmr, rank=8, alpha=16.0)

# 3. Apply LoRA to emotion cross-attention
apply_lora_to_emotion_cross_attention(model.t3.cond_enc.emotion_cross_attn)

# 4. Unfreeze emotion components
for param in model.emotion_embeddings.parameters():
    param.requires_grad = True
for param in model.t3.cond_enc.emotion_cross_attn.parameters():
    param.requires_grad = True

# 5. Training loop
for epoch in range(epochs):
    for batch in dataloader:
        # Forward pass
        loss_text, loss_speech, loss_total = model.t3.forward_train(
            text_tokens=batch["text_tokens"],
            speech_tokens=batch["speech_tokens"],
            t3_cond=batch["conditioning"],
        )

        # Backward pass (only LoRA params updated)
        loss_total.backward()
        optimizer.step()

        # Early stopping check
        if loss_total < early_stop_loss:
            early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                save_checkpoint()
                break
```

### Loss Function

```python
# Combined loss with speech weighting
loss_total = loss_text + 2.0 * loss_speech

# Where:
# - loss_text: Cross-entropy on text token predictions
# - loss_speech: Cross-entropy on speech token predictions
# - Speech weighted 2x because it directly affects audio quality
```

---

## Checkpoint Management

### Checkpoint Structure

```python
checkpoint = {
    "epoch": epoch,
    "loss": loss,
    "emotion_embeddings_state_dict": OrderedDict({
        "embedding.weight": tensor(11, 64),  # 704 params
    }),
    "t3_state_dict": OrderedDict({
        # Base layer weights (frozen, but saved for completeness)
        "tfmr.layers.0.self_attn.q_proj.base_layer.weight": tensor(1024, 1024),

        # LoRA weights (trainable)
        "tfmr.layers.0.self_attn.q_proj.lora.lora_A": tensor(8, 1024),
        "tfmr.layers.0.self_attn.q_proj.lora.lora_B": tensor(1024, 8),

        # Emotion cross-attention (trainable)
        "cond_enc.emotion_cross_attn.emotion_proj.weight": tensor(1024, 64),
        "cond_enc.emotion_cross_attn.query_tokens": tensor(1, 4, 1024),
        ...
    }),
    "optimizer_state_dict": {...},
}
```

### Checkpoint Sizes

| Checkpoint Type | Size | Contents |
|-----------------|------|----------|
| Full training checkpoint | ~2.2 GB | Base weights + LoRA + optimizer state |
| Merged LoRA-only | ~86 MB | Only LoRA + emotion weights |

### Checkpoint Merging

**Location**: `merge_emotion_checkpoints.py`

Combines multiple fine-tuned checkpoints into a unified model.

```bash
# Merge with auto-weights (based on dataset sizes)
python merge_emotion_checkpoints.py \
    --auto-weights \
    --output checkpoints/emotion_merged/checkpoint_merged.pt

# Custom weights
python merge_emotion_checkpoints.py \
    --weights 0.15,0.78,0.07 \
    --output checkpoints/emotion_merged/checkpoint_merged.pt
```

**Merging Strategies**:

1. **Weighted Average** (default):
```python
merged_weight = sum(weight_i * checkpoint_i[key] for i in checkpoints)
merged_weight /= sum(weights)
```

2. **TIES Merging** (Trim, Elect Sign, Merge):
```python
# 1. Trim: Keep only top-k% magnitude values
# 2. Elect Sign: Majority vote on sign
# 3. Merge: Average values that agree with elected sign
```

**Auto-Computed Weights** (based on dataset sizes):
```
RAVDESS: 1,440 / 9,482 = 15.2%
CREMA-D: 7,442 / 9,482 = 78.5%
IESC:      600 / 9,482 =  6.3%
```

### Loading Merged Checkpoint

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load base model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Load merged emotion checkpoint
model.load_emotion_checkpoint("checkpoints/emotion_merged/checkpoint_merged.pt")

# Generate with emotion
audio = model.generate(
    text="Hello! I'm so happy to see you!",
    language_id="en",
    emotion="happy",
    emotion_intensity=1.2
)
```

---

## Parameter Analysis

### Trainable vs Frozen Parameters

```
┌─────────────────────────────────────────────────────────────────┐
│                    T3 Model (~540M parameters)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  FROZEN: Base Transformer Weights (~518M)               │   │
│  │  - embed_tokens                                          │   │
│  │  - layers.*.self_attn.{q,k,v,o}_proj.base_layer         │   │
│  │  - layers.*.mlp.{gate,up,down}_proj.base_layer          │   │
│  │  - layers.*.{input,post_attention}_layernorm            │   │
│  │  - norm, lm_heads                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TRAINABLE: LoRA Weights (~4.7M)                        │   │
│  │  - layers.*.self_attn.{q,k,v,o}_proj.lora.{A,B}        │   │
│  │  - layers.*.mlp.{gate,up,down}_proj.lora.{A,B}         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TRAINABLE: EmotionCrossAttention (~17M)                │   │
│  │  - emotion_proj: 64 × 1024 = 65K                        │   │
│  │  - query_tokens: 1 × 4 × 1024 = 4K                      │   │
│  │  - cross_attn (Q,K,V,O): 4 × 1024 × 1024 = 4.2M        │   │
│  │  - self_attn (Q,K,V,O): 4 × 1024 × 1024 = 4.2M         │   │
│  │  - FFN: 1024 × 4096 × 2 = 8.4M                          │   │
│  │  - LayerNorms: ~16K                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TRAINABLE: EmotionEmbeddings (704)                     │   │
│  │  - embedding.weight: 11 × 64 = 704                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Requirements

| Training Mode | GPU Memory | Speed |
|---------------|------------|-------|
| Full fine-tuning | ~40GB | Slow |
| LoRA (rank=8) | ~12GB | Fast |
| LoRA (rank=4) | ~10GB | Faster |
| Inference only | ~6GB | - |

---

## Usage Examples

### Basic Emotion Control

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
model.load_emotion_checkpoint("checkpoints/emotion_merged/checkpoint_merged.pt")

# Generate with single emotion
audio = model.generate(
    text="Hello, how are you today?",
    language_id="en",
    emotion="happy",
    emotion_intensity=1.0,  # Default
)

# Generate with increased intensity
audio = model.generate(
    text="This is amazing news!",
    language_id="en",
    emotion="excited",
    emotion_intensity=1.3,  # Exaggerated
)
```

### Emotion Blending

```python
# Blend multiple emotions
audio = model.generate(
    text="I'm happy you're here, but sad you have to leave.",
    language_id="en",
    emotion_blend={"happy": 0.4, "sad": 0.6},
)

# Nervous excitement
audio = model.generate(
    text="I can't believe this is happening!",
    language_id="en",
    emotion_blend={"excited": 0.6, "fearful": 0.4},
)
```

### Multilingual Emotions

```python
# Hindi with emotion
audio = model.generate(
    text="मैं बहुत खुश हूं!",
    language_id="hi",
    emotion="happy",
)

# Spanish with emotion
audio = model.generate(
    text="¡Estoy muy emocionado!",
    language_id="es",
    emotion="excited",
)
```

### Loading Specific Checkpoints

```python
# Load RAVDESS-only checkpoint (best for 8 emotions)
model.load_emotion_checkpoint("checkpoints/emotion_lora_ravdess/checkpoint_early_stop.pt")

# Load CREMA-D-only checkpoint (largest dataset)
model.load_emotion_checkpoint("checkpoints/emotion_lora_cremad/checkpoint_early_stop.pt")

# Load IESC-only checkpoint (best for Hindi)
model.load_emotion_checkpoint("checkpoints/emotion_lora_iesc/checkpoint_early_stop.pt")

# Load merged checkpoint (best overall)
model.load_emotion_checkpoint("checkpoints/emotion_merged/checkpoint_merged.pt")
```

---

## Technical Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA rank | 8 | Low-rank dimension |
| LoRA alpha | 16.0 | Scaling factor |
| LoRA scaling | 2.0 | alpha / rank |
| Learning rate | 5e-5 | AdamW optimizer |
| Batch size | 2 | Limited by memory |
| Early stop loss | 0.4 | Stop when loss < threshold |
| Early stop patience | 50 | Consecutive batches |
| Emotion embed dim | 64 | Embedding dimension |
| Query tokens | 4 | Cross-attention outputs |
| Attention heads | 8 | For cross/self attention |

### T3Config Parameters

```python
class T3Config:
    def __init__(self):
        # Emotion-specific config
        self.emotion_embed_dim = 64
        self.emotion_num_query_tokens = 4
        self.emotion_cross_attn_heads = 8

        # Model dimensions
        self.n_channels = 1024
        self.n_layers = 24
        self.n_heads = 16
        self.speaker_embed_size = 256
```

### Files Reference

| File | Purpose |
|------|---------|
| `src/chatterbox/models/t3/modules/emotion_embeddings.py` | 64D learnable embeddings |
| `src/chatterbox/models/t3/modules/emotion_cross_attention.py` | Cross-attention module |
| `src/chatterbox/models/t3/modules/cond_enc.py` | Conditioning encoder |
| `src/chatterbox/models/t3/modules/lora_adapter.py` | LoRA layers |
| `src/chatterbox/models/t3/modules/t3_config.py` | Configuration |
| `src/chatterbox/mtl_tts.py` | Main TTS class |
| `train_emotion_lora.py` | Training script |
| `merge_emotion_checkpoints.py` | Checkpoint merger |
| `prepare_emotion_data.py` | Dataset preparation |
| `example_english_emotions.py` | English examples |
| `example_hindi_emotions.py` | Hindi examples |
| `example_english_emotions_merged.py` | Merged checkpoint examples |

---

## Design Decisions & Alternatives

This section documents the design choices made and the alternatives that were considered.

### 1. Embedding Dimension: 64D vs Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **8D** (v0.1) | Minimal parameters, fast | Limited expressiveness, couldn't capture nuanced prosodic patterns | Rejected |
| **32D** | Good balance | May still be limiting for complex emotions | Considered |
| **64D** (chosen) | Rich representation, structured initialization possible, sufficient for 11 emotions | More parameters (~700 vs ~88) | **Selected** |
| **128D** | Maximum expressiveness | Diminishing returns, harder to train with limited data | Rejected |

**Rationale**: 64D allows structured initialization (VAD + prosodic + learned features) while keeping the embedding table small enough to train on ~10K samples.

### 2. Conditioning Mechanism: Cross-Attention vs Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Simple concatenation** (v0.1) | Simple, fast, fewer parameters | Single token gets "washed out", no text awareness | Rejected |
| **FiLM conditioning** | Proven in image synthesis | Requires modifying all transformer layers, complex integration | Considered for future |
| **Cross-attention** (chosen) | Text-aware, multiple output tokens, established pattern (CLIP, Flamingo) | ~17M parameters, adds latency | **Selected** |
| **Prefix tuning** | Parameter efficient | Less expressive than cross-attention | Rejected |

**Rationale**: Cross-attention provides text-aware emotion modulation (e.g., questions vs. statements can have different prosody even with same emotion). The 4 output tokens provide stronger conditioning signal than single-token approaches.

### 3. Number of Query Tokens: 4 vs Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **1 token** | Minimal overhead | Weak signal, same issue as v0.1 | Rejected |
| **2 tokens** | Low overhead | May not capture all prosodic aspects | Considered |
| **4 tokens** (chosen) | Maps to pitch/energy/rate/quality, good balance | 4x conditioning sequence length | **Selected** |
| **8 tokens** | Maximum expressiveness | Diminishing returns, longer conditioning | Rejected |

**Rationale**: 4 tokens roughly correspond to the main prosodic dimensions (pitch patterns, energy dynamics, speaking rate, voice quality). More tokens showed diminishing returns in informal testing.

### 4. Fine-tuning Strategy: LoRA vs Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Full fine-tuning** | Maximum adaptation | ~40GB GPU, slow, catastrophic forgetting risk | Rejected |
| **LoRA** (chosen) | ~4% parameters, memory efficient, no forgetting | May have limited capacity | **Selected** |
| **Adapters** | Well-studied, similar efficiency | More complex architecture changes | Available as option |
| **Prompt tuning** | Very few parameters | Limited for audio generation tasks | Rejected |
| **BitFit** (bias-only) | Minimal parameters | Insufficient for emotion adaptation | Rejected |

**Rationale**: LoRA provides the best balance of efficiency and capacity. The ~22.5M trainable parameters (4.2%) is sufficient to learn emotion-to-prosody mappings without risking base model degradation.

### 5. LoRA Rank: 8 vs Alternatives

| Option | Parameters/Layer | Capacity | Memory | Decision |
|--------|------------------|----------|--------|----------|
| **rank=4** | ~8K | Limited | Lower | For constrained hardware |
| **rank=8** (chosen) | ~16K | Good balance | Moderate | **Selected** |
| **rank=16** | ~32K | High | Higher | For larger datasets |
| **rank=32** | ~64K | Very high | High | Overkill for emotion |

**Rationale**: rank=8 with alpha=16.0 (scaling=2.0) provides sufficient capacity for emotion adaptation while keeping memory usage practical for consumer GPUs (~12GB).

### 6. Checkpoint Merging: Weighted Average vs Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Weighted average** (chosen) | Simple, predictable, works well in practice | May average out specialized knowledge | **Default** |
| **TIES merging** | Handles conflicting updates better | More complex, requires base checkpoint | Available as option |
| **Task arithmetic** | Flexible composition | Requires careful tuning | Rejected |
| **Model soup** | Good for similar tasks | Assumes similar training trajectories | Rejected |

**Rationale**: Weighted average (by dataset size) is simple and effective. TIES is available for cases where datasets have conflicting updates (e.g., different prosodic conventions).

### 7. Loss Function: 2x Speech Weight vs Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Equal weights** | Simple | Text loss dominates, poor audio quality | Rejected |
| **2x speech** (chosen) | Better audio quality, balanced gradients | May slightly hurt text prediction | **Selected** |
| **Speech only** | Focus on audio | Loses text-speech alignment | Rejected |
| **Learned weights** | Adaptive | More complex, needs validation data | Future work |

**Rationale**: Speech token prediction directly determines audio quality, so weighting it 2x helps the model focus on the primary objective.

### 8. Early Stopping: Loss < 0.4 vs Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Fixed epochs** | Simple | May overfit or underfit | Rejected |
| **Validation loss** | Gold standard | Requires held-out data | Ideal but data-limited |
| **Training loss threshold** (chosen) | Practical, prevents overfitting | Threshold needs tuning | **Selected** |
| **Gradient norm** | Detects convergence | Noisy signal | Rejected |

**Rationale**: With limited emotion data (~10K samples), validation splits would further reduce training data. The loss < 0.4 threshold was empirically determined to indicate good convergence without overfitting.

---

### Critical Analysis

**Potential Weaknesses**:

1. **Cross-attention adds latency**: The EmotionCrossAttention module adds ~17M parameters and computation. For real-time applications, this may be a bottleneck.

2. **Limited emotion granularity**: 11 predefined emotions may not capture all nuances (e.g., "sarcastic", "bored", "loving"). The interpolation feature partially addresses this.

3. **Dataset imbalance in merging**: CREMA-D dominates (78.5% weight). This may bias the model toward CREMA-D's prosodic patterns.

4. **No emotion-text alignment loss**: The model doesn't explicitly learn to match emotions to semantically appropriate text. A contrastive loss could help.

5. **Single-speaker emotion patterns**: Training data may encode speaker-specific emotion patterns rather than universal emotion prosody.

**Future Improvements**:

1. **FiLM conditioning**: Add feature-wise linear modulation throughout transformer layers for stronger emotion control
2. **Emotion trajectory**: Allow emotion to vary within an utterance
3. **Contrastive loss**: Add text-emotion alignment objective
4. **Larger embedding (128D)**: For more fine-grained emotions with more training data
5. **Dynamic query tokens**: Learn the optimal number of query tokens per emotion

---

## Summary

The v0.2 emotion system provides:

1. **Rich Representations**: 64D embeddings capture nuanced emotion characteristics
2. **Context-Aware Conditioning**: Cross-attention adapts emotion to text content
3. **Efficient Fine-tuning**: LoRA enables training with ~4% of parameters
4. **Multi-Dataset Support**: Merged checkpoints combine knowledge from multiple sources
5. **Flexible API**: Intensity control, emotion blending, and multi-language support

The architecture balances expressiveness with efficiency, enabling high-quality emotion control while maintaining practical training and inference costs.
