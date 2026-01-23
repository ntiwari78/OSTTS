# Emotion Architecture v0.3 - Advanced Training & Temporal Dynamics

This document provides a comprehensive overview of the emotion control system implemented in ChatterboxMultilingualTTS v0.3, featuring per-dataset training infrastructure, nonlinear intensity control, temporal emotion dynamics, and advanced checkpoint merging strategies.

## Table of Contents
- [Overview](#overview)
- [Architecture Evolution](#architecture-evolution)
- [Core Components](#core-components)
- [New Components in v0.3](#new-components-in-v03)
- [Per-Dataset Training Infrastructure](#per-dataset-training-infrastructure)
- [Checkpoint Merging Strategies](#checkpoint-merging-strategies)
- [Parameter Analysis](#parameter-analysis)
- [Usage Examples](#usage-examples)
- [Testing Framework](#testing-framework)

---

## Overview

The v0.3 emotion system introduces significant enhancements for training, inference, and emotion control:

1. **Per-Dataset Training**: Separate checkpoints for RAVDESS, CREMA-D, and IESC with full data validation
2. **Nonlinear Intensity Transform**: MLP-based intensity mapping for perceptually accurate emotion scaling
3. **Emotion Trajectory Module**: Temporal emotion dynamics (static/transition/keyframe modes)
4. **Emotion-Audio Alignment Losses**: Consistency, contrastive, and optional SER verification losses
5. **Expanded Emotion Vocabulary**: 16 emotions (5 new: sarcastic, bored, affectionate, contemptuous, awed)
6. **Balanced Sampling**: Per-emotion and per-dataset balanced training
7. **Advanced Merging**: DARE, task arithmetic, and dataset-adaptive merging strategies

### Key Metrics

| Component | Parameters | Trainable | Notes |
|-----------|------------|-----------|-------|
| Base T3 Transformer | ~520M | Frozen | LLaMA-based, 24 layers |
| LoRA on Transformer | ~4.5M | Yes | rank=8, applied to Q/K/V/O + MLP |
| EmotionCrossAttention | ~17M | Yes | Full module trained |
| EmotionEmbeddings | 1,024 | Yes | 16 emotions x 64D |
| IntensityTransform | ~25K | Yes | Nonlinear intensity MLP |
| EmotionTrajectory | ~200K | Yes | Temporal dynamics module |
| **Total Trainable** | **~22.7M** | - | **~4.2% of total model** |

---

## Architecture Evolution

### v0.2 -> v0.3 Changes

| Feature | v0.2 | v0.3 |
|---------|------|------|
| Emotions | 11 | 16 (+sarcastic, bored, affectionate, contemptuous, awed) |
| Intensity control | Linear interpolation | Nonlinear MLP-based transform |
| Temporal dynamics | None (constant emotion) | Static/Transition/Keyframe modes |
| Training | Single combined dataset | Per-dataset with validation |
| Alignment loss | None | Consistency + Contrastive + optional SER |
| Sampling | Random | Balanced emotion/dataset sampling |
| Merging strategies | Weighted average, TIES | +DARE, Task arithmetic, Dataset-adaptive |
| Test coverage | Manual testing | Comprehensive test suite (8 test classes) |

### High-Level Architecture

```
+-----------------------------------------------------------------------------+
|                      ChatterboxMultilingualTTS v0.3                          |
|                                                                              |
|  +-----------------------------------------------------------------------+  |
|  |                    T3 Model (with LoRA)                               |  |
|  |  +----------------------------------------------------------------+   |  |
|  |  |  LLaMA Transformer (24 layers)                                 |   |  |
|  |  |  +----------------------------------------------------------+  |   |  |
|  |  |  |  Self-Attention (with LoRA on Q, K, V, O projections)    |  |   |  |
|  |  |  |  MLP (with LoRA on gate, up, down projections)           |  |   |  |
|  |  |  +----------------------------------------------------------+  |   |  |
|  |  +----------------------------------------------------------------+   |  |
|  |                                                                        |  |
|  |  +----------------------------------------------------------------+   |  |
|  |  |               T3CondEnc (Conditioning Encoder)                 |   |  |
|  |  |  +----------------+  +-------------------------------------+   |   |  |
|  |  |  | Speaker Emb    |  | EmotionCrossAttention (with LoRA)   |   |   |  |
|  |  |  |  Projection    |  |   emotion_proj (64D -> 1024D)       |   |   |  |
|  |  |  +----------------+  |   query_tokens (4 x 1024D)          |   |   |  |
|  |  |                      |   cross_attn + self_attn + FFN      |   |   |  |
|  |  |  +----------------+  +-------------------------------------+   |   |  |
|  |  |  | Prompt Audio   |                                            |   |  |
|  |  |  |  (Perceiver)   |                                            |   |  |
|  |  |  +----------------+                                            |   |  |
|  |  +----------------------------------------------------------------+   |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  +-------------------+  +-------------------+  +-------------------------+   |
|  | EmotionEmbeddings |  | IntensityTransform|  | EmotionTrajectory       |   |
|  |  (16 x 64D)       |  |  (Nonlinear MLP)  |  |  (Static/Trans/Keyframe)|   |
|  +-------------------+  +-------------------+  +-------------------------+   |
|                                                                              |
|  +-----------------------------------------------------------------------+  |
|  |                    Training Losses                                     |  |
|  |  +---------------+  +-----------------+  +------------------+          |  |
|  |  | TTS Loss      |  | Consistency     |  | Contrastive      |          |  |
|  |  | (text+speech) |  | (MSE alignment) |  | (InfoNCE)        |          |  |
|  |  +---------------+  +-----------------+  +------------------+          |  |
|  |                                                                        |  |
|  |  +---------------+  +------------------+                               |  |
|  |  | Discriminator |  | SER Integration  |   (Optional)                  |  |
|  |  | (Adversarial) |  | (wav2vec2-emo)   |                               |  |
|  |  +---------------+  +------------------+                               |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
```

---

## Core Components

### 1. EmotionEmbeddings (64D, 16 Emotions)

**Location**: `src/chatterbox/models/t3/modules/emotion_embeddings.py`

The emotion embedding module now stores 64-dimensional learnable vectors for 16 emotion types.

```python
class EmotionEmbeddings(nn.Module):
    def __init__(
        self,
        emotion_embed_dim: int = 64,
        use_nonlinear_intensity: bool = False,
    ):
        self.embedding = nn.Embedding(num_emotions, emotion_embed_dim)
        self.intensity_transform = IntensityTransform(emotion_dim) if use_nonlinear_intensity else None
```

**64D Structure**:
| Dimensions | Purpose | Initialization |
|------------|---------|----------------|
| 0-2 | VAD (Valence, Arousal, Dominance) | Psychological models |
| 3-15 | Prosodic features (pitch, energy, rate, quality) | Heuristic values |
| 16-63 | Fine-grained learned features | Zeros (learned) |

**Supported Emotions** (16 types):
```
Original (11): neutral, happy, sad, angry, excited, calm, surprised, fearful, disgusted, whisper, shout
New (5):       sarcastic, bored, affectionate, contemptuous, awed
```

**New Emotions VAD Mappings**:
| Emotion | Valence | Arousal | Dominance | Description |
|---------|---------|---------|-----------|-------------|
| sarcastic | -0.2 | 0.3 | 0.4 | Ironic, slightly mocking tone |
| bored | -0.3 | -0.6 | -0.2 | Disinterested, flat affect |
| affectionate | 0.9 | 0.3 | 0.3 | Warm, loving, tender |
| contemptuous | -0.5 | 0.1 | 0.6 | Dismissive superiority |
| awed | 0.6 | 0.5 | -0.3 | Wonder, amazement |

### 2. IntensityTransform (NEW)

**Location**: `src/chatterbox/models/t3/modules/emotion_embeddings.py`

Nonlinear MLP-based intensity mapping for perceptually accurate emotion scaling.

```python
class IntensityTransform(nn.Module):
    """Nonlinear intensity mapping via MLP."""

    def __init__(self, emotion_dim: int = 64, hidden_dim: int = 128):
        self.transform = nn.Sequential(
            nn.Linear(emotion_dim + 1, hidden_dim),  # +1 for intensity scalar
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emotion_dim),
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
```

**Architecture Flow**:
```
Input: target_emotion (B, 64), neutral_emotion (B, 64), intensity (B, 1)
         |
         v
+--------------------------------+
| Concatenate: [target, intensity]|  -> (B, 65)
+--------------------------------+
         |
         v
+--------------------------------+
| MLP: 65 -> 128 -> 128 -> 64    |  GELU activations
+--------------------------------+
         |
         v nonlinear_delta
+--------------------------------+
| Blend with linear result:      |
| alpha = sigmoid(residual_weight)|
| result = alpha * linear +      |
|          (1-alpha) * nonlinear |
+--------------------------------+
         |
         v
Output: intensity-scaled embedding (B, 64)
```

**Why Nonlinear?**
- Linear interpolation assumes emotions lie on straight lines through neutral
- In reality, emotion space is curved (manifold)
- Nonlinear transform learns perceptually meaningful intensity curves
- Example: `intensity=0.5` for "happy" should produce "content", not "half-happy"

### 3. EmotionTrajectory (NEW)

**Location**: `src/chatterbox/models/t3/modules/emotion_trajectory.py`

Temporal emotion dynamics module supporting three modes.

```python
class EmotionTrajectory(nn.Module):
    """Generate time-varying emotion embeddings for an utterance."""

    def __init__(
        self,
        emotion_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_keyframes: int = 5,
        text_hidden_size: int = 1024,
    ):
        self.time_embed = nn.Sequential(...)       # Position encoding
        self.interpolation_net = nn.Sequential(...) # Learned interpolation
        self.text_cross_attn = nn.MultiheadAttention(...)  # Text-aware
```

**Three Modes**:

| Mode | Method | Use Case |
|------|--------|----------|
| Static | `forward_static()` | Same emotion throughout (backward compatible) |
| Transition | `forward_transition()` | Smooth interpolation between start/end emotions |
| Keyframe | `forward_keyframes()` | Multiple emotion keyframes with learned interpolation |

**Transition Mode Architecture**:
```
Input: start_embed (B, 64), end_embed (B, 64), seq_len
         |
         v
+----------------------------------------+
| Generate positions [0, 1] for seq_len  |
+----------------------------------------+
         |
         v
+----------------------------------------+
| Concatenate: [start, end, position]    |  -> (B, seq_len, 129)
+----------------------------------------+
         |
         v
+----------------------------------------+
| Interpolation MLP: 129 -> 128 -> 64    |  Learned interpolation
+----------------------------------------+
         |
         v
+----------------------------------------+
| Add time embedding modulation          |  Small contribution
+----------------------------------------+
         |
         v (optional)
+----------------------------------------+
| Text cross-attention                   |  Context-aware transitions
+----------------------------------------+
         |
         v
Output: trajectory (B, seq_len, 64)
```

**Keyframe Mode Example**:
```python
# Emotional arc: calm -> excited -> calm
keyframes = [calm_embed, excited_embed, calm_embed]
positions = [0.0, 0.5, 1.0]
trajectory = emotion_trajectory.forward_keyframes(keyframes, positions, seq_len=100)
```

### 4. Emotion Losses (NEW)

**Location**: `src/chatterbox/models/t3/modules/emotion_losses.py`

Four specialized loss functions for emotion-aware training.

#### 4.1 EmotionConsistencyLoss

```python
class EmotionConsistencyLoss(nn.Module):
    """MSE + Contrastive loss between emotion embed and audio features."""

    def forward(self, target_emotion_embed, audio_features, labels=None):
        # Project audio to emotion space
        predicted_emotion = self.audio_to_emotion(audio_features)

        # L2 consistency loss
        consistency_loss = F.mse_loss(predicted_emotion, target_emotion_embed)

        # InfoNCE contrastive loss
        sim_matrix = normalized_pred @ normalized_target.T / temperature
        contrastive_loss = F.cross_entropy(sim_matrix, torch.arange(B))

        return {
            "consistency_loss": consistency_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": consistency_loss + 0.5 * contrastive_loss,
        }
```

#### 4.2 EmotionDiscriminatorLoss

```python
class EmotionDiscriminatorLoss(nn.Module):
    """Adversarial training for emotion-speaker disentanglement."""

    def forward(self, emotion_embed, emotion_labels):
        logits = self.discriminator(emotion_embed)
        loss = F.cross_entropy(logits, emotion_labels)
        accuracy = (logits.argmax(dim=-1) == emotion_labels).float().mean()
        return {"discriminator_loss": loss, "discriminator_accuracy": accuracy}
```

#### 4.3 SERIntegrationLoss (Optional)

```python
class SERIntegrationLoss(nn.Module):
    """Uses pretrained wav2vec2-emotion to verify generated audio."""

    def forward(self, audio, target_emotions, sample_rate=16000):
        # Lazy load HuggingFace SER model
        logits = self._ser_model(audio)
        target_indices = [self.emotion_mapping[e] for e in target_emotions]
        return {
            "ser_loss": F.cross_entropy(logits, target_indices),
            "ser_accuracy": accuracy,
        }
```

#### 4.4 CombinedEmotionLoss

```python
class CombinedEmotionLoss(nn.Module):
    """Unified loss combining TTS + emotion losses."""

    def forward(self, tts_loss, emotion_embed, audio_features=None, ...):
        total_loss = tts_loss

        if audio_features is not None:
            total_loss += self.consistency_weight * consistency_loss

        if self.use_ser and audio is not None:
            total_loss += self.ser_weight * ser_loss

        if self.use_discriminator and emotion_labels is not None:
            total_loss += self.discriminator_weight * discriminator_loss

        return {"total_loss": total_loss, ...}
```

---

## Per-Dataset Training Infrastructure

### Dataset Configuration

**Location**: `train_emotion_lora.py`

```python
@dataclass
class DatasetConfig:
    name: str                              # Dataset identifier
    default_path: str                      # Default data directory
    emotion_mapping: Dict[str, str]        # Raw -> standard emotion names
    expected_samples: int                  # For validation
    language: str = "en"                   # Language code
    description: str = ""                  # Human-readable description
    unique_emotions: List[str] = field(default_factory=list)

DATASET_CONFIGS = {
    "ravdess": DatasetConfig(
        name="ravdess",
        default_path="data/ravdess_emotions",
        emotion_mapping={...},
        expected_samples=1440,
        unique_emotions=["neutral", "calm", "happy", "sad", "angry",
                         "fearful", "disgusted", "surprised"],
    ),
    "cremad": DatasetConfig(
        name="cremad",
        default_path="data/cremad_emotions",
        emotion_mapping={...},
        expected_samples=7442,
        unique_emotions=["neutral", "happy", "sad", "angry",
                         "fearful", "disgusted"],
    ),
    "iesc": DatasetConfig(
        name="iesc",
        default_path="data/hindi_emotions",
        emotion_mapping={...},
        expected_samples=600,
        language="hi",
        unique_emotions=["neutral", "happy", "sad", "angry", "surprised"],
    ),
}
```

### Data Validation

```python
def validate_dataset_coverage(dataset, config: DatasetConfig):
    """Ensure ALL samples are loaded, none missed."""
    actual_count = len(dataset)
    expected_count = config.expected_samples

    # Scan directories for all audio files
    all_files = set()
    for emotion_dir in Path(data_dir).iterdir():
        if emotion_dir.is_dir():
            for f in emotion_dir.glob("*.wav"):
                all_files.add(f.name)

    loaded_files = set(dataset.get_all_filenames())
    missed_files = all_files - loaded_files

    return {
        "valid": len(missed_files) == 0 and actual_count >= expected_count * 0.95,
        "actual_count": actual_count,
        "expected_count": expected_count,
        "missed_files": missed_files,
        "coverage_percent": (actual_count / expected_count) * 100,
    }
```

### Checkpoint Structure

```
checkpoints/
+-- emotion_lora_ravdess/
|   +-- checkpoint_epoch_1.pt
|   +-- checkpoint_epoch_2.pt
|   +-- checkpoint_early_stop.pt
|   +-- training_log.json          # Metrics, validation results
+-- emotion_lora_cremad/
|   +-- ...
+-- emotion_lora_iesc/
|   +-- ...
+-- emotion_merged/
    +-- checkpoint_merged.pt
    +-- merge_config.json          # Merge settings, weights used
```

### Training Log

```python
@dataclass
class TrainingLog:
    dataset_name: str
    start_time: str
    config: Dict
    epochs: List[Dict]                   # Per-epoch metrics
    final_metrics: Dict
    validation_results: Dict             # Data coverage validation
    early_stop_epoch: Optional[int]

    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str): ...
```

---

## Checkpoint Merging Strategies

**Location**: `merge_emotion_checkpoints.py`

### Available Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `weighted_average` | Weighted sum by dataset size | General use (default) |
| `ties` | Trim, Elect Sign, Merge | Conflicting updates |
| `dare` | Drop And Rescale | Reducing interference |
| `task_arithmetic` | base + sum(weight * (finetuned - base)) | Flexible composition |
| `dataset_adaptive` | Auto-weights based on emotion coverage | Balanced representation |

### DARE Merging (NEW)

```python
def dare_merge(state_dicts, weights, drop_rate=0.1, scaling_factor=1.0):
    """
    DARE: Drop And Rescale - reduces parameter interference.

    1. For each parameter, randomly drop some deltas (set to 0)
    2. Rescale remaining deltas by 1/(1-drop_rate)
    3. Merge rescaled deltas
    """
    for key in merged:
        mask = torch.rand_like(param) > drop_rate  # Keep mask
        delta = (param - base_param) * mask / (1 - drop_rate)  # Rescale
        merged[key] += weight * delta

    return merged
```

### Task Arithmetic Merging (NEW)

```python
def task_arithmetic_merge(base_state, task_vectors, weights):
    """
    Task Arithmetic: base + sum(weight * task_vector)

    Task vector = finetuned - base (direction of learning)
    """
    merged = base_state.copy()
    for key in merged:
        for task_vec, weight in zip(task_vectors, weights):
            merged[key] += weight * task_vec[key]
    return merged
```

### Dataset-Adaptive Weights (NEW)

```python
def compute_adaptive_weights(configs: List[DatasetConfig]):
    """Auto-compute weights based on dataset characteristics."""
    weights = []
    for cfg in configs:
        base = cfg.expected_samples
        # Bonus for emotion coverage (normalized to 0-1)
        emotion_bonus = len(cfg.unique_emotions) / 11
        weights.append(base * (1 + 0.2 * emotion_bonus))

    # Normalize to sum to 1
    total = sum(weights)
    return [w / total for w in weights]
```

**Computed Adaptive Weights**:
```
RAVDESS: 1440 * (1 + 0.2 * 8/11) = 1440 * 1.145 = 1649  -> 17.4%
CREMA-D: 7442 * (1 + 0.2 * 6/11) = 7442 * 1.109 = 8253  -> 87.0%
IESC:    600  * (1 + 0.2 * 5/11) = 600  * 1.091 = 655   -> 6.9%

Note: After emotion bonus, RAVDESS gets slight boost for having 8 emotions
```

---

## Balanced Sampling

### BalancedEmotionSampler

```python
class BalancedEmotionSampler(Sampler):
    """Ensures equal emotion representation per epoch."""

    def __init__(self, dataset):
        # Group indices by emotion
        self.emotion_indices = defaultdict(list)
        for idx, (_, _, emotion) in enumerate(dataset):
            self.emotion_indices[emotion].append(idx)

        # Find max count, oversample minorities
        self.max_count = max(len(v) for v in self.emotion_indices.values())

    def __iter__(self):
        # Oversample each emotion to max_count
        indices = []
        for emotion, idx_list in self.emotion_indices.items():
            # Repeat indices to match max_count
            repeated = idx_list * (self.max_count // len(idx_list) + 1)
            indices.extend(repeated[:self.max_count])
        random.shuffle(indices)
        return iter(indices)
```

### DatasetWeightedSampler

```python
class DatasetWeightedSampler(Sampler):
    """Balances across datasets in combined training."""

    def __init__(self, datasets: List, target_samples_per_dataset: int = None):
        # Equal samples from each dataset despite size differences
        if target_samples_per_dataset is None:
            target_samples_per_dataset = min(len(d) for d in datasets)

        self.indices = []
        offset = 0
        for dataset in datasets:
            dataset_indices = list(range(offset, offset + len(dataset)))
            # Sample with replacement if dataset is smaller
            sampled = random.choices(dataset_indices, k=target_samples_per_dataset)
            self.indices.extend(sampled)
            offset += len(dataset)
```

---

## Parameter Analysis

### v0.3 Trainable Parameters Breakdown

```
+--------------------------------------------------------------------+
|                    Trainable Parameters (v0.3)                      |
+--------------------------------------------------------------------+
| Component                        | Parameters | Notes               |
+----------------------------------|------------|---------------------|
| EmotionEmbeddings                |      1,024 | 16 emotions x 64D   |
| IntensityTransform               |     25,216 | 65->128->128->64    |
| EmotionTrajectory                |    ~200,000| Interp + text attn  |
|   - time_embed                   |      8,320 | 1->128->64          |
|   - interpolation_net            |     33,472 | 129->128->128->64   |
|   - text_proj                    |     65,600 | 1024->64            |
|   - text_cross_attn              |     16,512 | 4 heads             |
|   - output_norm                  |        128 | LayerNorm           |
+----------------------------------|------------|---------------------|
| EmotionCrossAttention            | ~17,000,000| Full module         |
| LoRA on Transformer (24 layers)  |  ~4,660,000| rank=8              |
+----------------------------------|------------|---------------------|
| TOTAL                            | ~22,700,000| ~4.2% of model      |
+--------------------------------------------------------------------+
```

### Training Loss Components

```python
# Combined loss computation
loss_total = (
    loss_tts                                    # text + 2*speech CE
    + consistency_weight * emotion_consistency  # MSE alignment
    + contrastive_weight * contrastive_loss     # InfoNCE
    + ser_weight * ser_loss                     # Optional SER
    + discriminator_weight * disc_loss          # Optional adversarial
)

# Default weights
consistency_weight = 0.5
contrastive_weight = 0.25  # Part of consistency loss
ser_weight = 0.3           # If enabled
discriminator_weight = 0.1 # If enabled
```

---

## Usage Examples

### Basic Emotion Control

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
model.load_emotion_checkpoint("checkpoints/emotion_merged/checkpoint_merged.pt")

# Generate with single emotion (all 16 emotions available)
audio = model.generate(
    text="This is quite remarkable, isn't it?",
    language_id="en",
    emotion="awed",          # NEW emotion
    emotion_intensity=1.0,
)

# Generate with new emotions
audio = model.generate(
    text="Oh, how fascinating. Really.",
    language_id="en",
    emotion="sarcastic",     # NEW emotion
    emotion_intensity=1.2,
)
```

### Emotion Transition (Trajectory)

```python
# Generate with emotion transition (start to end)
audio = model.generate(
    text="I started feeling hopeful, then became excited!",
    language_id="en",
    emotion="calm",           # Start emotion
    emotion_end="excited",    # End emotion (NEW)
)

# The audio will transition from calm prosody to excited prosody
```

### Emotion Blending

```python
# Blend multiple emotions including new ones
audio = model.generate(
    text="I suppose I find this mildly interesting.",
    language_id="en",
    emotion_blend={"bored": 0.5, "sarcastic": 0.3, "neutral": 0.2},
)
```

### Per-Dataset Training

```bash
# Train separately on each dataset
python train_emotion_lora.py \
    --dataset ravdess \
    --output_dir checkpoints/emotion_lora_ravdess \
    --epochs 3 \
    --balanced_sampling

python train_emotion_lora.py \
    --dataset cremad \
    --output_dir checkpoints/emotion_lora_cremad \
    --epochs 3 \
    --balanced_sampling

python train_emotion_lora.py \
    --dataset iesc \
    --language hi \
    --output_dir checkpoints/emotion_lora_iesc \
    --epochs 3 \
    --balanced_sampling
```

### Advanced Checkpoint Merging

```bash
# DARE merging (reduces parameter interference)
python merge_emotion_checkpoints.py \
    --checkpoints checkpoints/emotion_lora_*/checkpoint_early_stop.pt \
    --method dare \
    --drop-rate 0.1 \
    --output checkpoints/emotion_merged/checkpoint_dare.pt

# Dataset-adaptive merging (auto-weights by emotion coverage)
python merge_emotion_checkpoints.py \
    --checkpoints checkpoints/emotion_lora_*/checkpoint_early_stop.pt \
    --method dataset_adaptive \
    --output checkpoints/emotion_merged/checkpoint_adaptive.pt \
    --save-config
```

---

## Testing Framework

**Location**: `test_emotion_system.py`

### Test Classes

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestEmotionEmbeddings` | 5 | All 16 emotions, shapes, intensity |
| `TestIntensityTransform` | 4 | Nonlinear behavior, zero=neutral |
| `TestEmotionTrajectory` | 5 | Static/transition/keyframe modes |
| `TestEmotionCrossAttention` | 3 | Output shape, text context |
| `TestEmotionLosses` | 4 | Consistency, contrastive, combined |
| `TestCheckpointLoading` | 3 | Per-dataset checkpoints |
| `TestDatasetConfig` | 4 | Config validation, coverage |
| `TestBalancedSampling` | 3 | Samplers, distribution |

### Running Tests

```bash
# Run all tests
python test_emotion_system.py

# Run specific test class
python -m pytest test_emotion_system.py::TestEmotionTrajectory -v

# Run with coverage
python -m pytest test_emotion_system.py --cov=chatterbox.models.t3.modules
```

### Example Test

```python
class TestEmotionTrajectory(unittest.TestCase):
    def test_transition_mode(self):
        trajectory = EmotionTrajectory(emotion_dim=64)
        start = torch.randn(2, 64)
        end = torch.randn(2, 64)

        result = trajectory.forward_transition(start, end, seq_len=50)

        self.assertEqual(result.shape, (2, 50, 64))
        # Verify smooth transition
        distances = torch.norm(result[:, 1:] - result[:, :-1], dim=-1)
        self.assertTrue(distances.max() < 1.0, "Transition should be smooth")
```

---

## Files Reference

### Core Module Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `emotion_embeddings.py` | Embeddings + Intensity | `EmotionEmbeddings`, `IntensityTransform` |
| `emotion_trajectory.py` | Temporal dynamics | `EmotionTrajectory`, `EmotionKeyframe` |
| `emotion_losses.py` | Training losses | `EmotionConsistencyLoss`, `SERIntegrationLoss`, `CombinedEmotionLoss` |
| `emotion_cross_attention.py` | Cross-attention | `EmotionCrossAttention` |
| `cond_enc.py` | Conditioning encoder | `T3Cond`, `T3CondEnc` |
| `lora_adapter.py` | LoRA layers | `LoRALayer`, `LoRALinear` |

### Training & Utility Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `train_emotion_lora.py` | Training script | `DatasetConfig`, `TrainingLog`, `BalancedEmotionSampler` |
| `merge_emotion_checkpoints.py` | Checkpoint merging | `dare_merge`, `task_arithmetic_merge`, `dataset_adaptive_merge` |
| `test_emotion_system.py` | Test suite | 8 test classes, 30+ tests |

---

## Summary

The v0.3 emotion system provides significant enhancements:

### New Capabilities

1. **16 Emotions**: 5 new emotions (sarcastic, bored, affectionate, contemptuous, awed)
2. **Nonlinear Intensity**: Perceptually accurate intensity scaling via MLP
3. **Temporal Dynamics**: Emotion transitions and keyframe-based arcs
4. **Alignment Losses**: Explicit emotion-audio consistency training
5. **Per-Dataset Training**: Separate checkpoints with full validation
6. **Balanced Sampling**: Equal emotion/dataset representation
7. **Advanced Merging**: DARE, task arithmetic, adaptive weights
8. **Comprehensive Testing**: 8 test classes with 30+ tests

### Backward Compatibility

- All v0.2 APIs remain functional
- Static emotion mode is the default (no trajectory)
- Linear intensity still available (nonlinear is optional)
- Existing checkpoints can be loaded

### Performance Impact

| Operation | v0.2 | v0.3 | Notes |
|-----------|------|------|-------|
| Emotion lookup | 0.1ms | 0.1ms | Same |
| Intensity scaling | 0.1ms | 0.3ms | MLP overhead |
| Trajectory (static) | - | 0.1ms | Same as v0.2 |
| Trajectory (transition) | - | 1-2ms | Depends on seq_len |
| Cross-attention | 4-10ms | 4-10ms | Same |
| **Total emotion overhead** | 4-10ms | 5-12ms | +1-2ms |

The v0.3 architecture provides a robust foundation for emotion control with clear paths for future enhancement while maintaining practical training and inference costs.
