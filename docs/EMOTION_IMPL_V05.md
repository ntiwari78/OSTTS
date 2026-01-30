# Emotion Implementation V0.5 - Detailed Code Changes

**Version**: 0.5
**Date**: 2026-01-30
**Branch**: v05
**Companion Document**: [EMOTION_ARCH_V05.md](EMOTION_ARCH_V05.md) (analysis & improvement plan)

---

## 1. Change Summary

| Stat | Value |
|------|-------|
| Files modified | 5 |
| Insertions | 149 |
| Deletions | 31 |
| Net lines added | 118 |

```
src/chatterbox/models/t3/modules/cond_enc.py            |  5 ++
src/chatterbox/models/t3/modules/emotion_cross_attention.py | 22 ++---
src/chatterbox/models/t3/modules/t3_config.py            |  3 +
src/chatterbox/models/t3/modules/training_utils_v04.py   | 55 +++++++++----
train_emotion_lora.py                                    | 95 +++++++++++++++++--
```

---

## 2. Fix-by-Fix Implementation

### Fix 1: Cosine LR Scheduler with Warmup

**File**: `train_emotion_lora.py`
**Root cause**: Constant `lr=1e-4` with no decay caused the model to overshoot past the optimal point at epoch 3 and degrade for the remaining 12 epochs.

#### 2.1.1 Import (line 40)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
```

#### 2.1.2 CLI Arguments (lines 1065-1071)

```python
parser.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "step", "none"],
                   help="Learning rate scheduler (default: cosine)")
parser.add_argument("--warmup_steps", type=int, default=500,
                   help="Number of warmup steps for LR scheduler (default: 500)")
parser.add_argument("--min_lr", type=float, default=1e-6,
                   help="Minimum learning rate for cosine scheduler (default: 1e-6)")
```

#### 2.1.3 Scheduler Creation (lines 1614-1631)

Created after dataloader setup so `total_steps` is known:

```python
total_steps = args.epochs * len(dataloader)
scheduler = None
if args.scheduler == "cosine":
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - args.warmup_steps, 1),
        eta_min=args.min_lr,
    )
elif args.scheduler == "step":
    from torch.optim.lr_scheduler import StepLR
    step_size = max(total_steps // 3, 1)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
```

#### 2.1.4 Per-Batch Warmup & Step (lines 1705-1714)

Inside the training loop, after each `train_step` call:

```python
global_step += 1
if scheduler is not None:
    if global_step <= args.warmup_steps:
        # Linear warmup: scale LR from 0 to base_lr
        warmup_factor = global_step / max(args.warmup_steps, 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * warmup_factor
    else:
        scheduler.step()
```

#### 2.1.5 LR in Progress Bar (line 1733-1739)

```python
current_lr = optimizer.param_groups[0]['lr']
progress_bar.set_postfix({
    "loss": f"{loss:.4f}" if loss > 0 else "skip",
    "lr": f"{current_lr:.2e}",
    ...
})
```

#### 2.1.6 LR Logged in Training Config (lines 1202-1205)

```python
"weight_decay": args.weight_decay,
"scheduler": args.scheduler,
"warmup_steps": args.warmup_steps,
"min_lr": args.min_lr,
```

---

### Fix 2: V0.4 Architecture Opt-In

**Root cause**: `EmotionCrossAttention.__init__` defaulted to V0.4 features ON (`use_gated_projection=True`, `use_film_fusion=True`, `use_attention_bias=True`). `cond_enc.py` did NOT pass these flags, so every model instance silently used under-trained V0.4 modules.

#### 2.2a `t3_config.py` (line 30)

Added config flag to control V0.4 features:

```python
# V0.4 emotion architecture features (disabled by default for v03 compat)
self.use_v04_emotion_features = False
```

#### 2.2b `cond_enc.py` (lines 71-80)

Now reads config and explicitly passes V0.4 flags:

```python
# V0.4 features are only enabled if config explicitly sets use_v04_emotion_features=True
use_v04 = getattr(hp, 'use_v04_emotion_features', False)
self.emotion_cross_attn = EmotionCrossAttention(
    hidden_size=hp.n_channels,
    emotion_dim=hp.emotion_embed_dim,
    num_heads=hp.emotion_cross_attn_heads,
    num_query_tokens=hp.emotion_num_query_tokens,
    use_gated_projection=use_v04,
    use_film_fusion=use_v04,
    use_attention_bias=use_v04,
)
```

#### 2.2c `emotion_cross_attention.py` (lines 308-319)

Changed constructor defaults from V0.4-ON to V0.3-safe:

```python
def __init__(
    self,
    hidden_size: int = 1024,
    emotion_dim: int = 64,
    num_heads: int = 8,
    num_query_tokens: int = 4,      # Was: 8.  V0.3 default
    dropout: float = 0.1,
    use_flash_attention: bool = True,
    use_gated_projection: bool = False,  # Was: True.  V0.4 feature, off by default
    use_film_fusion: bool = False,       # Was: True.  V0.4 feature, off by default
    use_attention_bias: bool = False,    # Was: True.  V0.4 feature, off by default
    num_emotions: int = 16,
):
```

#### 2.2d `train_emotion_lora.py` (lines 1235-1253)

Added `--use_v04_architecture` CLI flag. When set, reconfigures the model after `from_pretrained`:

```python
if args.use_v04_architecture:
    model.t3.hp.use_v04_emotion_features = True
    model.t3.hp.emotion_num_query_tokens = 8
    # Reinitialize emotion cross attention with v04 features
    from chatterbox.models.t3.modules.emotion_cross_attention import EmotionCrossAttention
    model.t3.cond_enc.emotion_cross_attn = EmotionCrossAttention(
        hidden_size=model.t3.hp.n_channels,
        emotion_dim=model.t3.hp.emotion_embed_dim,
        num_heads=model.t3.hp.emotion_cross_attn_heads,
        num_query_tokens=8,
        use_gated_projection=True,
        use_film_fusion=True,
        use_attention_bias=True,
    )
```

---

### Fix 3: Decouple Curriculum from `--v04_all`

**File**: `train_emotion_lora.py`
**Root cause**: `--v04_all` bundled curriculum learning, causing distribution shift mid-training when new emotions were introduced.

Changed ~5 locations from:
```python
# BEFORE (V0.4)
if args.use_curriculum or args.v04_all:
```

To:
```python
# AFTER (V0.5)
if args.use_curriculum:
```

Updated `--v04_all` help text (line 1140-1142):
```python
parser.add_argument("--v04_all", action="store_true",
                   help="Enable V0.4 training improvements (dynamic weights + hard negatives). "
                        "Curriculum learning requires explicit --use_curriculum flag.")
```

---

### Fix 4: Fix Dynamic Loss Weighting

**File**: `training_utils_v04.py`
**Root cause**: `apply_loss_weights` was always called with `predictions=None` in the TTS training loop, so `_accumulate_stats` never ran and weights never updated.

#### 2.4.1 New `_accumulate_loss_stats` Method (lines 232-243)

Tracks per-emotion loss when predictions are unavailable:

```python
def _accumulate_loss_stats(
    self,
    labels: torch.Tensor,
    loss: torch.Tensor,
):
    """Accumulate loss statistics for weight updates (no predictions needed)."""
    labels_np = labels.cpu().numpy()
    loss_val = loss.item() if loss.dim() == 0 else loss.detach().mean().item()

    for label in labels_np:
        self.total_counts[label] += 1
        self.loss_sums[label] += loss_val
```

#### 2.4.2 Updated `forward()` Fallback (lines 199-204)

Calls `_accumulate_loss_stats` when predictions are `None`:

```python
if predictions is not None:
    self._accumulate_stats(indices, predictions, loss)
else:
    # Track loss-only stats when predictions unavailable (e.g., TTS training)
    self._accumulate_loss_stats(indices, loss)
```

#### 2.4.3 Dual-Mode `_update_weights()` (lines 248-295)

Supports both accuracy-based and loss-based adaptation:

```python
def _update_weights(self):
    new_weights = self.weights.clone()
    has_accuracy_data = bool(self.correct_counts)

    for emotion_idx in range(self.num_emotions):
        total = self.total_counts.get(emotion_idx, 0)
        if total == 0:
            continue

        avg_loss = self.loss_sums.get(emotion_idx, 1.0) / total

        # Update running loss with momentum
        self.running_loss[emotion_idx] = (
            self.momentum * self.running_loss[emotion_idx]
            + (1 - self.momentum) * avg_loss
        )

        if has_accuracy_data:
            # Accuracy-based: higher weight for lower accuracy
            correct = self.correct_counts.get(emotion_idx, 0)
            accuracy = correct / total
            self.running_accuracy[emotion_idx] = (
                self.momentum * self.running_accuracy[emotion_idx]
                + (1 - self.momentum) * accuracy
            )
            new_weight = 1.0 + (1.0 - self.running_accuracy[emotion_idx])
        else:
            # Loss-based: higher weight for above-average loss
            mean_loss = self.running_loss.mean().item()
            if mean_loss > 0:
                loss_ratio = self.running_loss[emotion_idx].item() / mean_loss
                new_weight = max(0.5, min(2.0, loss_ratio))
            else:
                new_weight = 1.0

        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        new_weights[emotion_idx] = new_weight

    self.weights.copy_(new_weights)
```

---

### Fix 5: Increase Weight Decay

**File**: `train_emotion_lora.py`
**Root cause**: `weight_decay=1e-5` was 1000x below the standard `0.01` for AdamW with ~22.7M trainable parameters and ~8,882 samples.

#### 2.5.1 CLI Argument (line 1063)

```python
parser.add_argument("--weight_decay", type=float, default=0.01,
                   help="Weight decay for AdamW optimizer (default: 0.01)")
```

#### 2.5.2 Optimizer Usage (line 1508)

```python
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
```

Previously this was hardcoded to `weight_decay=1e-5`.

---

### Fix 6: Auto-Balance Combined Datasets

**File**: `train_emotion_lora.py`
**Root cause**: CREMA-D (7,442 samples, 83.8%) dominated combined training over RAVDESS (1,440 samples, 16.2%). RAVDESS-specific emotions (calm, surprised) had ~180 samples vs ~1,240 for shared emotions.

#### 2.6.1 CLI Argument (lines 1078-1079)

```python
parser.add_argument("--no_balanced_sampling", action="store_true",
                   help="Disable automatic balanced sampling for combined datasets")
```

#### 2.6.2 Auto-Enable Logic (lines 1153-1157)

After `parse_args()`, automatically enables balanced sampling for combined datasets:

```python
if args.dataset == "combined" and not args.no_balanced_sampling:
    if not args.balanced_sampling and not args.balanced_datasets:
        print("Auto-enabling balanced dataset sampling for combined training")
        args.balanced_datasets = True
```

---

### Fix 7: Improved V0.4 Module Initialization

**File**: `emotion_cross_attention.py`
**Root cause**: When V0.4 IS explicitly enabled, the near-identity initialization was too weak. Under-trained modules suppressed the emotion signal.

#### 2.7.1 `GatedEmotionProjection._init_weights` (lines 77-87)

Gate bias changed from `0` to `1.0` (sigmoid(1) ~ 0.73 vs sigmoid(0) = 0.5):

```python
def _init_weights(self):
    # Gate initialized to ~0.73 output (sigmoid(1) ~ 0.73) for stronger initial emotion signal
    nn.init.constant_(self.gate_proj[0].bias, 1.0)
    nn.init.xavier_uniform_(self.gate_proj[0].weight, gain=0.5)

    for module in self.value_proj:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            nn.init.zeros_(module.bias)
```

#### 2.7.2 `EmotionQueryFusion._init_weights` (lines 142-148)

Scale net bias changed from `0` to `0.1` (slight positive modulation):

```python
def _init_weights(self):
    nn.init.zeros_(self.scale_net[-1].weight)
    nn.init.constant_(self.scale_net[-1].bias, 0.1)  # Slight positive scale
    nn.init.zeros_(self.shift_net[-1].weight)
    nn.init.zeros_(self.shift_net[-1].bias)
```

#### 2.7.3 `EmotionAttentionBias._init_weights` (lines 220-224)

Embedding init std changed from `0.02` to `0.05`:

```python
def _init_weights(self):
    nn.init.normal_(self.emotion_bias.weight, mean=0.0, std=0.05)
    nn.init.xavier_uniform_(self.context_proj.weight, gain=0.5)
    nn.init.zeros_(self.context_proj.bias)
```

---

## 3. New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--weight_decay` | float | `0.01` | AdamW weight decay (was hardcoded `1e-5`) |
| `--scheduler` | str | `cosine` | LR scheduler: `cosine`, `step`, or `none` |
| `--warmup_steps` | int | `500` | Linear warmup steps before scheduler |
| `--min_lr` | float | `1e-6` | Minimum LR for cosine scheduler |
| `--use_v04_architecture` | flag | off | Enable V0.4 emotion modules |
| `--no_balanced_sampling` | flag | off | Disable auto-balance for combined datasets |
| `--use_curriculum` | flag | off | Explicitly enable curriculum learning |

---

## 4. Backward Compatibility

### Default Behavior (V0.3 Architecture)

```bash
python train_emotion_lora.py --dataset ravdess --epochs 15
```

- Architecture: V0.3 (Linear projection, 4 query tokens, additive injection)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Scheduler: Cosine decay with 500-step warmup, min_lr=1e-6
- No curriculum, no dynamic weighting

### Combined Dataset (Auto-Balanced)

```bash
python train_emotion_lora.py --dataset combined --epochs 15
```

- Same as above + `balanced_datasets=True` auto-enabled
- Balanced sampling ensures RAVDESS emotions get fair representation

### V0.4 Architecture (Explicit)

```bash
python train_emotion_lora.py --dataset combined --epochs 15 \
    --use_v04_architecture --v04_all
```

- Architecture: V0.4 (gated projection, FiLM, attention bias, 8 query tokens)
- Dynamic loss weighting: active, loss-based adaptation
- Hard negative mining: active
- Curriculum: OFF (requires separate `--use_curriculum`)

### Full V0.4 with Curriculum

```bash
python train_emotion_lora.py --dataset combined --epochs 15 \
    --use_v04_architecture --v04_all --use_curriculum
```

### Checkpoint Compatibility

V0.3 checkpoints load without changes because:
1. `T3Config.use_v04_emotion_features` defaults to `False`
2. `EmotionCrossAttention` defaults to V0.3 params (4 tokens, no gating, no FiLM, no bias)
3. `cond_enc.py` reads config via `getattr(hp, 'use_v04_emotion_features', False)`

---

## 5. Data Flow Changes

### V0.4 (Broken)

```
Model loads → cond_enc.__init__() creates EmotionCrossAttention with defaults
                → defaults = v04 ON (gated=True, film=True, bias=True, 8 tokens)
                → BUT config says 4 tokens → query token mismatch
                → Under-trained modules inject noise
                → Training overshoots at epoch 3 due to constant LR
                → No regularization (weight_decay=1e-5)
                → Curriculum shifts distribution mid-training
                → Dynamic weights stuck at 1.0 (predictions=None)
```

### V0.5 (Fixed)

```
Model loads → T3Config sets use_v04_emotion_features=False
           → cond_enc.__init__() reads config, passes v04=False
           → EmotionCrossAttention uses v03 defaults (4 tokens, linear proj, additive)
           → Optimizer: AdamW with weight_decay=0.01
           → Scheduler: Cosine + 500-step warmup
           → Combined datasets: auto-balanced
           → No curriculum unless --use_curriculum
           → Dynamic weights: loss-based adaptation when predictions=None

If --use_v04_architecture:
   → train script reconfigures after from_pretrained
   → hp.use_v04_emotion_features = True
   → hp.emotion_num_query_tokens = 8
   → Reinitializes EmotionCrossAttention with v04 features
   → Improved init: gate bias=1.0, scale bias=0.1, attn std=0.05
```

---

## 6. Verification

### Quick Smoke Test

```python
# Verify V0.3 defaults
python -c "
from chatterbox.models.t3.modules.emotion_cross_attention import EmotionCrossAttention
m = EmotionCrossAttention()
print('query_tokens:', m.num_query_tokens, 'gated:', m.use_gated_projection)
# Expected: query_tokens: 4 gated: False
"
```

### CLI Args

```bash
python train_emotion_lora.py --help | grep -E "scheduler|warmup|min_lr|weight_decay|v04_architecture|no_balanced|use_curriculum"
```

Expected new args: `--scheduler`, `--warmup_steps`, `--min_lr`, `--weight_decay`, `--use_v04_architecture`, `--no_balanced_sampling`, `--use_curriculum`.

### Training Dry Run

```bash
python train_emotion_lora.py --dataset ravdess --epochs 1
```

Verify:
- Cosine scheduler prints at startup
- LR shown in progress bar (`lr: X.XXe-XX`)
- `weight_decay=0.01` in config output
- V0.3 architecture message: "Using V0.3 emotion architecture (simple projection, 4 query tokens)"

---

## 7. File Reference

| File | Lines Changed | Purpose |
|------|--------------|---------|
| [t3_config.py](../src/chatterbox/models/t3/modules/t3_config.py) | 29-30 | V0.4 feature flag |
| [cond_enc.py](../src/chatterbox/models/t3/modules/cond_enc.py) | 70-80 | Pass V0.4 flags from config |
| [emotion_cross_attention.py](../src/chatterbox/models/t3/modules/emotion_cross_attention.py) | 77-87, 142-148, 220-224, 308-319 | V0.3 defaults + improved V0.4 init |
| [training_utils_v04.py](../src/chatterbox/models/t3/modules/training_utils_v04.py) | 199-204, 232-243, 248-295 | Loss-based dynamic weighting |
| [train_emotion_lora.py](../train_emotion_lora.py) | 40, 1063-1071, 1078-1079, 1132, 1140-1145, 1153-1157, 1235-1253, 1614-1631, 1705-1714 | Scheduler, CLI args, auto-balance, V0.4 opt-in |
