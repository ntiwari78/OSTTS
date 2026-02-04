# Emotion Architecture V0.5 - Training Stability & Regression Fix

**Version**: 0.5
**Date**: 2026-01-30
**Status**: Implemented
**Previous Version**: V0.4 (gated projection + FiLM + attention bias)
**Branch**: v05

---

## 1. Executive Summary

V0.5 is a stability-focused release that diagnoses and fixes the benchmark regression introduced in V0.4 (55.2% vs V0.3's 69.0% on emotion2vec). Rather than adding new architectural features, V0.5 corrects training infrastructure, makes V0.4 features opt-in, and adds regularization to prevent overfitting.

### Performance History

| Version | Checkpoint | emotion2vec | wav2vec2 (ehcalabres) | wav2vec2 (Dpngtm) |
|---------|-----------|-------------|----------------------|-------------------|
| V0.3 | RAVDESS-SER (epoch 15) | **69.0%** (20/29) | 17.2% (5/29) | 27.6% (8/29) |
| V0.4 | Combined-V04 (epoch 3) | **55.2%** (16/29) | 10.3% (3/29) | 20.7% (6/29) |
| V0.5 | *(pending training)* | Target: 72%+ | - | - |

### V0.4 Regression: -13.8% on emotion2vec

The V0.4 training ran for 15 epochs but peaked at epoch 3. The remaining 12 epochs degraded performance. Seven root causes were identified and fixed.

---

## 2. Regression Analysis

### 2.1 Per-Sample Regressions (V0.3 correct -> V0.4 incorrect)

| File | V0.3 Prediction | V0.4 Prediction | Pattern |
|------|----------------|----------------|---------|
| `basic_disgusted_1.0.wav` | disgusted (0.95) | **surprised** (0.90) | Lost emotion |
| `intensity_angry_1.0.wav` | angry (0.94) | **neutral** (0.999) | Neutral bias |
| `transition_calm_excited.wav` | happy (0.996) | **neutral** (1.0) | Neutral bias |
| `new_affectionate_1.0.wav` | neutral (0.78) | **angry** (0.65) | Confusion |
| `intensity_angry_0.5.wav` | angry (0.92) | **neutral** (0.69) | Neutral bias |

Only 1 improvement (basic_angry: fearful -> angry). Net: -4 correct predictions.

**Key pattern**: 3 of 5 regressions predict "neutral" with high confidence (0.69-1.0). The V0.4 model exhibits a strong neutral bias caused by under-trained V0.4 modules suppressing the emotion signal.

### 2.2 Per-Emotion Accuracy Comparison (emotion2vec)

| Emotion | V0.3 | V0.4 | Delta | Notes |
|---------|------|------|-------|-------|
| sad | 100% (5/5) | 100% (5/5) | 0% | Stable |
| fearful | 100% (1/1) | 100% (1/1) | 0% | Stable |
| neutral | 100% (1/1) | 100% (1/1) | 0% | Stable |
| happy | 71.4% (5/7) | 71.4% (5/7) | 0% | Stable |
| bored | 100% (1/1) | 100% (1/1) | 0% | Stable |
| sarcastic | 100% (1/1) | 100% (1/1) | 0% | Stable |
| disgusted | 100% (1/1) | **0%** (0/1) | **-100%** | Regressed |
| angry | 40% (2/5) | **20%** (1/5) | **-20%** | Regressed |
| excited | 100% (2/2) | **50%** (1/2) | **-50%** | Regressed |
| affectionate | 100% (1/1) | **0%** (0/1) | **-100%** | Regressed |

---

## 3. Root Cause Analysis

### 3.1 No Learning Rate Schedule (Primary Cause)

**Impact**: Training peaks at epoch 3, degrades for remaining 12 epochs.

The V0.4 training used a constant `lr=1e-4` with no warmup or decay. With ~8,882 samples at batch_size=4, each epoch has ~2,220 steps. By epoch 3 (~6,660 steps), the model has learned the basic emotion patterns. The constant LR then pushes the model past the optimal point, causing oscillation and overfitting.

```
Loss Trajectory (conceptual):

  Loss   ████                         Constant LR: overshoots optimal
  ▲      ██  ████
  │     ██      ████████████████      With cosine decay: converges
  │    ██          ▓▓▓▓▓▓▓▓▓▓▓▓
  │   ██              ▓▓▓▓▓▓▓▓▓
  │  ██
  └──────────────────────────────► Epoch
       3        8        15
```

### 3.2 V0.4 Architecture Always Active (Critical Bug)

**Impact**: Under-trained modules inject noise, cause neutral bias.

`EmotionCrossAttention.__init__` defaulted to V0.4 features ON:
- `use_gated_projection=True` (gate sigmoid(0)=0.5 suppresses emotion by 50%)
- `use_film_fusion=True` (scale=0, shift=0, no modulation)
- `use_attention_bias=True` (std=0.02, near-zero bias)

`cond_enc.py` did NOT pass these flags, so the defaults applied even for V0.3 training. Result: every model instance silently used under-trained V0.4 modules.

### 3.3 Curriculum Learning Bundled with `--v04_all`

**Impact**: Distribution shift mid-training destabilizes learned representations.

The `--v04_all` flag enabled curriculum learning, which restricts early epochs to "easy" emotions:
- Epochs 1-4: happy, angry, sad, surprised, whisper, shout
- Epochs 5-9: +fearful, disgusted, excited, neutral
- Epochs 10-15: +calm, bored, sarcastic, affectionate, contemptuous, awed

Since whisper/shout don't exist in RAVDESS/CREMA-D, early epochs trained on only 4 emotions. When new emotions were introduced, the distribution shift disrupted previously learned weights.

### 3.4 Dynamic Loss Weighting Was Dead Code

**Impact**: Feature provided zero benefit despite being enabled.

```python
# train_emotion_lora.py line 957
loss = v04_manager.apply_loss_weights(loss, list(emotions), predictions=None)
#                                                            ^^^^^^^^^^^^^^^^
# Always None -> _accumulate_stats never called -> weights stuck at 1.0
```

The `DynamicEmotionLossWeight` module required prediction labels to compute per-emotion accuracy, but predictions were never available in the TTS training loop. The weights never updated.

### 3.5 Insufficient Regularization

**Impact**: Model overfits to training distribution, poor generalization.

`weight_decay=1e-5` is 1000x below the standard `0.01` for AdamW. With ~22.7M trainable parameters and only ~8,882 samples, strong regularization is essential.

### 3.6 Unbalanced Combined Dataset

**Impact**: CREMA-D (84% of data) dominates, RAVDESS-specific emotions under-represented.

| Dataset | Samples | % of Combined | Unique Emotions |
|---------|---------|---------------|-----------------|
| CREMA-D | 7,442 | 83.8% | 6 (no calm, surprised) |
| RAVDESS | 1,440 | 16.2% | 8 |

Emotions like calm and surprised had only ~180 samples vs ~1,240 for shared emotions. Without balanced sampling, the optimizer gradient was dominated by CREMA-D patterns.

### 3.7 Query Token Config Mismatch

**Impact**: V0.4 features designed for 8 tokens operated with only 4.

`t3_config.py` set `emotion_num_query_tokens=4` (V0.3 value), but V0.4 architecture was designed for 8 semantic tokens (pitch, energy, timing, voice quality pairs). The FiLM fusion's `_init_query_tokens()` tried to scale 8 token indices, but only 4 existed.

---

## 4. V0.5 Fix Strategy

### Design Principles

1. **Safe defaults**: V0.3 architecture and behavior by default
2. **Opt-in complexity**: V0.4 features require explicit flags
3. **No new architecture**: Fix training infrastructure, not model design
4. **Backward compatible**: V0.3 checkpoints load and work without changes

### Fix Summary

| # | Fix | Impact | Files Modified |
|---|-----|--------|----------------|
| 1 | Cosine LR scheduler with warmup | Prevents overfitting past optimal point | `train_emotion_lora.py` |
| 2 | V0.4 architecture opt-in | Eliminates under-trained module noise | `t3_config.py`, `cond_enc.py`, `emotion_cross_attention.py`, `train_emotion_lora.py` |
| 3 | Decouple curriculum from `--v04_all` | Prevents accidental distribution shift | `train_emotion_lora.py` |
| 4 | Fix dynamic loss weighting | Makes feature functional via loss-based adaptation | `training_utils_v04.py` |
| 5 | Increase weight_decay to 0.01 | Proper regularization against overfitting | `train_emotion_lora.py` |
| 6 | Auto-balance combined datasets | Prevents CREMA-D dominance | `train_emotion_lora.py` |
| 7 | Improved V0.4 module initialization | Stronger initial emotion signal when V0.4 enabled | `emotion_cross_attention.py` |

---

## 5. Expected Training Behavior (V0.5)

### Default Configuration (V0.3 Architecture)

```bash
python train_emotion_lora.py --dataset ravdess --epochs 15
```

- Architecture: V0.3 (simple linear projection, 4 query tokens, additive injection)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Scheduler: Cosine decay with 500-step linear warmup, min_lr=1e-6
- No curriculum, no dynamic weighting
- Expected: steady improvement through epoch 10-12, gradual plateau

### Combined Dataset

```bash
python train_emotion_lora.py --dataset combined --epochs 15
```

- Same as above, plus auto-balanced dataset sampling
- Balanced sampling ensures RAVDESS emotions (calm, surprised) get fair representation

### V0.4 Architecture (Explicit Opt-In)

```bash
python train_emotion_lora.py --dataset combined --epochs 15 \
    --use_v04_architecture --v04_all
```

- Architecture: V0.4 (gated projection, FiLM fusion, attention bias, 8 query tokens)
- Dynamic loss weighting: active, loss-based adaptation
- Hard negative mining: active
- Curriculum: OFF (requires separate `--use_curriculum`)

---

## 6. Architecture Comparison

### V0.3 vs V0.4 vs V0.5 Defaults

| Component | V0.3 | V0.4 (broken) | V0.5 Default | V0.5 + --use_v04_architecture |
|-----------|------|---------------|--------------|-------------------------------|
| Emotion projection | Linear(64->1024) | GatedEmotionProjection | Linear(64->1024) | GatedEmotionProjection |
| Query injection | Additive (q + e) | FiLM (q * (1+s) + b) | Additive (q + e) | FiLM (q * (1+s) + b) |
| Query tokens | 4 | 4 (config) / 8 (code) | 4 | 8 |
| Attention bias | None | EmotionAttentionBias | None | EmotionAttentionBias |
| Weight decay | 1e-5 | 1e-5 | **0.01** | **0.01** |
| LR schedule | None | None | **Cosine + warmup** | **Cosine + warmup** |
| Curriculum | Off | **On** (via v04_all) | Off | Off (explicit only) |
| Dynamic weights | Off | On (broken) | Off | **On (fixed)** |
| Balanced sampling | Optional | Optional | **Auto for combined** | **Auto for combined** |

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| V0.3 architecture still under-performs on hard emotions (calm, surprised, disgusted) | High | These were 0% in both V0.3 and V0.4. V0.5 focuses on training stability, not architecture. V0.4 architecture can be opted in once training is stable. |
| Cosine scheduler decays too fast for small datasets | Low | min_lr=1e-6 prevents complete decay. `--warmup_steps` and `--min_lr` are configurable. |
| Balanced sampling slows convergence on well-represented emotions | Medium | Balanced sampling is auto-enabled only for combined datasets. Single-dataset training is unchanged. |
| V0.4 architecture still doesn't converge when opted in | Medium | Improved initialization (Fix 7) gives stronger initial signal. Combined with LR schedule and weight decay, V0.4 modules have a better chance of training properly. |

---

## 8. Future Work (V0.6+)

1. **Validation-based checkpoint selection**: Add a held-out validation set and select best checkpoint by SER accuracy, not training loss
2. **Per-module learning rates**: Use higher LR for V0.4 modules (GatedProjection, FiLM, AttentionBias) and lower LR for base model LoRA
3. **Gradient accumulation**: Enable effective larger batch sizes for better gradient estimates
4. **SER-in-the-loop training**: Use emotion2vec predictions during training to directly optimize for SER accuracy
5. **Emotion-specific intensity calibration**: Apply `emotion_intensity_calibration.py` during inference to boost under-recognized emotions
