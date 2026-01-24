# Emotion TTS Benchmark Guide v0.3

This document provides detailed steps to benchmark the v0.3 emotion system across three separate dataset checkpoints: RAVDESS, CREMA-D, and IESC.

## Table of Contents

1. [Overview](#overview)
2. [Checkpoints to Benchmark](#checkpoints-to-benchmark)
3. [v0.3 New Features to Test](#v03-new-features-to-test)
4. [Benchmark Test Cases](#benchmark-test-cases)
   - [Basic Emotion Tests](#basic-emotion-tests)
   - [New Emotion Tests](#new-emotion-tests-v03)
   - [Intensity Variation Tests](#intensity-variation-tests)
   - [Nonlinearity Verification Tests](#nonlinearity-verification-tests)
   - [Emotion Transition Tests](#emotion-transition-tests)
   - [Keyframe Mode Tests](#keyframe-mode-tests)
   - [Context-Aware Transition Tests](#context-aware-transition-tests)
   - [Hindi Tests](#hindi-tests-iesc-only)
5. [Running the Benchmark](#running-the-benchmark)
6. [Expected Results](#expected-results)
7. [Output Structure](#output-structure)

---

## Overview

### What's New in v0.3 Benchmark

The v0.3 benchmark extends v0.2 with testing for:

| Feature | v0.2 | v0.3 |
|---------|------|------|
| Emotions tested | 11 | 16 (5 new emotions) |
| Intensity control | Linear only | Linear + Nonlinear (verified) |
| Temporal dynamics | None | Static/Transition/Keyframe/Context-aware |
| Per-dataset testing | Combined only | Individual + Combined |
| New emotions | - | sarcastic, bored, affectionate, contemptuous, awed |
| Test coverage | 11 emotions, basic tests | 16 emotions, 38 test cases |

### Benchmark Goals

1. **Per-Dataset Quality**: Evaluate each checkpoint (RAVDESS, CREMA-D, IESC) independently
2. **New Emotion Support**: Test 5 new emotions added in v0.3
3. **Intensity Nonlinearity**: Verify nonlinear intensity transform behavior (explicit verification tests)
4. **Emotion Transitions**: Test temporal emotion dynamics (static, transition, keyframe modes)
5. **Context-Aware Transitions**: Verify text cross-attention affects emotion transitions
6. **Cross-Dataset Comparison**: Compare prosodic features across checkpoints

---

## Checkpoints to Benchmark

| Checkpoint | Path | Dataset | Samples | Emotions | Language |
|------------|------|---------|---------|----------|----------|
| RAVDESS | `checkpoints/emotion_lora_ravdess/checkpoint_epoch_10.pt` | RAVDESS | 1,440 | 8 | English |
| CREMA-D | `checkpoints/emotion_lora_cremad/checkpoint_epoch_10.pt` | CREMA-D | 7,442 | 6 | English |
| IESC | `checkpoints/emotion_lora_iesc/checkpoint_epoch_10.pt` | IESC | 600 | 5 | Hindi |

### Dataset-Specific Emotions

| Dataset | Supported Emotions |
|---------|-------------------|
| RAVDESS | neutral, calm, happy, sad, angry, fearful, disgusted, surprised |
| CREMA-D | neutral, happy, sad, angry, fearful, disgusted |
| IESC | neutral, happy, sad, angry, surprised |

---

## v0.3 New Features to Test

### 1. Five New Emotions

| Emotion | VAD Profile | Prosodic Signature |
|---------|-------------|-------------------|
| **sarcastic** | V:-0.2, A:0.3, D:0.4 | High pitch variation, deliberate pacing |
| **bored** | V:-0.3, A:-0.6, D:-0.2 | Flat pitch, low energy, slow rate |
| **affectionate** | V:0.9, A:0.3, D:0.3 | Warm tone, soft variations, slower pace |
| **contemptuous** | V:-0.5, A:0.1, D:0.6 | Controlled pitch, dismissive undertone |
| **awed** | V:0.6, A:0.5, D:-0.3 | Higher pitch, breathy quality, slower pace |

### 2. Nonlinear Intensity Transform

Test that intensity scaling is perceptually accurate:
- `intensity=0.0` → Close to neutral
- `intensity=0.5` → Subtle emotion (not just half)
- `intensity=1.0` → Full emotion
- `intensity=1.5` → Exaggerated emotion

### 3. Emotion Transitions

Test temporal dynamics:
- **Static**: Same emotion throughout (baseline)
- **Transition**: Start emotion → End emotion smoothly
- **Keyframe**: Multiple emotion points in sequence

---

## Benchmark Test Cases

### Test Suite Structure

```
benchmark_v03/
├── ravdess/
│   ├── basic_emotions/          # 8 emotions × standard text
│   ├── intensity_tests/         # 4 intensity levels × 3 emotions
│   ├── nonlinearity_tests/      # Nonlinearity verification (NL-01 to NL-03)
│   ├── new_emotions/            # 5 new v0.3 emotions
│   ├── transition_tests/        # 3 emotion transitions
│   ├── keyframe_tests/          # 3 keyframe mode tests
│   ├── context_tests/           # 3 context-aware transition tests
│   └── results.json
├── cremad/
│   ├── basic_emotions/          # 6 emotions × standard text
│   ├── intensity_tests/
│   ├── new_emotions/
│   ├── transition_tests/
│   └── results.json
├── iesc/
│   ├── basic_emotions/          # 5 emotions (Hindi text)
│   ├── intensity_tests/
│   ├── new_emotions/
│   ├── transition_tests/
│   └── results.json
└── comparison/
    ├── cross_checkpoint/        # Same text, different checkpoints
    └── summary_report.md
```

### Basic Emotion Tests

| Test ID | Text | Emotion | Intensity | Description |
|---------|------|---------|-----------|-------------|
| BE-01 | "This is a test of emotion control." | neutral | 1.0 | Baseline |
| BE-02 | "I am so happy to see you today!" | happy | 1.0 | Positive valence |
| BE-03 | "I feel really sad about this news." | sad | 1.0 | Negative valence |
| BE-04 | "I am absolutely furious right now!" | angry | 1.0 | High arousal negative |
| BE-05 | "I am really scared and worried." | fearful | 1.0 | Fear/anxiety |
| BE-06 | "Wow, I cannot believe this happened!" | surprised | 1.0 | Surprise |
| BE-07 | "I feel calm and peaceful right now." | calm | 1.0 | Low arousal positive |
| BE-08 | "This is absolutely disgusting!" | disgusted | 1.0 | Disgust |
| BE-09 | "I am so excited about this opportunity!" | excited | 1.0 | High arousal positive |

### New Emotion Tests (v0.3)

| Test ID | Text | Emotion | Expected Prosody |
|---------|------|---------|------------------|
| NE-01 | "Oh, how fascinating. Really." | sarcastic | High pitch variation, slow |
| NE-02 | "I suppose this is interesting." | bored | Flat, monotone, low energy |
| NE-03 | "I care about you so much, darling." | affectionate | Warm, soft, gentle pace |
| NE-04 | "How utterly predictable of you." | contemptuous | Controlled, dismissive |
| NE-05 | "This is absolutely magnificent!" | awed | Breathy, wonder, slower |

### Intensity Variation Tests

| Test ID | Emotion | Intensities | Purpose |
|---------|---------|-------------|---------|
| IV-01 | happy | 0.0, 0.5, 1.0, 1.5 | Verify nonlinear scaling |
| IV-02 | angry | 0.0, 0.5, 1.0, 1.5 | High arousal emotion |
| IV-03 | sad | 0.0, 0.5, 1.0, 1.5 | Low arousal emotion |

### Nonlinearity Verification Tests

These tests explicitly verify that the intensity transform is nonlinear (not just linear interpolation).

| Test ID | Emotion | Intensities | Verification Method |
|---------|---------|-------------|---------------------|
| NL-01 | happy | 0.5, 1.0 | Verify 0.5 intensity ≠ 0.5 × 1.0 intensity (prosodic features) |
| NL-02 | sad | 0.5, 1.0 | Verify nonlinearity for low-arousal emotion |
| NL-03 | angry | 0.5, 1.0 | Verify nonlinearity for high-arousal emotion |

**Verification Criteria:**
- Extract prosodic features (pitch_mean, pitch_std, energy_mean) for intensity 0.5 and 1.0
- Compute linear interpolation: `features_0.5_linear = 0.5 × features_1.0`
- Compare actual `features_0.5` with `features_0.5_linear`
- **Pass if**: `||features_0.5 - features_0.5_linear|| > threshold` (threshold = 10% of feature range)
- This ensures the MLP-based `IntensityTransform` is actually being used, not just linear interpolation

### Emotion Transition Tests

| Test ID | Start | End | Text | Purpose |
|---------|-------|-----|------|---------|
| TR-01 | neutral | happy | "I started feeling neutral, then became happy!" | Basic transition |
| TR-02 | sad | hopeful | "Though I was sad, I'm feeling hopeful now." | Negative to positive |
| TR-03 | calm | excited | "Starting calm but getting more excited!" | Energy transition |

### Keyframe Mode Tests

Test multiple emotion keyframes with learned interpolation (v0.3 feature).

| Test ID | Keyframes | Text | Purpose |
|---------|-----------|------|---------|
| KF-01 | neutral@0.0, happy@0.3, sad@1.0 | "I started neutral, got happy, then felt sad." | 3-keyframe transition |
| KF-02 | calm@0.0, excited@0.3, calm@1.0 | "I was calm, got excited, then calmed down." | Return to start emotion |
| KF-03 | sad@0.0, neutral@0.5, happy@1.0 | "I was sad, then neutral, and finally happy." | 3-stage emotional arc |

**Keyframe Format:**
- Each keyframe: `emotion@position` where position is 0.0 (start) to 1.0 (end)
- First keyframe must be at position 0.0
- Last keyframe must be at position 1.0
- Intermediate keyframes can be at any position between 0.0 and 1.0

**Verification:**
- Verify smooth interpolation between keyframes (no jumps)
- Verify endpoint accuracy (first/last frames match keyframe emotions)
- Verify learned interpolation (not just linear between keyframes)

### Context-Aware Transition Tests

Test transitions with text cross-attention (v0.3 feature).

| Test ID | Start | End | Text | Expected Behavior |
|---------|-------|-----|------|-------------------|
| CT-01 | sad | happy | "I was sad, but then I heard the good news!" | Transition accelerates at "good news" |
| CT-02 | calm | excited | "Starting calm but getting more excited as I think about it!" | Gradual transition with context |
| CT-03 | neutral | fearful | "I was fine, but then I saw something scary!" | Transition triggered by "scary" |

**Verification:**
- Compare transition with and without text context
- Verify transition adapts to text content (e.g., accelerates at emotion-relevant words)
- Measure cross-attention contribution (should be ~0.2 weight as per implementation)

### Hindi Tests (IESC Only)

| Test ID | Text (Hindi) | Emotion | Translation |
|---------|--------------|---------|-------------|
| HI-01 | "यह भावना नियंत्रण का परीक्षण है।" | neutral | This is a test of emotion control. |
| HI-02 | "मैं आज आपसे मिलकर बहुत खुश हूं!" | happy | I am so happy to see you today! |
| HI-03 | "मुझे इस खबर से बहुत दुख हुआ।" | sad | I am very sad about this news. |
| HI-04 | "मैं अभी बहुत गुस्से में हूं!" | angry | I am very angry right now! |
| HI-05 | "वाह, यह आश्चर्यजनक है!" | surprised | Wow, this is amazing! |

---

## Running the Benchmark

### Prerequisites

```bash
# Install dependencies
pip install torch torchaudio librosa soundfile numpy
pip install transformers  # For SER evaluation (optional)
pip install matplotlib pandas  # For visualization

# Verify checkpoints exist
ls -la checkpoints/emotion_lora_ravdess/checkpoint_epoch_10.pt
ls -la checkpoints/emotion_lora_cremad/checkpoint_epoch_10.pt
ls -la checkpoints/emotion_lora_iesc/checkpoint_epoch_10.pt
```

### Run Complete Benchmark

```bash
# Benchmark all three checkpoints
python benchmark_v03.py --all

# Or individually:
python benchmark_v03.py --checkpoint ravdess --output benchmark_output/ravdess
python benchmark_v03.py --checkpoint cremad --output benchmark_output/cremad
python benchmark_v03.py --checkpoint iesc --output benchmark_output/iesc

# Generate comparison report
python benchmark_v03.py --compare --output benchmark_output/comparison
```

### Benchmark Script Usage

```bash
python benchmark_v03.py [OPTIONS]

Options:
  --checkpoint CHECKPOINT  Checkpoint to benchmark: ravdess, cremad, iesc, or all
  --output OUTPUT          Output directory for results
  --device DEVICE          Device: auto, cuda, cpu (default: auto)
  --audio_prompt PATH      Reference audio for voice cloning
  --skip-transitions       Skip transition tests (faster)
  --skip-new-emotions      Skip new v0.3 emotion tests
  --generate-report        Generate markdown comparison report
```

---

## Expected Results

### Per-Checkpoint Expectations

#### RAVDESS Checkpoint
- **Best for**: 8 emotions with balanced representation
- **Strengths**: Calm, surprised, fearful (unique to RAVDESS)
- **Expected accuracy**: >70% on supported emotions

#### CREMA-D Checkpoint
- **Best for**: 6 core emotions, largest dataset
- **Strengths**: Happy, sad, angry (most training data)
- **Expected accuracy**: >75% on supported emotions

#### IESC Checkpoint
- **Best for**: Hindi language, 5 emotions
- **Strengths**: Hindi prosody patterns
- **Expected accuracy**: >65% (smaller dataset)

### Prosodic Feature Targets

| Emotion | Pitch Mean (Hz) | Pitch Std (Hz) | Energy (RMS) | Tempo (BPM) |
|---------|-----------------|----------------|--------------|-------------|
| neutral | 150-170 | 20-35 | 0.02-0.03 | 130-150 |
| happy | 180-220 | 40-60 | 0.03-0.04 | 160-180 |
| sad | 130-150 | 15-25 | 0.015-0.02 | 100-130 |
| angry | 180-220 | 45-65 | 0.035-0.05 | 165-190 |
| fearful | 175-210 | 40-55 | 0.025-0.035 | 155-175 |
| calm | 140-160 | 15-25 | 0.015-0.025 | 110-140 |
| surprised | 185-225 | 50-70 | 0.03-0.04 | 145-165 |
| excited | 190-230 | 50-70 | 0.035-0.045 | 170-195 |

### New Emotion Targets (v0.3)

| Emotion | Pitch Mean (Hz) | Pitch Std (Hz) | Energy (RMS) | Tempo (BPM) |
|---------|-----------------|----------------|--------------|-------------|
| sarcastic | 160-180 | 45-65 | 0.025-0.035 | 120-140 |
| bored | 130-150 | 10-20 | 0.015-0.02 | 100-120 |
| affectionate | 165-185 | 25-40 | 0.02-0.03 | 115-135 |
| contemptuous | 155-175 | 20-35 | 0.02-0.03 | 130-150 |
| awed | 170-200 | 35-50 | 0.025-0.035 | 110-135 |

### Transition Test Expectations

- **Smoothness**: Consecutive frames should have small delta (<1.0 norm)
- **Endpoint accuracy**: First/last frames close to start/end emotions
- **Monotonicity**: Features should progress consistently (no jumps)

### Nonlinearity Test Expectations

- **Nonlinearity Verified**: For each emotion, `features_0.5 ≠ 0.5 × features_1.0`
- **Threshold**: Difference should be >10% of feature range
- **MLP Usage**: Confirms `IntensityTransform` MLP is active (not just linear interpolation)

### Keyframe Test Expectations

- **Smoothness**: Interpolation between keyframes should be smooth (<1.0 norm delta)
- **Endpoint Accuracy**: First frame matches first keyframe, last frame matches last keyframe (>0.9 similarity)
- **Learned Interpolation**: Path through emotion space should be learned (not linear between keyframes)
- **Keyframe Accuracy**: Intermediate keyframes should be reached at specified positions (±5% tolerance)

### Context-Aware Test Expectations

- **Context Adaptation**: Transitions should adapt to text content (e.g., accelerate at emotion-relevant words)
- **Cross-Attention Contribution**: Text context should influence transition (~0.2 weight as per implementation)
- **Comparison**: Context-aware transitions should differ from context-free transitions (measured by prosodic feature differences)

---

## Output Structure

### Per-Checkpoint Output

```
benchmark_output/
├── ravdess/
│   ├── audio/
│   │   ├── basic_neutral_1.0.wav
│   │   ├── basic_happy_1.0.wav
│   │   ├── intensity_happy_0.0.wav
│   │   ├── intensity_happy_0.5.wav
│   │   ├── intensity_happy_1.0.wav
│   │   ├── intensity_happy_1.5.wav
│   │   ├── new_sarcastic_1.0.wav
│   │   ├── transition_neutral_happy.wav
│   │   ├── keyframe_neutral_happy_sad.wav
│   │   ├── context_sad_happy_news.wav
│   │   └── ...
│   ├── prosody_analysis.json
│   ├── benchmark_results.json
│   └── prosody_comparison.png
├── cremad/
│   └── ...
├── iesc/
│   └── ...
└── comparison/
    ├── cross_checkpoint_comparison.json
    ├── emotion_accuracy_chart.png
    ├── prosody_heatmap.png
    └── BENCHMARK_RESULTS_V03.md
```

### Results JSON Structure

```json
{
  "timestamp": "2026-01-23T15:30:00",
  "checkpoint": "checkpoints/emotion_lora_ravdess/checkpoint_epoch_10.pt",
  "checkpoint_name": "ravdess",
  "device": "cuda",
  "v03_features": {
    "nonlinear_intensity": true,
    "emotion_trajectory": true,
    "new_emotions": ["sarcastic", "bored", "affectionate", "contemptuous", "awed"]
  },
  "test_results": {
    "basic_emotions": {
      "total": 9,
      "passed": 9,
      "failed": 0,
      "results": [...]
    },
    "intensity_tests": {
      "total": 12,
      "passed": 12,
      "failed": 0,
      "nonlinearity_verified": true,
      "results": [...]
    },
    "new_emotions": {
      "total": 5,
      "passed": 5,
      "failed": 0,
      "results": [...]
    },
    "transition_tests": {
      "total": 3,
      "passed": 3,
      "failed": 0,
      "smoothness_score": 0.92,
      "results": [...]
    },
    "nonlinearity_tests": {
      "total": 3,
      "passed": 3,
      "failed": 0,
      "nonlinearity_verified": true,
      "results": [...]
    },
    "keyframe_tests": {
      "total": 3,
      "passed": 3,
      "failed": 0,
      "smoothness_score": 0.91,
      "endpoint_accuracy": 0.95,
      "results": [...]
    },
    "context_aware_tests": {
      "total": 3,
      "passed": 3,
      "failed": 0,
      "context_adaptation_score": 0.88,
      "results": [...]
    }
  },
  "prosody_analysis": {
    "by_emotion": {
      "happy": {
        "pitch_mean": 195.3,
        "pitch_std": 48.2,
        "energy_mean": 0.032,
        "tempo": 168.5
      },
      ...
    }
  },
  "summary": {
    "total_tests": 38,
    "total_passed": 38,
    "success_rate": 1.0,
    "average_duration": 3.45
  }
}
```

---

## Comparison Metrics

### Cross-Checkpoint Comparison

| Metric | RAVDESS | CREMA-D | IESC | Target |
|--------|---------|---------|------|--------|
| Basic Emotion Pass Rate | - | - | - | >90% |
| Intensity Control | - | - | - | Nonlinear |
| Nonlinearity Verified | - | - | - | Yes (3/3 tests) |
| New Emotion Support | - | - | - | 5/5 |
| Transition Smoothness | - | - | - | >0.85 |
| Keyframe Accuracy | - | - | - | >0.90 |
| Context Adaptation | - | - | - | >0.80 |
| Average Generation Time | - | - | - | <5s |

### Emotion-Specific Comparison

For each emotion, compare:
1. **Pitch profile**: Mean and variation
2. **Energy profile**: Mean and dynamic range
3. **Temporal features**: Tempo, rate
4. **Consistency**: Across multiple generations

---

## Quick Reference

### Run Full Benchmark

```bash
# Complete benchmark with all checkpoints
cd /Users/ntiwari/IITP/attempt4/chatterbox
python benchmark_v03.py --all --output benchmark_output --generate-report
```

### View Results

```bash
# View summary
cat benchmark_output/comparison/BENCHMARK_RESULTS_V03.md

# View per-checkpoint results
cat benchmark_output/ravdess/benchmark_results.json | jq '.summary'
cat benchmark_output/cremad/benchmark_results.json | jq '.summary'
cat benchmark_output/iesc/benchmark_results.json | jq '.summary'
```

### Generate Visualizations

```bash
# Generate comparison charts
python benchmark_v03.py --visualize --output benchmark_output/comparison
```

---

## References

1. **RAVDESS**: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song
2. **CREMA-D**: Cao H, et al. (2014). Crowd-sourced Emotional Multimodal Actors Dataset
3. **IESC**: Indian Emotional Speech Corpus
4. **v0.3 Architecture**: See [EMOTION_ARCH_V03.md](EMOTION_ARCH_V03.md)
5. **v0.3 Implementation**: See [EMOTION_IMPL_V03.md](EMOTION_IMPL_V03.md)
