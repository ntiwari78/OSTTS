# Emotion Recognition Accuracy Improvement Plan

**Target**: Increase SER model recognition accuracy from ~25% to **85%**
**Current State**: 20-44% accuracy across checkpoints
**Document Version**: v1.0
**Date**: 2026-01-24

## Executive Summary

The current TTS emotion system generates audio with emotions that are **not reliably recognized** by external Speech Emotion Recognition (SER) models. This indicates a **domain gap** between TTS-generated speech and human speech patterns that SER models expect.

### Current Performance

| Checkpoint | ehcalabres | Dpngtm | Target |
|------------|-----------|--------|--------|
| RAVDESS | 20.7% | 31.0% | 85% |
| CREMA-D | 33.3% | 25.9% | 85% |
| IESC | 22.2% | 44.4% | 85% |

### Root Cause Analysis

| Issue | Impact | Priority |
|-------|--------|----------|
| **emotion2vec not working** | 0% accuracy from best model | Critical |
| **Weak prosodic variation** | Emotions sound similar | High |
| **No SER feedback in training** | Model doesn't learn recognizable patterns | High |
| **Low confidence scores (~0.13)** | SER models are uncertain | Medium |
| **Domain gap** | TTS voice ≠ human voice patterns | Medium |

---

## Phase 1: Critical Fixes (Week 1)

### 1.1 Fix emotion2vec Integration

**Problem**: emotion2vec shows 0% accuracy - the model output parsing is broken.

**Action Items**:

```python
# File: benchmark_llm_emotions.py
# Fix Emotion2VecEvaluator.classify() method

def classify(self, audio_path: str) -> Dict:
    try:
        res = self.model.generate(
            input=str(audio_path),
            granularity="utterance",
            extract_embedding=False,
            cache={}
        )

        # DEBUG: Print raw output to understand format
        print(f"DEBUG emotion2vec raw output: {res}")

        if res and len(res) > 0:
            result = res[0]

            # Handle different output formats
            if isinstance(result, dict):
                if "labels" in result and "scores" in result:
                    labels = result["labels"]
                    scores = result["scores"]
                    # Find max score index
                    max_idx = scores.index(max(scores))
                    predicted = labels[max_idx]
                    confidence = scores[max_idx]
                elif "label" in result:
                    predicted = result["label"]
                    confidence = result.get("score", 0.0)
                else:
                    # Try direct access
                    predicted = str(list(result.values())[0])
                    confidence = 0.5
            elif isinstance(result, str):
                predicted = result
                confidence = 0.5
            elif isinstance(result, list):
                predicted = str(result[0])
                confidence = 0.5
            else:
                predicted = str(result)
                confidence = 0.5

            return {
                "predicted_raw": predicted,
                "predicted": self.map_emotion(predicted),
                "confidence": float(confidence),
                "status": "success"
            }
```

**Verification**:
```bash
# Test emotion2vec in isolation
python -c "
from funasr import AutoModel
model = AutoModel(model='iic/emotion2vec_plus_base', device='cpu')
res = model.generate(input='benchmark_output/ravdess/audio/basic_happy_1.0.wav', granularity='utterance')
print('Output:', res)
"
```

**Expected Outcome**: emotion2vec accuracy should jump to 40-60% after fix.

---

### 1.2 Enable SER Integration Loss During Training

**Problem**: `SERIntegrationLoss` exists in `emotion_losses.py` but is NOT used during training.

**Action Items**:

```python
# File: train_emotion_lora.py
# Enable SER loss in training

from chatterbox.models.t3.modules.emotion_losses import CombinedEmotionLoss

# In training initialization
combined_loss = CombinedEmotionLoss(
    emotion_dim=64,
    audio_feature_dim=1024,
    consistency_weight=0.5,
    ser_weight=0.3,          # ENABLE SER LOSS
    use_ser=True,            # ENABLE SER MODEL
    use_discriminator=False,
)

# In training loop
loss_result = combined_loss(
    tts_loss=tts_loss,
    emotion_embed=emotion_embed,
    audio_features=audio_features,
    audio=generated_audio,           # Pass generated audio
    target_emotions=[emotion_name],  # Pass emotion names
)

total_loss = loss_result["total_loss"]
```

**New Training Command**:
```bash
python train_emotion_lora.py \
    --dataset ravdess \
    --use_ser_loss \
    --ser_weight 0.3 \
    --epochs 15 \
    --output_dir checkpoints/emotion_lora_ravdess_ser
```

**Expected Outcome**: Model learns to generate audio that SER models recognize.

---

## Phase 2: Prosodic Enhancement (Week 2)

### 2.1 Increase Prosodic Parameter Ranges

**Problem**: Current VAD parameters produce subtle differences. SER models need exaggerated prosody.

**Current vs Proposed Parameters**:

```python
# File: emotion_embeddings.py
# Increase prosodic expressiveness

EMOTION_TYPES = {
    "happy": {
        "valence": 0.8,
        "arousal": 0.7,       # Was 0.6 -> increase
        "dominance": 0.6,
        "pitch_mean": 0.4,    # Was 0.3 -> increase
        "pitch_var": 0.5,     # Was 0.4 -> increase
        "energy": 0.5,        # Was 0.3 -> increase
        "speaking_rate": 0.3, # Was 0.2 -> increase
    },
    "sad": {
        "valence": -0.7,
        "arousal": -0.5,      # Was -0.4 -> more negative
        "dominance": -0.4,
        "pitch_mean": -0.4,   # Was -0.3 -> lower
        "pitch_var": -0.4,    # Was -0.3 -> flatter
        "energy": -0.5,       # Was -0.3 -> lower energy
        "speaking_rate": -0.4, # Was -0.3 -> slower
    },
    "angry": {
        "valence": -0.6,
        "arousal": 0.9,       # Was 0.8 -> maximum arousal
        "dominance": 0.8,
        "pitch_mean": 0.3,
        "pitch_var": 0.5,     # Was 0.4 -> more variable
        "energy": 0.8,        # Was 0.6 -> high energy
        "speaking_rate": 0.2,
    },
    # ... similar adjustments for other emotions
}
```

**Prosodic Scaling Multiplier**:

```python
# Add global expressiveness multiplier
class EmotionEmbeddings(nn.Module):
    def __init__(self, expressiveness_scale: float = 1.5):
        self.expressiveness_scale = expressiveness_scale

    def get_emotion_embedding(self, emotion_name, intensity=1.0):
        embed = self._get_base_embedding(emotion_name)
        # Scale prosodic features
        embed = embed * self.expressiveness_scale * intensity
        return embed
```

---

### 2.2 Add Prosodic Feature Predictor

**New Module**: Explicitly predict prosodic features from emotion embedding.

```python
# File: emotion_prosody.py (NEW)

class EmotionProsodyPredictor(nn.Module):
    """
    Predicts explicit prosodic features from emotion embedding.
    These features guide the acoustic model.
    """

    def __init__(self, emotion_dim=64, hidden_dim=128):
        super().__init__()
        self.prosody_head = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),  # pitch_mean, pitch_var, energy, tempo, duration
        )

        # Target ranges for each prosodic feature
        self.feature_ranges = {
            "pitch_mean": (80, 400),    # Hz
            "pitch_var": (0, 100),      # Hz std
            "energy": (0.01, 0.15),     # RMS
            "tempo": (0.5, 2.0),        # relative
            "duration": (0.5, 2.0),     # relative
        }

    def forward(self, emotion_embed):
        """Predict prosodic features."""
        raw = self.prosody_head(emotion_embed)

        # Scale to target ranges
        features = {}
        for i, (name, (min_v, max_v)) in enumerate(self.feature_ranges.items()):
            features[name] = torch.sigmoid(raw[:, i]) * (max_v - min_v) + min_v

        return features

class ProsodyMatchingLoss(nn.Module):
    """
    Loss that compares predicted prosody with actual audio prosody.
    """

    def forward(self, predicted_prosody, audio, sr=24000):
        # Extract actual prosody from audio
        actual_prosody = self.extract_prosody(audio, sr)

        loss = 0
        for key in predicted_prosody:
            loss += F.mse_loss(predicted_prosody[key], actual_prosody[key])

        return loss
```

---

## Phase 3: Training Improvements (Week 3)

### 3.1 SER-Guided Data Filtering

**Concept**: Only train on samples where SER models agree with the label.

```python
# File: ser_data_filter.py (NEW)

class SERGuidedDataFilter:
    """
    Filter training data to only include samples where
    SER models correctly recognize the emotion.
    """

    def __init__(self):
        self.evaluators = {
            "emotion2vec": Emotion2VecEvaluator(),
            "dpngtm": DpngtmEvaluator(),
        }
        for e in self.evaluators.values():
            e.load_model()

    def filter_dataset(self, audio_files, emotions):
        """
        Filter to samples where at least one SER model agrees.

        Returns:
            filtered_files: List of files where SER agrees
            agreement_scores: Dict with per-sample agreement
        """
        filtered = []

        for audio_file, expected_emotion in zip(audio_files, emotions):
            votes = 0
            for evaluator in self.evaluators.values():
                result = evaluator.classify(audio_file)
                if result["predicted"].lower() == expected_emotion.lower():
                    votes += 1

            # Keep if at least one model agrees
            if votes >= 1:
                filtered.append((audio_file, expected_emotion))

        return filtered

# Usage in training
filter = SERGuidedDataFilter()
filtered_data = filter.filter_dataset(train_files, train_emotions)
print(f"Kept {len(filtered_data)}/{len(train_files)} samples")
```

---

### 3.2 Adversarial SER Training

**Concept**: Train generator to fool SER model into correct prediction.

```python
# File: adversarial_ser_training.py (NEW)

class AdversarialSERTrainer:
    """
    Adversarial training: Generator tries to produce audio
    that SER model classifies as target emotion.
    """

    def __init__(self, tts_model, ser_model, ser_processor):
        self.tts = tts_model
        self.ser = ser_model
        self.ser_processor = ser_processor

        # Freeze SER model
        for p in self.ser.parameters():
            p.requires_grad = False

    def compute_ser_loss(self, audio, target_emotion_idx):
        """
        Compute cross-entropy loss for SER prediction.

        Lower loss = SER predicts target emotion correctly.
        """
        # Resample if needed
        inputs = self.ser_processor(
            audio.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.ser(**inputs)
            logits = outputs.logits

        # We want to maximize probability of target emotion
        loss = F.cross_entropy(logits, target_emotion_idx)

        return loss

    def train_step(self, text, emotion_name, emotion_idx):
        # Generate audio
        audio = self.tts.generate(text, emotion=emotion_name)

        # Compute TTS loss (if applicable)
        tts_loss = self.tts.compute_loss()

        # Compute SER adversarial loss
        ser_loss = self.compute_ser_loss(audio, emotion_idx)

        # Combined loss
        total_loss = tts_loss + 0.3 * ser_loss

        return total_loss
```

---

### 3.3 Emotion Contrastive Pre-training

**Concept**: Pre-train emotion embeddings to be maximally separable.

```python
# File: emotion_contrastive_pretrain.py (NEW)

class EmotionContrastivePretrainer:
    """
    Pre-train emotion embeddings using contrastive learning
    on real emotional speech datasets.
    """

    def __init__(self, emotion_embeddings, ser_model):
        self.embeddings = emotion_embeddings
        self.ser = ser_model

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def contrastive_loss(self, embeddings, labels, temperature=0.07):
        """
        InfoNCE contrastive loss.

        Same emotion = positive pair
        Different emotion = negative pair
        """
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

        # Create label mask (1 for same emotion, 0 for different)
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.shape[0], device=mask.device)

        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Average over positive pairs
        loss = -(mask * log_prob).sum() / mask.sum()

        return loss

    def pretrain(self, real_audio_dataset, epochs=10):
        """
        Pre-train on real emotional speech.
        """
        optimizer = torch.optim.Adam(
            list(self.embeddings.parameters()) +
            list(self.projection.parameters()),
            lr=1e-4
        )

        for epoch in range(epochs):
            for batch in real_audio_dataset:
                audio, emotion_labels = batch

                # Get emotion embeddings
                embeds = []
                for label in emotion_labels:
                    embeds.append(self.embeddings.get_emotion_embedding(label))
                embeds = torch.cat(embeds, dim=0)

                # Project
                projected = self.projection(embeds)

                # Contrastive loss
                loss = self.contrastive_loss(
                    projected,
                    torch.tensor([self.label_to_idx[l] for l in emotion_labels])
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## Phase 4: Model Architecture Changes (Week 4)

### 4.1 Emotion-Conditioned Vocoder Fine-tuning

**Problem**: The vocoder may not preserve emotional prosody.

**Solution**: Fine-tune vocoder on emotional speech.

```python
# File: finetune_vocoder_emotion.py (NEW)

def finetune_vocoder_on_emotions(
    vocoder,
    emotion_audio_pairs,  # [(mel, audio, emotion), ...]
    epochs=5
):
    """
    Fine-tune vocoder to better preserve emotional characteristics.
    """
    optimizer = torch.optim.Adam(vocoder.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for mel, target_audio, emotion in emotion_audio_pairs:
            # Generate audio from mel
            generated_audio = vocoder(mel)

            # Standard vocoder loss
            recon_loss = F.l1_loss(generated_audio, target_audio)

            # Prosody preservation loss
            target_prosody = extract_prosody(target_audio)
            gen_prosody = extract_prosody(generated_audio)
            prosody_loss = prosody_mse(target_prosody, gen_prosody)

            # Combined loss
            loss = recon_loss + 0.5 * prosody_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

### 4.2 Emotion Intensity Calibration

**Problem**: Intensity 1.0 may not produce recognizable emotions.

**Solution**: Calibrate intensity based on SER recognition.

```python
# File: emotion_intensity_calibrator.py (NEW)

class EmotionIntensityCalibrator:
    """
    Find optimal intensity for each emotion based on SER recognition.
    """

    def __init__(self, tts_model, ser_evaluator):
        self.tts = tts_model
        self.ser = ser_evaluator

    def calibrate(self, text, emotion, intensity_range=(0.5, 2.0), steps=10):
        """
        Find intensity that maximizes SER recognition.
        """
        best_intensity = 1.0
        best_confidence = 0.0

        intensities = np.linspace(intensity_range[0], intensity_range[1], steps)

        for intensity in intensities:
            # Generate with this intensity
            audio = self.tts.generate(text, emotion=emotion, intensity=intensity)

            # Evaluate with SER
            result = self.ser.classify(audio)

            if result["predicted"] == emotion and result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_intensity = intensity

        return best_intensity, best_confidence

    def calibrate_all_emotions(self, test_texts):
        """
        Create calibration map for all emotions.
        """
        calibration = {}

        for emotion in EMOTION_TYPES:
            intensities = []
            for text in test_texts:
                best_int, conf = self.calibrate(text, emotion)
                intensities.append(best_int)

            calibration[emotion] = {
                "recommended_intensity": np.mean(intensities),
                "intensity_std": np.std(intensities),
            }

        return calibration
```

---

## Phase 5: Evaluation & Iteration (Week 5)

### 5.1 Multi-Model Ensemble Evaluation

```python
# File: ensemble_evaluator.py (NEW)

class EnsembleEmotionEvaluator:
    """
    Ensemble of multiple SER models for robust evaluation.
    """

    def __init__(self):
        self.evaluators = {
            "emotion2vec": Emotion2VecEvaluator("base"),
            "dpngtm": DpngtmEvaluator(),
            "ehcalabres": Wav2Vec2EhcalabresEvaluator(),
        }

        # Weights based on model accuracy
        self.weights = {
            "emotion2vec": 0.5,  # Best model, highest weight
            "dpngtm": 0.3,
            "ehcalabres": 0.2,
        }

    def classify(self, audio_path):
        """
        Weighted voting across models.
        """
        votes = defaultdict(float)

        for name, evaluator in self.evaluators.items():
            result = evaluator.classify(audio_path)
            if result["status"] == "success":
                emotion = result["predicted"]
                confidence = result["confidence"]
                votes[emotion] += self.weights[name] * confidence

        if not votes:
            return {"status": "error", "error": "All models failed"}

        # Return emotion with highest weighted vote
        best_emotion = max(votes, key=votes.get)

        return {
            "predicted": best_emotion,
            "confidence": votes[best_emotion],
            "all_votes": dict(votes),
            "status": "success"
        }
```

---

### 5.2 Success Metrics

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Target |
|--------|---------|---------|---------|---------|---------|--------|
| emotion2vec | 0% | 50% | 60% | 70% | 80% | 85% |
| Dpngtm | 31% | 45% | 55% | 65% | 75% | 85% |
| ehcalabres | 21% | 35% | 45% | 55% | 65% | 75% |
| Ensemble | N/A | 50% | 60% | 72% | 82% | 85% |

---

## Implementation Priority

### Week 1 (Critical) - **COMPLETED**
1. ✅ Fix emotion2vec output parsing - Added robust multi-format parsing with debug mode
2. ✅ Enable SER loss in training - Added `--use_ser_loss` flag to train_emotion_lora.py
3. ⏳ Retrain with SER feedback - Ready to run with new training command

### Week 2 (High Priority) - **COMPLETED**
4. ✅ Increase prosodic parameter ranges - Enhanced VAD and prosodic values in emotion_embeddings.py
5. ✅ Add prosody prediction head - Created emotion_prosody.py with EmotionProsodyPredictor
6. ✅ Calibrate intensity per emotion - Created emotion_intensity_calibrator.py

### Week 3 (Medium Priority)
7. Implement SER-guided data filtering
8. Adversarial SER training
9. Contrastive pre-training

### Week 4 (Enhancement)
10. Vocoder fine-tuning
11. Architecture adjustments
12. Multi-model ensemble

### Week 5 (Validation)
13. Full benchmark on all checkpoints
14. Human listening tests
15. Documentation update

---

## New Training Pipeline

```bash
# Step 1: Pre-train emotion embeddings contrastively (optional, Phase 3)
python emotion_contrastive_pretrain.py \
    --dataset ravdess \
    --epochs 10

# Step 2: Train with SER feedback and prosody loss (Phase 1 + Phase 2)
python train_emotion_lora.py \
    --dataset ravdess \
    --use_ser_loss \
    --ser_weight 0.3 \
    --use_prosody_loss \
    --prosody_weight 0.2 \
    --expressiveness_scale 1.5 \
    --epochs 20 \
    --output_dir checkpoints/emotion_lora_ravdess_v04

# Step 3: Calibrate intensity
python -c "
from chatterbox.models.t3.modules.emotion_intensity_calibrator import IntensityCalibrator
# Load models and calibrate - see module documentation
"

# Step 4: Benchmark
python benchmark_llm_emotions.py \
    --checkpoint ravdess \
    --output benchmark_output/comparison/BENCHMARK_LLM_RESULTS_V04.md
```

### Quick Training Commands (Phase 2)

```bash
# RAVDESS with SER + Prosody loss
python train_emotion_lora.py \
    --dataset ravdess \
    --use_ser_loss --ser_weight 0.3 \
    --use_prosody_loss --prosody_weight 0.2 \
    --expressiveness_scale 1.3 \
    --epochs 15 \
    --output_dir checkpoints/emotion_lora_ravdess_phase2

# IESC (Hindi) with SER + Prosody loss
python train_emotion_lora.py \
    --dataset iesc \
    --use_ser_loss --ser_weight 0.3 \
    --use_prosody_loss --prosody_weight 0.2 \
    --expressiveness_scale 1.3 \
    --epochs 15 \
    --output_dir checkpoints/emotion_lora_iesc_phase2
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SER loss hurts TTS quality | Balance SER weight (start low: 0.1) |
| Over-exaggerated emotions | Add intensity calibration |
| Domain gap persists | Fine-tune SER on TTS audio |
| Training instability | Gradual loss weight increase |
| Model overfits to SER | Use multiple SER models |

---

## Expected Outcomes

After implementing all phases:

1. **emotion2vec**: 80-85% accuracy (from 0%)
2. **Dpngtm**: 80-85% accuracy (from 31%)
3. **ehcalabres**: 70-75% accuracy (from 21%)
4. **Ensemble**: 85%+ accuracy

The generated emotions will be:
- Recognizable by external SER models
- Prosodically distinct and expressive
- Natural sounding (not over-exaggerated)
- Consistent across different text inputs
