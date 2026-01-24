# Emotion Recognition Models Implementation v0.1

**Document Version**: v0.1
**Script**: `benchmark_llm_emotions.py`
**Date**: 2026-01-24

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Class Hierarchy](#class-hierarchy)
4. [Model Integrations](#model-integrations)
5. [Emotion Mapping Strategy](#emotion-mapping-strategy)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Output Formats](#output-formats)
8. [Usage Examples](#usage-examples)
9. [Extending the System](#extending-the-system)

---

## Overview

The `benchmark_llm_emotions.py` script evaluates TTS-generated emotional speech using multiple Speech Emotion Recognition (SER) models. It provides an objective assessment of whether the generated emotions are recognizable by state-of-the-art emotion recognition systems.

### Purpose

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  TTS Generated      │     │  Emotion Recognition │     │  Benchmark Report   │
│  Audio Files        │ --> │  Models (3 models)   │ --> │  (Accuracy, etc.)   │
│  (benchmark_output) │     │                      │     │  (Markdown + JSON)  │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

### Key Features

- **Multi-model evaluation**: Uses 3 different SER models for consensus
- **Fuzzy matching**: Handles model-specific emotion vocabularies
- **Per-checkpoint analysis**: Separate evaluation for RAVDESS, CREMA-D, IESC
- **Detailed reporting**: Markdown and JSON output formats

---

## Architecture

### High-Level Flow

```
                                    ┌─────────────────────────┐
                                    │      main()             │
                                    │  - Parse arguments      │
                                    │  - Initialize models    │
                                    │  - Run evaluation       │
                                    └───────────┬─────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
        │ Emotion2VecEval   │       │ Wav2Vec2Ehcalabres│       │ DpngtmEvaluator   │
        │ (Primary Model)   │       │ (Secondary Model) │       │ (Tertiary Model)  │
        └─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
                  │                           │                           │
                  └───────────────────────────┼───────────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │    evaluate_checkpoint()      │
                              │  - Load audio files           │
                              │  - Extract expected emotions  │
                              │  - Classify with each model   │
                              │  - Compute accuracy           │
                              └───────────────┬───────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │  generate_markdown_report()   │
                              │  - Summary tables             │
                              │  - Per-emotion breakdown      │
                              │  - Sample predictions         │
                              └───────────────────────────────┘
```

### Module Structure

```
benchmark_llm_emotions.py
│
├── Configuration Constants
│   ├── CHECKPOINT_CONFIGS      # Audio directories per checkpoint
│   ├── EMOTION_PATTERNS        # Filename -> expected emotion mapping
│   ├── EMOTION2VEC_MAPPING     # emotion2vec output normalization
│   ├── WAV2VEC2_EHCALABRES_MAPPING
│   └── DPNGTM_MAPPING
│
├── Base Class
│   └── EmotionEvaluator        # Abstract base for all evaluators
│
├── Model Evaluators
│   ├── Emotion2VecEvaluator    # emotion2vec+ (funasr)
│   ├── Wav2Vec2EhcalabresEvaluator  # ehcalabres model
│   └── DpngtmEvaluator         # Dpngtm model
│
├── Helper Functions
│   ├── get_expected_emotion()  # Extract emotion from filename
│   └── is_compatible_emotion() # Check model support
│
├── Core Functions
│   ├── evaluate_checkpoint()   # Main evaluation loop
│   └── generate_markdown_report()  # Report generation
│
└── Entry Point
    └── main()                  # CLI handling
```

---

## Class Hierarchy

### Base Class: EmotionEvaluator

```python
class EmotionEvaluator:
    """Abstract base class for emotion evaluators."""

    def __init__(self, device: str = "auto"):
        # Device selection: cuda > mps > cpu
        # Note: MPS disabled due to compatibility issues

    def load_model(self) -> bool:
        """Load the model. Override in subclasses."""
        raise NotImplementedError

    def classify(self, audio_path: str) -> Dict:
        """Classify emotion from audio file."""
        # Returns: {
        #   "predicted_raw": str,  # Model's raw output
        #   "predicted": str,      # Normalized emotion
        #   "confidence": float,   # Confidence score
        #   "status": "success" | "error"
        # }
        raise NotImplementedError

    def map_emotion(self, predicted: str) -> str:
        """Map model output to our emotion system."""
        return predicted.lower()
```

### Derived Classes

```
EmotionEvaluator (Abstract)
        │
        ├── Emotion2VecEvaluator
        │   ├── Uses: funasr.AutoModel
        │   ├── Model: iic/emotion2vec_plus_base
        │   ├── Emotions: 9 (Angry, Happy, Sad, Neutral, Fearful,
        │   │              Disgusted, Surprised, Other, Unknown)
        │   └── Features: Multilingual, foundation model
        │
        ├── Wav2Vec2EhcalabresEvaluator
        │   ├── Uses: transformers.AutoModelForAudioClassification
        │   ├── Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
        │   ├── Emotions: 7 (angry, calm, disgust, fear, happy, neutral, sad)
        │   └── Features: Well-established, ~75% accuracy
        │
        └── DpngtmEvaluator
            ├── Uses: transformers.Wav2Vec2ForSequenceClassification
            ├── Model: Dpngtm/wav2vec2-emotion-recognition
            ├── Emotions: 8 (angry, calm, disgust, fearful, happy,
            │              neutral, sad, surprised)
            └── Features: Trained on RAVDESS/CREMA-D, ~80% accuracy
```

---

## Model Integrations

### 1. emotion2vec+ (Primary Model)

**Why Primary**: Best accuracy, multilingual support, foundation model architecture.

```python
class Emotion2VecEvaluator(EmotionEvaluator):
    def load_model(self):
        from funasr import AutoModel
        self.model = AutoModel(
            model="iic/emotion2vec_plus_base",
            model_revision="v2.0.4",
            device=self.device
        )

    def classify(self, audio_path: str) -> Dict:
        res = self.model.generate(
            input=str(audio_path),
            granularity="utterance",  # Whole-utterance classification
            extract_embedding=False,   # Classification, not embeddings
            cache={}
        )
        # Parse response format:
        # [{"labels": ["Happy", "Sad", ...], "scores": [0.8, 0.1, ...]}]
```

**Output Format**:
```python
{
    "labels": ["Happy", "Neutral", "Sad", ...],
    "scores": [0.85, 0.10, 0.03, ...]
}
```

### 2. wav2vec2 ehcalabres (Secondary Model)

**Why Secondary**: Well-established baseline, good for English.

```python
class Wav2Vec2EhcalabresEvaluator(EmotionEvaluator):
    def load_model(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)

    def classify(self, audio_path: str) -> Dict:
        # 1. Load and resample audio to 16kHz
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # 2. Process through feature extractor
        inputs = self.processor(waveform.squeeze().numpy(),
                                sampling_rate=16000,
                                return_tensors="pt")

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            pred_label = self.model.config.id2label[pred_id]
```

### 3. wav2vec2 Dpngtm (Tertiary Model)

**Why Tertiary**: 8 emotions, trained on same datasets as our TTS.

```python
class DpngtmEvaluator(EmotionEvaluator):
    def load_model(self):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

        model_id = "Dpngtm/wav2vec2-emotion-recognition"
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
```

---

## Emotion Mapping Strategy

### The Challenge

Each model has a different emotion vocabulary:

| Our System | emotion2vec | ehcalabres | Dpngtm |
|------------|-------------|------------|--------|
| happy | Happy | happy | happy |
| sad | Sad | sad | sad |
| angry | Angry | angry | angry |
| fearful | Fearful | fear | fearful |
| disgusted | Disgusted | disgust | disgust |
| surprised | Surprised | - | surprised |
| calm | - | calm | calm |
| neutral | Neutral | neutral | neutral |
| excited | - | - | - |
| sarcastic | Other | - | - |
| bored | Other | - | - |
| affectionate | Other | - | - |
| contemptuous | Other | - | - |
| awed | Other | - | - |

### Solution: Multi-Level Mapping

```python
# Level 1: Direct mapping (model output -> our system)
EMOTION2VEC_MAPPING = {
    "Angry": "angry",
    "Happy": "happy",
    "Sad": "sad",
    "Neutral": "neutral",
    "Fearful": "fearful",
    "Disgusted": "disgusted",
    "Surprised": "surprised",
    "Other": "neutral",      # Fallback
    "Unknown": "neutral",    # Fallback
}

# Level 2: Fuzzy matching for evaluation
close_matches = {
    "excited": ["happy"],           # excited -> accept happy
    "sarcastic": ["angry", "neutral", "other"],
    "bored": ["neutral", "sad", "other"],
    "affectionate": ["happy", "neutral", "other"],
    "contemptuous": ["angry", "disgusted", "other"],
    "awed": ["surprised", "happy", "other"],
    "fearful": ["fear"],            # Vocabulary difference
    "disgusted": ["disgust"],       # Vocabulary difference
}
```

### Matching Logic

```python
def is_correct(expected, predicted):
    # 1. Exact match
    if predicted.lower() == expected.lower():
        return True

    # 2. Fuzzy match for unsupported emotions
    if expected.lower() in close_matches:
        if predicted.lower() in close_matches[expected.lower()]:
            return True

    return False
```

---

## Evaluation Pipeline

### Step 1: Extract Expected Emotion from Filename

```python
EMOTION_PATTERNS = {
    "basic_neutral": "neutral",
    "basic_happy": "happy",
    "intensity_happy": "happy",       # Intensity tests -> base emotion
    "transition_sad_happy": "happy",  # Transitions -> end emotion
    "new_sarcastic": "sarcastic",
    # ...
}

def get_expected_emotion(filename: str) -> Optional[str]:
    for pattern, emotion in EMOTION_PATTERNS.items():
        if pattern in filename:
            return emotion
    return None
```

### Step 2: Classify with Each Model

```python
def evaluate_checkpoint(checkpoint_name, evaluators, output_dir):
    audio_files = list(audio_dir.glob("*.wav"))

    for model_name, evaluator in evaluators.items():
        for audio_file in audio_files:
            expected = get_expected_emotion(audio_file.name)
            result = evaluator.classify(str(audio_file))

            if result["status"] == "success":
                predicted = result["predicted"]
                is_correct = check_match(expected, predicted)
                # Update statistics...
```

### Step 3: Compute Accuracy

```python
model_stats = {
    "total": 0,
    "correct": 0,
    "by_emotion": {
        "happy": {"total": 5, "correct": 4},
        "sad": {"total": 5, "correct": 5},
        # ...
    },
    "predictions": [
        {"file": "basic_happy_1.0.wav", "expected": "happy",
         "predicted": "happy", "confidence": 0.92, "correct": True},
        # ...
    ]
}

accuracy = model_stats["correct"] / model_stats["total"]
```

---

## Output Formats

### Markdown Report (BENCHMARK_LLM_RESULTS_V03.md)

```markdown
# Emotion Recognition Benchmark Results (LLM-based)

**Generated**: 2026-01-24 12:00:00

## Summary

| Checkpoint | Dataset | emotion2vec | wav2vec2 (ehcalabres) | wav2vec2 (Dpngtm) |
|------------|---------|-------------|----------------------|-------------------|
| RAVDESS | RAVDESS | 75.0% | 60.0% | 70.0% |
| CREMAD | CREMA-D | 80.0% | 65.0% | 75.0% |
| IESC | IESC | 70.0% | 55.0% | 65.0% |

## Detailed Results

### RAVDESS

#### emotion2vec_base

**Overall Accuracy**: 75.0% (22/29)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| happy | 4 | 5 | 80.0% |
| sad | 5 | 5 | 100.0% |
...
```

### JSON Report (BENCHMARK_LLM_RESULTS_V03.json)

```json
{
  "ravdess": {
    "checkpoint": "ravdess",
    "dataset": "RAVDESS",
    "language": "en",
    "timestamp": "2026-01-24T12:00:00",
    "model_results": {
      "emotion2vec_base": {
        "total": 29,
        "correct": 22,
        "accuracy": 0.758,
        "by_emotion": {
          "happy": {"total": 5, "correct": 4},
          "sad": {"total": 5, "correct": 5}
        },
        "predictions": [
          {
            "file": "basic_happy_1.0.wav",
            "expected": "happy",
            "predicted": "happy",
            "predicted_raw": "Happy",
            "confidence": 0.92,
            "correct": true
          }
        ]
      }
    }
  }
}
```

---

## Usage Examples

### Basic Usage

```bash
# Run all models on all checkpoints (default)
python benchmark_llm_emotions.py

# Equivalent to:
python benchmark_llm_emotions.py --checkpoint all --models all
```

### Specific Checkpoint

```bash
# Only evaluate RAVDESS checkpoint
python benchmark_llm_emotions.py --checkpoint ravdess

# Only evaluate CREMA-D
python benchmark_llm_emotions.py --checkpoint cremad
```

### Specific Models

```bash
# Only use emotion2vec (fastest single model)
python benchmark_llm_emotions.py --models emotion2vec

# Use emotion2vec and Dpngtm
python benchmark_llm_emotions.py --models emotion2vec dpngtm
```

### Custom Output

```bash
# Custom output path
python benchmark_llm_emotions.py --output results/my_benchmark.md

# Force CPU
python benchmark_llm_emotions.py --device cpu
```

---

## Extending the System

### Adding a New Model

1. **Create a new evaluator class**:

```python
class NewModelEvaluator(EmotionEvaluator):
    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.model_name = "new_model"

    def load_model(self):
        # Load your model
        self.model = load_model(...)
        return True

    def classify(self, audio_path: str) -> Dict:
        # Implement classification
        predicted = self.model.predict(audio_path)
        return {
            "predicted_raw": predicted,
            "predicted": self.map_emotion(predicted),
            "confidence": 0.9,
            "status": "success"
        }

    def map_emotion(self, predicted: str) -> str:
        mapping = {"model_emotion": "our_emotion", ...}
        return mapping.get(predicted, predicted.lower())
```

2. **Add mapping constant**:

```python
NEW_MODEL_MAPPING = {
    "model_happy": "happy",
    "model_sad": "sad",
    # ...
}
```

3. **Register in main()**:

```python
if use_all or "new_model" in args.models:
    nm = NewModelEvaluator(device=args.device)
    if nm.load_model():
        evaluators["new_model"] = nm
```

4. **Update argparse choices**:

```python
parser.add_argument(
    "--models",
    choices=["emotion2vec", "ehcalabres", "dpngtm", "new_model", "all"],
    # ...
)
```

### Adding New Emotions

1. **Update EMOTION_PATTERNS**:

```python
EMOTION_PATTERNS = {
    # ...
    "new_my_emotion": "my_emotion",
}
```

2. **Update close_matches**:

```python
close_matches = {
    # ...
    "my_emotion": ["happy", "neutral", "other"],  # Acceptable alternatives
}
```

---

## Dependencies

```bash
# Core
pip install torch torchaudio

# emotion2vec (primary model)
pip install funasr modelscope

# wav2vec2 models
pip install transformers
```

---

## References

1. **emotion2vec**: https://github.com/ddlBoJack/emotion2vec
2. **wav2vec2 ehcalabres**: https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
3. **wav2vec2 Dpngtm**: https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition
4. **EMOTION_RECOGNITION_MODELS.md**: Model selection rationale
