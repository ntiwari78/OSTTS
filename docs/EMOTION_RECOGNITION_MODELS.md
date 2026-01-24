# Open Source Models for Audio Emotion Recognition

This document lists open-source models that can predict emotions from audio input for benchmarking your TTS emotion system.

## Table of Contents

1. [Recommended Models](#recommended-models)
2. [Traditional SER Models](#traditional-ser-models)
3. [Foundation Models (emotion2vec)](#foundation-models-emotion2vec)
4. [Multimodal LLMs](#multimodal-llms)
5. [Model Comparison](#model-comparison)
6. [Implementation Examples](#implementation-examples)

---

## Recommended Models

### Top 3 Recommendations for Benchmarking

1. **emotion2vec+ (base or large)** - Best overall accuracy, universal model
2. **wav2vec2-emotion (ehcalabres)** - Good balance, already in your benchmark
3. **SpeechBrain emotion-recognition-wav2vec2-IEMOCAP** - Reliable, well-maintained

---

## Traditional SER Models

### 1. wav2vec2-lg-xlsr-en-speech-emotion-recognition

**Model ID**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

**Status**: ✅ Already in your BENCHMARK.md

**Emotions**: 7 classes
- angry, calm, disgust, fear, happy, neutral, sad

**Accuracy**: ~70-75% on standard benchmarks

**Usage**:
```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)
```

**Pros**:
- Well-established, widely used
- Good English performance
- Easy to integrate

**Cons**:
- Limited to 7 emotions (missing surprised, disgusted)
- May not handle Hindi well

---

### 2. SpeechBrain emotion-recognition-wav2vec2-IEMOCAP

**Model ID**: `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`

**Emotions**: 4 classes (IEMOCAP dataset)
- happy, sad, angry, neutral

**Accuracy**: 78.7% on IEMOCAP test set

**Usage**:
```python
from speechbrain.inference.classifiers import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP"
)

# Classify emotion
out_prob, score, index, text_lab = classifier.classify_file("audio.wav")
```

**Pros**:
- High accuracy on IEMOCAP
- Automated audio normalization
- Well-maintained by SpeechBrain

**Cons**:
- Only 4 emotion classes
- Trained on English only

---

### 3. Dpngtm/wav2vec2-emotion-recognition

**Model ID**: `Dpngtm/wav2vec2-emotion-recognition`

**Emotions**: 8 classes
- angry, calm, disgust, fearful, happy, neutral, sad, surprised

**Accuracy**: 79.57% accuracy

**Training Data**: TESS, CREMA-D, SAVEE, RAVDESS

**Usage**:
```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio

model_id = "Dpngtm/wav2vec2-emotion-recognition"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

# Process audio
audio, sr = torchaudio.load("audio.wav")
inputs = processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax(-1).item()
```

**Pros**:
- 8 emotions (matches your system better)
- Good accuracy
- Trained on multiple datasets

**Cons**:
- English only

---

### 4. AventIQ-AI/wav2vec2-base_speech_emotion_recognition

**Model ID**: `AventIQ-AI/wav2vec2-base_speech_emotion_recognition`

**Emotions**: 8 classes (RAVDESS)
- angry, calm, disgust, fearful, happy, neutral, sad, surprised

**Accuracy**: ~65%

**Pros**:
- Trained on RAVDESS (matches your checkpoint)
- 8 emotions

**Cons**:
- Lower accuracy than alternatives

---

## Foundation Models (emotion2vec)

### emotion2vec+ Series (Recommended)

**Status**: ✅ **Best choice for multilingual and high accuracy**

**Available Models**:
- `emotion2vec/emotion2vec_plus_seed` - 201 hours training
- `emotion2vec/emotion2vec_plus_base` - 4,788 hours training (~90M params)
- `emotion2vec/emotion2vec_plus_large` - 42,526 hours training (~300M params)

**Emotions**: 9 classes
- Angry, Disgusted, Fearful, Happy, Neutral, Other, Sad, Surprised, Unknown

**Key Features**:
- Universal model (works across languages and environments)
- Self-supervised pre-training
- Can extract embeddings or classify
- Frame-level or utterance-level features

**Installation**:
```bash
pip install -U funasr modelscope
```

**Usage**:
```python
from funasr import AutoModel

# Load model
model = AutoModel(
    model="emotion2vec_plus_base",  # or "emotion2vec_plus_large"
    model_revision="v2.0.4",
    device="cuda:0"
)

# Classify emotion
res = model.generate(
    input="path/to/audio.wav",
    granularity="utterance",  # or "frame" for frame-level
    extract_embedding=False,  # True for embeddings, False for classification
    cache={}
)

# Result format:
# {
#     "emotion": "happy",
#     "confidence": 0.95,
#     "embedding": [...]  # if extract_embedding=True
# }
```

**Pros**:
- ✅ Highest accuracy (foundation model)
- ✅ Multilingual support (may work better for Hindi)
- ✅ Universal model (generalizes well)
- ✅ Can extract embeddings for similarity analysis
- ✅ Frame-level or utterance-level features

**Cons**:
- Requires funasr/modelscope (additional dependency)
- Larger model size (especially large variant)

**GitHub**: https://github.com/ddlBoJack/emotion2vec

---

## Multimodal LLMs

### 1. Qwen2.5-Omni

**Model**: `Qwen/Qwen2.5-Omni`

**Status**: ✅ Open source, supports audio input

**Capabilities**:
- End-to-end multimodal (text, audio, vision, video)
- Real-time speech generation
- Can understand and reason about emotions
- Streaming audio processing

**Usefulness Assessment**: ⚠️ **Limited for Direct Emotion Classification**

**Pros**:
- ✅ Can reason about emotions (not just classify) - provides explanations
- ✅ Multimodal understanding (can combine audio + text context)
- ✅ Streaming support for real-time processing
- ✅ Can handle conversational queries about emotions
- ✅ General-purpose model (not limited to emotion tasks)

**Cons**:
- ❌ **Not specialized for emotion recognition** - designed for general multimodal tasks
- ❌ **No published emotion recognition benchmarks** - performance unknown
- ❌ **Larger model size** - requires more GPU memory (~7B+ parameters)
- ❌ **Slower inference** - not optimized for batch emotion classification
- ❌ **More complex setup** - requires more dependencies and configuration
- ❌ **May be less accurate** - specialized models (emotion2vec) likely outperform

**When to Use Qwen2.5-Omni**:
- ✅ You need **reasoning/explanations** about emotions (not just classification)
- ✅ You want to **combine audio with text context** for emotion understanding
- ✅ You need **conversational interaction** about emotions
- ✅ You're doing **research/exploration** of emotion understanding
- ✅ You need a **general-purpose multimodal model** that happens to handle audio

**When NOT to Use Qwen2.5-Omni**:
- ❌ **Pure emotion classification** - use emotion2vec or wav2vec2 instead
- ❌ **Batch processing** many audio files - too slow
- ❌ **Production benchmarking** - no published accuracy metrics
- ❌ **Limited GPU memory** - requires significant resources
- ❌ **High accuracy requirements** - specialized models are better

**Performance Comparison** (Based on Research):
- **Omni-LLMs** (similar to Qwen2.5-Omni): Competitive with fine-tuned models on IEMOCAP/MELD
- **emotion2vec**: Purpose-built for emotion recognition, likely superior for pure classification
- **Qwen2.5-Omni**: No specific benchmarks published yet

**Recommendation**: 
- **For benchmarking**: ❌ Not recommended - use emotion2vec or wav2vec2
- **For research/exploration**: ✅ Useful if you need reasoning capabilities
- **For production**: ❌ Not recommended - use specialized SER models

**Usage**:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni")

# Process audio and ask about emotion
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "path/to/audio.wav"},
            {"type": "text", "text": "What emotion is expressed in this audio?"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], audios=["path/to/audio.wav"], return_tensors="pt")
outputs = model.generate(**inputs)
response = processor.batch_decode(outputs, skip_special_tokens=True, audio_length_threshold=0.0)
```

**Pros**:
- Can reason about emotions (not just classify)
- Multimodal understanding
- Can provide explanations
- Streaming support

**Cons**:
- ❌ **Not specialized for emotion recognition** - lower accuracy likely
- ❌ **No published benchmarks** - performance unknown
- ❌ **Larger model size** (~7B parameters)
- ❌ **Slower for batch processing**
- ❌ **Requires more GPU memory** (16GB+ recommended)
- ❌ **More complex setup**

---

### 2. EmoQ Framework

**Status**: Research model, may require implementation

**Approach**: Speech-Aware Q-Former + LLM

**Features**:
- Cross-modal alignment between audio and text
- Multi-objective affective learning
- Soft-prompt injection

**Pros**:
- Good accuracy on benchmarks
- Can provide reasoning

**Cons**:
- May require custom implementation
- Less straightforward than direct SER models

---

## Model Comparison

| Model | Emotions | Accuracy | Multilingual | Size | Ease of Use | Recommendation |
|-------|----------|----------|--------------|------|-------------|----------------|
| **emotion2vec+ large** | 9 | Highest | ✅ Yes | ~300M | Medium | ⭐⭐⭐⭐⭐ Best overall |
| **emotion2vec+ base** | 9 | High | ✅ Yes | ~90M | Medium | ⭐⭐⭐⭐ Good balance |
| **Dpngtm/wav2vec2** | 8 | 79.57% | ❌ No | Medium | Easy | ⭐⭐⭐⭐ Good for English |
| **ehcalabres/wav2vec2** | 7 | ~75% | ❌ No | Medium | Easy | ⭐⭐⭐ Already in use |
| **SpeechBrain IEMOCAP** | 4 | 78.7% | ❌ No | Medium | Easy | ⭐⭐⭐ Reliable |
| **Qwen2.5-Omni** | Variable | Unknown* | ✅ Yes | Large (~7B) | Hard | ⭐⭐ For reasoning only |

---

## Implementation Examples

### Example 1: emotion2vec+ Integration

```python
"""
emotion2vec_benchmark.py

Use emotion2vec+ for emotion recognition benchmarking.
"""

from funasr import AutoModel
from pathlib import Path
import json

class Emotion2VecEvaluator:
    def __init__(self, model_size="base"):
        """
        Args:
            model_size: "seed", "base", or "large"
        """
        self.model = AutoModel(
            model=f"emotion2vec_plus_{model_size}",
            model_revision="v2.0.4",
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # Map to your emotion system
        self.emotion_mapping = {
            "Angry": "angry",
            "Happy": "happy",
            "Sad": "sad",
            "Neutral": "neutral",
            "Fearful": "fearful",
            "Disgusted": "disgusted",
            "Surprised": "surprised",
            "Other": "neutral",  # Fallback
            "Unknown": "neutral"  # Fallback
        }
    
    def classify_emotion(self, audio_path):
        """Classify emotion from audio file."""
        res = self.model.generate(
            input=str(audio_path),
            granularity="utterance",
            extract_embedding=False,
            cache={}
        )
        
        predicted = res[0]["emotion"]
        confidence = res[0].get("confidence", 0.0)
        
        # Map to your emotion system
        mapped_emotion = self.emotion_mapping.get(predicted, "neutral")
        
        return {
            "predicted": mapped_emotion,
            "original": predicted,
            "confidence": confidence
        }
    
    def evaluate_benchmark(self, audio_dir, expected_emotions):
        """
        Evaluate benchmark audio files.
        
        Args:
            audio_dir: Directory with generated audio files
            expected_emotions: Dict mapping filename patterns to expected emotions
        """
        results = []
        correct = 0
        total = 0
        
        audio_dir = Path(audio_dir)
        for audio_file in audio_dir.glob("*.wav"):
            # Find expected emotion from filename
            expected = None
            for pattern, emotion in expected_emotions.items():
                if pattern in audio_file.name:
                    expected = emotion
                    break
            
            if expected is None:
                continue
            
            # Classify
            result = self.classify_emotion(audio_file)
            result["file"] = audio_file.name
            result["expected"] = expected
            result["correct"] = result["predicted"].lower() == expected.lower()
            
            results.append(result)
            total += 1
            if result["correct"]:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }

# Usage
if __name__ == "__main__":
    evaluator = Emotion2VecEvaluator(model_size="base")
    
    expected_emotions = {
        "basic_happy": "happy",
        "basic_sad": "sad",
        "basic_angry": "angry",
        "new_sarcastic": "sarcastic",  # May map to "Other" or "Angry"
        # ... more patterns
    }
    
    results = evaluator.evaluate_benchmark(
        "benchmark_output/ravdess/audio",
        expected_emotions
    )
    
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct: {results['correct']}/{results['total']}")
```

---

### Example 2: Enhanced Benchmark Script with Multiple Models

```python
"""
multi_model_benchmark.py

Compare results from multiple emotion recognition models.
"""

from pathlib import Path
import json
from typing import Dict, List

class MultiModelEvaluator:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load multiple emotion recognition models."""
        # emotion2vec+
        try:
            from funasr import AutoModel
            self.models["emotion2vec_base"] = AutoModel(
                model="emotion2vec_plus_base",
                model_revision="v2.0.4",
                device="cuda:0"
            )
        except ImportError:
            print("Warning: emotion2vec not available (install funasr)")
        
        # wav2vec2-emotion
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.models["wav2vec2"] = {
                "extractor": AutoFeatureExtractor.from_pretrained(model_id),
                "model": AutoModelForAudioClassification.from_pretrained(model_id)
            }
        except Exception as e:
            print(f"Warning: wav2vec2 not available: {e}")
        
        # Dpngtm model
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            model_id = "Dpngtm/wav2vec2-emotion-recognition"
            self.models["dpngtm"] = {
                "processor": Wav2Vec2Processor.from_pretrained(model_id),
                "model": Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
            }
        except Exception as e:
            print(f"Warning: Dpngtm model not available: {e}")
    
    def evaluate_all_models(self, audio_path, expected_emotion):
        """Evaluate single audio file with all models."""
        results = {}
        
        # emotion2vec+
        if "emotion2vec_base" in self.models:
            try:
                res = self.models["emotion2vec_base"].generate(
                    input=str(audio_path),
                    granularity="utterance",
                    extract_embedding=False
                )
                results["emotion2vec"] = {
                    "predicted": res[0]["emotion"],
                    "confidence": res[0].get("confidence", 0.0)
                }
            except Exception as e:
                results["emotion2vec"] = {"error": str(e)}
        
        # wav2vec2
        if "wav2vec2" in self.models:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                
                inputs = self.models["wav2vec2"]["extractor"](
                    waveform.squeeze().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = self.models["wav2vec2"]["model"](**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1).item()
                    pred_label = self.models["wav2vec2"]["model"].config.id2label[pred_id]
                
                results["wav2vec2"] = {
                    "predicted": pred_label,
                    "confidence": probs[0, pred_id].item()
                }
            except Exception as e:
                results["wav2vec2"] = {"error": str(e)}
        
        # Add expected and compute accuracy
        results["expected"] = expected_emotion
        for model_name, result in results.items():
            if model_name != "expected" and "error" not in result:
                result["correct"] = result["predicted"].lower() == expected_emotion.lower()
        
        return results
    
    def benchmark_directory(self, audio_dir, expected_emotions):
        """Benchmark entire directory with all models."""
        audio_dir = Path(audio_dir)
        all_results = []
        
        for audio_file in audio_dir.glob("*.wav"):
            # Find expected emotion
            expected = None
            for pattern, emotion in expected_emotions.items():
                if pattern in audio_file.name:
                    expected = emotion
                    break
            
            if expected is None:
                continue
            
            # Evaluate with all models
            result = self.evaluate_all_models(audio_file, expected)
            result["file"] = audio_file.name
            all_results.append(result)
        
        # Compute aggregate statistics
        stats = {}
        for model_name in ["emotion2vec", "wav2vec2", "dpngtm"]:
            model_results = [r for r in all_results if model_name in r and "error" not in r[model_name]]
            if model_results:
                correct = sum(1 for r in model_results if r[model_name].get("correct", False))
                total = len(model_results)
                stats[model_name] = {
                    "accuracy": correct / total if total > 0 else 0,
                    "correct": correct,
                    "total": total
                }
        
        return {
            "statistics": stats,
            "detailed_results": all_results
        }

# Usage
if __name__ == "__main__":
    evaluator = MultiModelEvaluator()
    
    expected_emotions = {
        "basic_happy": "happy",
        "basic_sad": "sad",
        "basic_angry": "angry",
        # ... more patterns
    }
    
    results = evaluator.benchmark_directory(
        "benchmark_output/ravdess/audio",
        expected_emotions
    )
    
    print("\nModel Comparison:")
    for model_name, stats in results["statistics"].items():
        print(f"{model_name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save results
    with open("multi_model_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
```

---

## Recommendations for Your Benchmark

### For English Audio:
1. **Primary**: `emotion2vec_plus_base` - Best accuracy, universal model
2. **Secondary**: `Dpngtm/wav2vec2-emotion-recognition` - 8 emotions, good accuracy
3. **Fallback**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` - Already integrated

### For Hindi Audio:
1. **Primary**: `emotion2vec_plus_base` or `large` - Multilingual support, best accuracy
2. **Secondary**: Traditional wav2vec2 models (may have limited Hindi support)
3. **NOT Recommended**: `Qwen2.5-Omni` - No emotion benchmarks, not specialized

### For Comprehensive Benchmarking:
- Use **multiple models** and compare results
- emotion2vec+ for best accuracy
- wav2vec2 models for comparison
- Report agreement between models (consensus)

---

## Installation Commands

```bash
# emotion2vec+
pip install -U funasr modelscope

# Traditional models (already in your requirements)
pip install transformers torchaudio

# SpeechBrain
pip install speechbrain

# Qwen2.5-Omni (if needed)
pip install transformers accelerate
```

---

## References

1. **emotion2vec**: https://huggingface.co/emotion2vec
2. **emotion2vec GitHub**: https://github.com/ddlBoJack/emotion2vec
3. **Qwen2.5-Omni**: https://github.com/qwenlm/qwen2.5-omni
4. **SpeechBrain**: https://huggingface.co/speechbrain
5. **Hugging Face Audio Models**: https://huggingface.co/models?pipeline_tag=audio-classification

---

## Qwen2.5-Omni: Detailed Analysis

### Is Qwen2.5-Omni Useful for Emotion Recognition?

**Short Answer**: ⚠️ **Limited usefulness for direct emotion classification benchmarking**

### Detailed Analysis

#### 1. **Purpose-Built vs General-Purpose**

| Aspect | Qwen2.5-Omni | emotion2vec | wav2vec2-emotion |
|--------|--------------|-------------|------------------|
| **Primary Purpose** | General multimodal understanding | Emotion recognition | Speech understanding |
| **Specialization** | None (general-purpose) | ✅ Emotion-specific | Speech (not emotion) |
| **Pre-training** | General multimodal data | ✅ 262+ hours emotion data | General speech data |
| **Optimization** | Multimodal tasks | ✅ Emotion tasks | Speech tasks |

**Verdict**: Qwen2.5-Omni is **not optimized for emotion recognition** - it's a general-purpose model.

#### 2. **Performance Comparison**

Based on research (OmniVox study on similar omni-LLMs):
- **Zero-shot performance**: Competitive with fine-tuned models (within 2-7% on IEMOCAP/MELD)
- **vs emotion2vec**: emotion2vec likely superior (purpose-built, specialized pre-training)
- **vs wav2vec2**: Similar or slightly better, but wav2vec2 is faster

**Key Issue**: No specific benchmarks for Qwen2.5-Omni emotion recognition published yet.

#### 3. **Resource Requirements**

| Resource | Qwen2.5-Omni | emotion2vec+ base | wav2vec2 |
|----------|--------------|-------------------|----------|
| **Model Size** | ~7B parameters | ~90M parameters | ~300M parameters |
| **GPU Memory** | 16GB+ (recommended) | 4-8GB | 2-4GB |
| **Inference Speed** | Slow (LLM inference) | Fast | Fast |
| **Batch Processing** | Not optimized | ✅ Optimized | ✅ Optimized |

**Verdict**: Qwen2.5-Omni requires **significantly more resources** for similar or worse performance.

#### 4. **Use Cases Where Qwen2.5-Omni Shines**

✅ **Good for**:
- **Reasoning about emotions**: "Why does this audio sound sad?"
- **Contextual understanding**: Combining audio with text descriptions
- **Conversational queries**: "Compare the emotions in these two audio clips"
- **Research/exploration**: Understanding emotion perception
- **Multimodal tasks**: When you need to process audio + text + images together

❌ **Not good for**:
- **Pure classification**: Just labeling emotions
- **Batch processing**: Processing hundreds of audio files
- **Production systems**: Need reliable, fast classification
- **Benchmarking**: No published accuracy metrics
- **Resource-constrained environments**: Too heavy

#### 5. **Practical Recommendation**

**For Your Benchmarking Use Case**:

```
❌ DON'T use Qwen2.5-Omni for:
- Direct emotion classification
- Batch processing benchmark audio files
- Production emotion recognition
- When you need high accuracy

✅ DO use Qwen2.5-Omni for:
- Research on emotion understanding
- When you need explanations/reasoning
- Multimodal emotion analysis (audio + text)
- Exploring emotion perception
```

**Better Alternatives**:
1. **emotion2vec+ base** - Best accuracy, multilingual
2. **Dpngtm/wav2vec2** - Good accuracy, 8 emotions
3. **Your current wav2vec2** - Baseline comparison

### Conclusion

**Qwen2.5-Omni's usefulness for emotion recognition**: ⭐⭐ (2/5 stars)

- **Not recommended** for direct emotion classification benchmarking
- **Useful** only if you need reasoning/explanations about emotions
- **Specialized models** (emotion2vec, wav2vec2) are better for your use case
- **Higher resource cost** without clear accuracy benefits

**Bottom Line**: Stick with emotion2vec+ or wav2vec2 models for benchmarking. Qwen2.5-Omni is overkill and not optimized for this task.

---

## Next Steps

1. **Add emotion2vec+ to your benchmark script** - Best overall choice
2. **Compare multiple models** - Use ensemble for more reliable results
3. **Handle emotion mapping** - Map model outputs to your 16 emotions
4. **Report model agreement** - Show consensus between models
5. **Skip Qwen2.5-Omni** - Not suitable for direct emotion classification benchmarking
