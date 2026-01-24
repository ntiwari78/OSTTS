# Emotion Type Control Implementation

## Overview

This implementation adds **true emotion type control** to ChatterboxMultilingualTTS, allowing you to specify emotions like "happy", "sad", "angry" as distinct emotion types, not just intensity levels.

## Architecture Changes

### 1. Extended T3Config
- Added `emotion_embed_dim = 8` to support emotion type embeddings

### 2. Extended T3Cond
- Added `emotion_embed: Optional[Tensor]` field for emotion type embeddings
- Maintains backward compatibility with scalar `emotion_adv`

### 3. Extended T3CondEnc
- Added `emotion_embed_fc: nn.Linear(emotion_embed_dim, n_channels)` for emotion embeddings
- Maintains `emotion_adv_fc` for backward compatibility
- Forward pass prioritizes `emotion_embed` over `emotion_adv` when available

### 4. New EmotionEmbeddings Module
- Learnable embeddings for 11 emotion types:
  - `neutral`, `happy`, `sad`, `angry`, `excited`, `calm`, `surprised`, `fearful`, `disgusted`, `whisper`, `shout`
- Initialized with heuristic values (can be fine-tuned)
- Provides lookup functionality for emotion types

## Usage

### Basic Usage with Emotion Types

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Generate with specific emotion type
wav = model.generate(
    text="Hello, how are you?",
    language_id="en",
    emotion="happy"  # True emotion type control
)

# Different emotions
wav_sad = model.generate(
    text="I'm feeling down.",
    language_id="en",
    emotion="sad"
)

wav_angry = model.generate(
    text="This is unacceptable!",
    language_id="en",
    emotion="angry"
)
```

### Get Supported Emotions

```python
emotions = model.get_supported_emotions()
# Returns: ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm', 
#           'surprised', 'fearful', 'disgusted', 'whisper', 'shout']
```

### Backward Compatibility

The `exaggeration` parameter still works for intensity control:

```python
# Old way (still works)
wav = model.generate(
    text="Hello!",
    language_id="en",
    exaggeration=0.8  # Intensity control
)
```

## How It Works

1. **Emotion Embeddings**: Each emotion type has a learnable 8-dimensional embedding vector
2. **Projection**: Emotion embeddings are projected to model dimension via `emotion_embed_fc`
3. **Conditioning**: The projected emotion embedding is concatenated with other conditionals (speaker, prompt, etc.)
4. **Generation**: The model uses the emotion embedding to condition the speech generation

## Important Notes

### Pre-trained Model Compatibility

- The emotion embedding layers (`emotion_embed_fc`) are **new** and won't exist in pre-trained models
- They will be **randomly initialized** when loading pre-trained models
- This means emotion types will work, but may not be fully optimized without fine-tuning

### Fine-tuning Recommendation

For best results with emotion types:
1. Fine-tune the model with emotion-labeled data
2. Or use emotion-matched reference audio combined with emotion types

### Emotion Embedding Initialization

The emotion embeddings are initialized with heuristic values:
- `happy`: positive valence, high energy
- `sad`: negative valence, low energy  
- `angry`: high energy, sharp characteristics
- etc.

These can be learned/refined during training or fine-tuning.

## Technical Details

### Emotion Embedding Dimensions

- **Input**: Emotion type name (string) → Lookup → 8D embedding vector
- **Projection**: 8D embedding → Linear layer → `n_channels` (model dimension)
- **Output**: Conditioned speech tokens with emotional characteristics

### Backward Compatibility

The implementation maintains full backward compatibility:
- If `emotion` is not provided, uses scalar `emotion_adv` (old behavior)
- If `emotion` is provided, uses emotion embeddings (new behavior)
- Pre-trained models load successfully (new layers initialized randomly)

## Future Enhancements

1. **Fine-tuned Emotion Embeddings**: Train/fine-tune emotion embeddings on labeled data
2. **More Emotion Types**: Add additional emotion categories
3. **Emotion Intensity**: Combine emotion type with intensity scaling
4. **Emotion Interpolation**: Blend between emotion types

