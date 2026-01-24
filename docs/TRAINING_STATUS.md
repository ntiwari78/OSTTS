# Training Status

## Current Status

Training script is running with the IESC Hindi emotion dataset.

### Configuration
- **Data**: 600 files (120 per emotion: angry, happy, sad, neutral, fearful)
- **LoRA Rank**: 8
- **Batch Size**: 2
- **Epochs**: 1 (test run)
- **Learning Rate**: 1e-4
- **Device**: MPS (Mac GPU)

### Important Note

⚠️ **The current training script uses a placeholder loss function** and will not actually learn from the data. This is a skeleton implementation for testing the data loading and model setup.

### What's Working
- ✅ Data loading from IESC dataset
- ✅ Model loading with LoRA layers
- ✅ Emotion embedding setup
- ✅ Data organization (600 files across 5 emotions)

### What Needs Implementation
- ❌ Actual TTS loss calculation (currently placeholder)
- ❌ Speech tokenization from audio
- ❌ Speaker embedding extraction from audio
- ❌ Proper forward pass through T3 model

### Next Steps

To make training actually work, the `train_step` function needs to:

1. **Convert audio to speech tokens**:
   ```python
   # Resample audio to 16kHz
   audio_16k = resample(audio, orig_sr, 16000)
   # Tokenize using S3Gen tokenizer
   speech_tokens, speech_token_lens = model.s3gen.tokenizer(audio_16k)
   ```

2. **Extract speaker embeddings**:
   ```python
   speaker_emb = model.ve.embeds_from_wavs([audio_16k], sample_rate=16000)
   ```

3. **Calculate actual loss**:
   ```python
   loss = model.t3.loss(
       t3_cond=t3_cond,
       text_tokens=text_tokens,
       text_token_lens=text_token_lens,
       speech_tokens=speech_tokens,
       speech_token_lens=speech_token_lens,
   )
   ```

### Check Training Progress

```bash
# Check if training is running
ps aux | grep train_emotion_lora

# Check checkpoints
ls -lh checkpoints/emotion_lora/

# View logs (if redirected)
tail -f training.log
```

### To Stop Training

```bash
pkill -f train_emotion_lora
```

