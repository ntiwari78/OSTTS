# Fine-tuning Guide: Emotion Control with LoRA/Adapter

This guide explains how to fine-tune the Chatterbox T3 model for emotion control using LoRA (Low-Rank Adaptation) or Adapter layers.

## Overview

Fine-tuning with LoRA/Adapter allows you to:
- **Efficiently train** on emotion-labeled Hindi audio data
- **Preserve base model** weights (only train small adapter layers)
- **Specialize** emotion embeddings for better emotion control
- **Require less data** and compute compared to full fine-tuning

## Setup

### 1. Install Dependencies

```bash
pip install torch torchaudio librosa soundfile tqdm requests
```

### 2. Prepare Dataset

#### Option A: Use Existing Dataset

```bash
# Setup data directory structure
python download_hindi_emotion_data.py --output_dir data/hindi_emotions

# Add your audio files organized by emotion:
# data/hindi_emotions/
#   emotion_happy/
#     audio1.wav
#     audio2.wav
#   emotion_sad/
#     audio1.wav
#   ...
```

#### Option B: Download Public Datasets

**IndicTTS Dataset:**
- Website: https://www.iitm.ac.in/donlab/tts/
- Contains Hindi TTS data
- Organize by emotion labels

**AI4Bharat IndicTTS:**
- GitHub: https://github.com/ai4bharat/IndicTTS
- Follow their setup instructions
- Extract Hindi samples and label with emotions

**Common Voice Hindi:**
- Website: https://commonvoice.mozilla.org/
- Download Hindi dataset
- Manually label with emotions or use emotion classification

### 3. Organize Data

Expected structure:
```
data/hindi_emotions/
  emotion_happy/
    namaste_happy.wav
    khushi_happy.wav
    ...
  emotion_sad/
    dukhi_sad.wav
    ...
  emotion_angry/
    ...
  emotion_neutral/
    ...
  emotion_excited/
    ...
  emotion_calm/
    ...
  emotion_surprised/
    ...
  emotion_fearful/
    ...
```

### 4. Label Audio Files

Use the helper script:
```bash
python data/hindi_emotions/label_audio.py <audio_file> <emotion>
```

Or manually organize files into emotion folders.

## Training

### Basic Training with LoRA

```bash
python train_emotion_lora.py \
    --data_dir data/hindi_emotions \
    --output_dir checkpoints/emotion_lora \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4
```

### Training with Adapters

```bash
python train_emotion_lora.py \
    --data_dir data/hindi_emotions \
    --output_dir checkpoints/emotion_adapter \
    --use_adapter \
    --adapter_size 64 \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4
```

### Parameters

- `--lora_rank`: LoRA rank (lower = fewer parameters, default: 8)
- `--lora_alpha`: LoRA alpha scaling (default: 16.0)
- `--adapter_size`: Adapter bottleneck size (default: 64)
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device (auto, cuda, mps, cpu)

## Architecture

### LoRA (Low-Rank Adaptation)

LoRA applies low-rank decomposition to linear layers:
- **Base weights**: Frozen (not updated)
- **LoRA weights**: Small matrices A and B (trainable)
- **Output**: `base_output + lora_output * scaling`

**Advantages:**
- Very parameter-efficient
- Fast training
- Easy to merge weights for inference

### Adapter Layers

Adapters add bottleneck layers to transformer blocks:
- **Down projection**: hidden_size → adapter_size
- **Activation**: GELU
- **Up projection**: adapter_size → hidden_size
- **Residual connection**: `x + adapter(x)`

**Advantages:**
- Modular (easy to add/remove)
- Good for multi-task learning
- Stable training

## Training Process

1. **Load pre-trained model**: ChatterboxMultilingualTTS
2. **Apply LoRA/Adapter**: Add trainable layers
3. **Freeze base model**: Only train adapter parameters
4. **Train on emotion data**: Learn emotion-specific patterns
5. **Save checkpoints**: Save adapter weights

## Using Fine-tuned Model

### Load Fine-tuned Weights

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch

# Load base model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Load fine-tuned LoRA/Adapter weights
checkpoint = torch.load("checkpoints/emotion_lora/checkpoint_epoch_10.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Use with emotion types
wav = model.generate(
    text="नमस्ते",
    language_id="hi",
    emotion="happy"  # Now fine-tuned for better emotion control!
)
```

### Merge LoRA Weights (Optional)

For faster inference, merge LoRA weights into base model:

```python
# Merge LoRA weights
for module in model.t3.tfmr.modules():
    if isinstance(module, LoRALinear):
        module.merge_weights()
```

## Tips

1. **Data Quality**: Ensure audio files are clean and properly labeled
2. **Data Balance**: Try to have similar amounts of data per emotion
3. **Learning Rate**: Start with 1e-4, adjust based on loss
4. **Batch Size**: Adjust based on GPU memory
5. **Validation**: Split data into train/val sets for monitoring
6. **Early Stopping**: Stop training when validation loss plateaus

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Use gradient accumulation
- Use smaller LoRA rank/adapter size

**Poor Results:**
- Check data quality and labels
- Increase training data
- Adjust learning rate
- Try different LoRA rank/adapter size

**Slow Training:**
- Use GPU (CUDA or MPS)
- Reduce batch size
- Use mixed precision training

## Advanced: Multi-Emotion Fine-tuning

To fine-tune for multiple emotions simultaneously:

1. Organize data with multiple emotion labels
2. Use weighted loss for different emotions
3. Train with emotion-specific learning rates
4. Use curriculum learning (start with easy emotions)

## References

- LoRA Paper: https://arxiv.org/abs/2106.09685
- Adapter Paper: https://arxiv.org/abs/1902.00751
- IndicTTS: https://www.iitm.ac.in/donlab/tts/
- AI4Bharat IndicTTS: https://github.com/ai4bharat/IndicTTS

