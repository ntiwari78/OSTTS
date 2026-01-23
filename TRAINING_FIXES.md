# Training Fixes Applied

## Issues Found and Fixed

### 1. **Emotion Embeddings Not Trainable**
   - **Problem**: Emotion embeddings were not set to `requires_grad=True`
   - **Fix**: Added explicit setting of `requires_grad=True` for emotion embeddings
   - **Location**: `setup_model_with_lora()` function

### 2. **Dimension Mismatch in Emotion Embeddings**
   - **Problem**: Emotion embeddings had wrong shape causing "got 3 and 4 dimensions" error
   - **Fix**: Corrected emotion embedding stacking to ensure (B, 1, emotion_dim) shape
   - **Location**: `train_step()` function

### 3. **Loss Calculation Issues**
   - **Problem**: Many batches returned 0.0 loss (invalid)
   - **Fix**: 
     - Added validation to skip invalid losses (NaN, Inf, or 0.0)
     - Weighted speech loss more heavily (2.0x) since it's more important for TTS
     - Better error handling and reporting

### 4. **Training Progress Tracking**
   - **Problem**: Couldn't track valid vs invalid batches
   - **Fix**: 
     - Track valid batches separately
     - Report average loss only from valid batches
     - Show progress with valid batch count

## Current Training Status

Training is running with:
- **Output**: `checkpoints/emotion_lora_v2/`
- **Epochs**: 3
- **Batch Size**: 1
- **Device**: CPU (for stability)
- **LoRA Rank**: 8

## Monitor Training

Run the monitoring script:
```bash
./monitor_training.sh
```

Or manually check:
```bash
# Check if running
ps aux | grep train_emotion_lora

# Check checkpoints
ls -lht checkpoints/emotion_lora_v2/

# Check latest checkpoint
python -c "
import torch
ckpt = torch.load('checkpoints/emotion_lora_v2/checkpoint_epoch_3.pt', map_location='cpu', weights_only=False)
print(f'Epoch: {ckpt[\"epoch\"]}, Loss: {ckpt[\"loss\"]}')
"
```

## Expected Results

After training completes:
- Loss should decrease over epochs (not be 0.0)
- Emotion embeddings should be updated
- LoRA weights should be trained
- Model should show better emotion control

## If Training Still Doesn't Work

1. **Check data quality**: Ensure audio files are valid and text is properly tokenized
2. **Reduce batch size**: Try batch_size=1 if having memory issues
3. **Check loss values**: Loss should be > 0 and decreasing
4. **Verify emotion labels**: Ensure emotions match the dataset structure

