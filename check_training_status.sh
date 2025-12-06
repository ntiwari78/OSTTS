#!/bin/bash
# Script to check training status

echo "=== Training Status Check ==="
echo ""

# Check if process is running
echo "1. Checking if training process is running..."
if ps aux | grep -v grep | grep -q "train_emotion_lora"; then
    echo "   ✓ Training is still running"
    ps aux | grep -v grep | grep "train_emotion_lora" | head -1
else
    echo "   ✓ Training process is not running (likely completed)"
fi

echo ""

# Check checkpoint files
echo "2. Checking checkpoint files..."
if [ -d "checkpoints/emotion_lora" ]; then
    echo "   Checkpoint directory exists"
    echo "   Files:"
    ls -lht checkpoints/emotion_lora/*.pt 2>/dev/null | head -5
    echo ""
    
    # Check latest checkpoint
    latest_checkpoint=$(ls -t checkpoints/emotion_lora/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$latest_checkpoint" ]; then
        echo "   Latest checkpoint: $latest_checkpoint"
        file_size=$(ls -lh "$latest_checkpoint" | awk '{print $5}')
        file_time=$(ls -lht "$latest_checkpoint" | awk '{print $6, $7, $8}')
        echo "   Size: $file_size"
        echo "   Modified: $file_time"
        
        # Try to load checkpoint
        echo ""
        echo "3. Checking checkpoint contents..."
        python3 << EOF
import torch
import sys
try:
    checkpoint = torch.load("$latest_checkpoint", map_location='cpu', weights_only=False)
    print("   ✓ Checkpoint is valid")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"   Has T3 state: {'t3_state_dict' in checkpoint}")
    print(f"   Has emotion embeddings: {'emotion_embeddings_state_dict' in checkpoint}")
    print(f"   Has optimizer state: {'optimizer_state_dict' in checkpoint}")
except Exception as e:
    print(f"   ✗ Error loading checkpoint: {e}")
    sys.exit(1)
EOF
    else
        echo "   ✗ No checkpoint files found"
    fi
else
    echo "   ✗ Checkpoint directory does not exist"
fi

echo ""
echo "=== Status Check Complete ==="

