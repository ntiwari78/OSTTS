#!/bin/bash
# Monitor training progress

echo "=== Training Monitor ==="
echo ""

# Check if training is running
if ps aux | grep -v grep | grep -q "train_emotion_lora"; then
    echo "✓ Training is RUNNING"
    ps aux | grep -v grep | grep "train_emotion_lora" | head -1 | awk '{print "  PID:", $2, "CPU:", $3"%", "Memory:", $4"%"}'
else
    echo "✗ Training is NOT running (completed or stopped)"
fi

echo ""

# Check checkpoints
for checkpoint_dir in checkpoints/emotion_lora checkpoints/emotion_lora_v2; do
    if [ -d "$checkpoint_dir" ]; then
        echo "Checkpoints in $checkpoint_dir:"
        ls -lht "$checkpoint_dir"/*.pt 2>/dev/null | head -3 | awk '{print "  ", $9, "(" $5 ")", $6, $7, $8}'
        
        # Check latest checkpoint
        latest=$(ls -t "$checkpoint_dir"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo ""
            echo "Latest checkpoint: $latest"
            python3 << EOF
import torch
try:
    ckpt = torch.load("$latest", map_location='cpu', weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Loss: {ckpt.get('loss', 'N/A')}")
    print(f"  ✓ Valid checkpoint")
except Exception as e:
    print(f"  ✗ Error: {e}")
EOF
        fi
        echo ""
    fi
done

echo "=== End Monitor ==="

