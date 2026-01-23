#!/usr/bin/env python3
"""
Quick test to verify emotion checkpoint loading is working correctly.
This script tests that the load_emotion_checkpoint method properly loads
both t3_state_dict and emotion_embeddings_state_dict from checkpoints.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def test_checkpoint_loading():
    """Test that emotion checkpoints are loaded correctly"""

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}\n")

    # Load base model
    print("Loading base model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Base model loaded\n")

    # Store initial emotion embedding weights for comparison
    initial_weights = model.emotion_embeddings.embedding.weight.clone()
    print(f"Initial emotion embedding shape: {initial_weights.shape}")
    print(f"Initial emotion embedding sample: {initial_weights[0, :5]}\n")

    # Test loading merged checkpoint
    checkpoint_path = "checkpoints/emotion_merged/checkpoint_merged.pt"

    if Path(checkpoint_path).exists():
        print(f"Testing checkpoint: {checkpoint_path}")
        print("-" * 70)

        try:
            model.load_emotion_checkpoint(checkpoint_path)

            # Check if weights changed
            new_weights = model.emotion_embeddings.embedding.weight
            print(f"New emotion embedding shape: {new_weights.shape}")
            print(f"New emotion embedding sample: {new_weights[0, :5]}")

            # Verify weights actually changed
            weights_changed = not torch.allclose(initial_weights, new_weights, atol=1e-6)

            if weights_changed:
                print("\n✓ SUCCESS: Emotion embeddings were updated!")
                print(f"  Weight difference (L2 norm): {torch.norm(initial_weights - new_weights).item():.6f}")
            else:
                print("\n✗ FAILURE: Emotion embeddings were NOT updated!")
                print("  This means the checkpoint was not loaded correctly.")
                return False

        except Exception as e:
            print(f"\n✗ FAILURE: Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    print("\n" + "=" * 70)
    print("Checkpoint loading test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)
