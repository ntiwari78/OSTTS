#!/usr/bin/env python3
"""
Merge multiple emotion LoRA checkpoints into a unified model.

This script combines the LoRA weights from multiple emotion-trained checkpoints
(RAVDESS, CREMA-D, IESC) into a single merged checkpoint that benefits from
all training data.

Merging strategies:
1. Weighted average: Combine based on dataset size or quality
2. Task arithmetic: Add/subtract task vectors
3. TIES merging: Trim, elect signs, merge

Usage:
    python merge_emotion_checkpoints.py --output checkpoints/emotion_merged/checkpoint.pt

    # Custom weights (based on dataset size)
    python merge_emotion_checkpoints.py --weights 0.15,0.78,0.07 --output merged.pt
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Dict:
    """Load a checkpoint and return its state dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def flatten_checkpoint(checkpoint: Dict) -> Dict:
    """
    Flatten a nested checkpoint structure into a single state dict.

    Handles checkpoints with nested structures like:
    - emotion_embeddings_state_dict: OrderedDict
    - t3_state_dict: OrderedDict

    Returns a flat dict with prefixed keys.
    """
    flat = {}

    # Handle emotion embeddings
    if "emotion_embeddings_state_dict" in checkpoint:
        for key, value in checkpoint["emotion_embeddings_state_dict"].items():
            flat[f"emotion_embeddings.{key}"] = value

    # Handle T3 state dict
    if "t3_state_dict" in checkpoint:
        for key, value in checkpoint["t3_state_dict"].items():
            flat[f"t3.{key}"] = value

    # If not nested (already flat), use as-is
    if not flat:
        flat = checkpoint

    return flat


def unflatten_checkpoint(flat: Dict, original_structure: Dict) -> Dict:
    """
    Convert a flat state dict back to the original nested structure.
    """
    result = {}

    # Check if original had nested structure
    has_nested = "emotion_embeddings_state_dict" in original_structure or "t3_state_dict" in original_structure

    if not has_nested:
        return flat

    # Reconstruct nested structure
    emotion_dict = OrderedDict()
    t3_dict = OrderedDict()
    other = {}

    for key, value in flat.items():
        if key.startswith("emotion_embeddings."):
            new_key = key.replace("emotion_embeddings.", "")
            emotion_dict[new_key] = value
        elif key.startswith("t3."):
            new_key = key.replace("t3.", "")
            t3_dict[new_key] = value
        else:
            other[key] = value

    if emotion_dict:
        result["emotion_embeddings_state_dict"] = emotion_dict
    if t3_dict:
        result["t3_state_dict"] = t3_dict

    # Add metadata from original
    if "epoch" in original_structure:
        result["epoch"] = "merged"
    if "loss" in original_structure:
        result["loss"] = 0.0

    result.update(other)

    return result


def get_lora_keys(state_dict: Dict) -> List[str]:
    """Extract LoRA-related keys from state dict."""
    lora_keys = []
    for key in state_dict.keys():
        if "lora" in key.lower() or "emotion" in key.lower():
            lora_keys.append(key)
    return sorted(lora_keys)


def weighted_average_merge(
    checkpoints: List[Dict],
    weights: List[float],
    keys_to_merge: Optional[List[str]] = None,
) -> Dict:
    """
    Merge checkpoints using weighted average.

    Args:
        checkpoints: List of checkpoint state dicts
        weights: List of weights for each checkpoint (should sum to 1.0)
        keys_to_merge: Specific keys to merge (None = all matching keys)

    Returns:
        Merged state dict
    """
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    print(f"Merging with normalized weights: {weights}")

    # Start with a copy of the first checkpoint
    merged = {}

    # Get all keys from all checkpoints
    all_keys = set()
    for ckpt in checkpoints:
        all_keys.update(ckpt.keys())

    # If specific keys provided, filter
    if keys_to_merge:
        all_keys = set(k for k in all_keys if any(pattern in k for pattern in keys_to_merge))

    # Merge each key
    for key in sorted(all_keys):
        # Check which checkpoints have this key
        available_ckpts = [(i, ckpt) for i, ckpt in enumerate(checkpoints) if key in ckpt]

        if not available_ckpts:
            continue

        # Get tensor from first available checkpoint
        first_tensor = available_ckpts[0][1][key]

        if not torch.is_tensor(first_tensor):
            # Non-tensor values: take from first checkpoint
            merged[key] = first_tensor
            continue

        # Weighted average for tensors
        merged_tensor = torch.zeros_like(first_tensor, dtype=torch.float32)
        total_available_weight = 0.0

        for idx, ckpt in available_ckpts:
            tensor = ckpt[key].float()
            if tensor.shape == merged_tensor.shape:
                merged_tensor += weights[idx] * tensor
                total_available_weight += weights[idx]
            else:
                print(f"Warning: Shape mismatch for {key}, skipping checkpoint {idx}")

        # Renormalize if not all checkpoints had this key
        if total_available_weight > 0 and total_available_weight < 1.0:
            merged_tensor = merged_tensor / total_available_weight

        # Preserve original dtype
        merged[key] = merged_tensor.to(first_tensor.dtype)

    return merged


def ties_merge(
    checkpoints: List[Dict],
    base_checkpoint: Dict,
    weights: List[float],
    density: float = 0.5,
    keys_to_merge: Optional[List[str]] = None,
) -> Dict:
    """
    TIES (Trim, Elect Sign, Merge) merging strategy.

    This method is more robust than simple averaging as it:
    1. Trims small-magnitude changes
    2. Resolves sign conflicts
    3. Merges only the agreed-upon directions

    Args:
        checkpoints: List of fine-tuned checkpoint state dicts
        base_checkpoint: Base/pretrained model state dict
        weights: Weights for each checkpoint
        density: Fraction of parameters to keep (0.0-1.0)
        keys_to_merge: Specific keys to merge

    Returns:
        Merged state dict
    """
    print(f"TIES merge with density={density}")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    merged = {}

    # Get all keys
    all_keys = set()
    for ckpt in checkpoints:
        all_keys.update(ckpt.keys())

    if keys_to_merge:
        all_keys = set(k for k in all_keys if any(pattern in k for pattern in keys_to_merge))

    for key in sorted(all_keys):
        available_ckpts = [(i, ckpt) for i, ckpt in enumerate(checkpoints) if key in ckpt]

        if not available_ckpts:
            continue

        first_tensor = available_ckpts[0][1][key]

        if not torch.is_tensor(first_tensor):
            merged[key] = first_tensor
            continue

        # Get base tensor if available
        if key in base_checkpoint:
            base_tensor = base_checkpoint[key].float()
        else:
            base_tensor = torch.zeros_like(first_tensor, dtype=torch.float32)

        # Compute task vectors (delta from base)
        task_vectors = []
        for idx, ckpt in available_ckpts:
            tensor = ckpt[key].float()
            if tensor.shape == base_tensor.shape:
                delta = tensor - base_tensor
                task_vectors.append((idx, delta))

        if not task_vectors:
            merged[key] = first_tensor
            continue

        # Stack task vectors
        deltas = torch.stack([tv[1] for tv in task_vectors])
        task_weights = torch.tensor([weights[tv[0]] for tv in task_vectors])

        # Step 1: Trim - keep only top-k% magnitude values per task vector
        k = int(density * deltas[0].numel())
        if k > 0:
            trimmed_deltas = []
            for delta in deltas:
                flat = delta.abs().flatten()
                if k < len(flat):
                    threshold = torch.topk(flat, k).values[-1]
                    mask = delta.abs() >= threshold
                    trimmed_deltas.append(delta * mask.float())
                else:
                    trimmed_deltas.append(delta)
            deltas = torch.stack(trimmed_deltas)

        # Step 2: Elect sign - majority vote on sign
        signs = torch.sign(deltas)
        weighted_signs = (signs * task_weights.view(-1, *([1] * (deltas.dim() - 1)))).sum(dim=0)
        elected_sign = torch.sign(weighted_signs)

        # Step 3: Merge - average values that agree with elected sign
        mask = (signs == elected_sign.unsqueeze(0)).float()
        weighted_deltas = deltas * mask * task_weights.view(-1, *([1] * (deltas.dim() - 1)))

        # Sum and normalize
        merged_delta = weighted_deltas.sum(dim=0)
        normalizer = (mask * task_weights.view(-1, *([1] * (deltas.dim() - 1)))).sum(dim=0).clamp(min=1e-8)
        merged_delta = merged_delta / normalizer

        # Add back to base
        merged_tensor = base_tensor + merged_delta
        merged[key] = merged_tensor.to(first_tensor.dtype)

    return merged


def merge_checkpoints(
    checkpoint_paths: List[str],
    weights: Optional[List[float]] = None,
    method: str = "weighted_average",
    output_path: str = "merged_checkpoint.pt",
    base_checkpoint_path: Optional[str] = None,
    density: float = 0.5,
) -> str:
    """
    Main function to merge multiple checkpoints.

    Args:
        checkpoint_paths: List of paths to checkpoints to merge
        weights: Optional weights (default: equal weights)
        method: Merging method ("weighted_average" or "ties")
        output_path: Path to save merged checkpoint
        base_checkpoint_path: Base checkpoint for TIES merge
        density: TIES density parameter

    Returns:
        Path to merged checkpoint
    """
    # Load all checkpoints
    raw_checkpoints = [load_checkpoint(p) for p in checkpoint_paths]

    # Flatten nested checkpoint structures
    print("\nFlattening checkpoint structures...")
    checkpoints = [flatten_checkpoint(ckpt) for ckpt in raw_checkpoints]

    for i, (raw, flat) in enumerate(zip(raw_checkpoints, checkpoints)):
        print(f"  Checkpoint {i}: {len(raw)} top-level keys -> {len(flat)} flattened keys")

    # Default weights: proportional to dataset size or equal
    if weights is None:
        weights = [1.0] * len(checkpoints)

    print(f"\nMerging {len(checkpoints)} checkpoints using {method}")
    print(f"Weights: {weights}")

    # Keys to merge (LoRA and emotion-related)
    merge_patterns = ["lora", "emotion", "cross_attn"]

    if method == "weighted_average":
        merged_flat = weighted_average_merge(checkpoints, weights, merge_patterns)
    elif method == "ties":
        if base_checkpoint_path:
            base_raw = load_checkpoint(base_checkpoint_path)
            base = flatten_checkpoint(base_raw)
        else:
            # Use first checkpoint as pseudo-base
            print("Warning: No base checkpoint provided for TIES, using zeros")
            base = {}
        merged_flat = ties_merge(checkpoints, base, weights, density, merge_patterns)
    else:
        raise ValueError(f"Unknown merge method: {method}")

    # Unflatten back to original structure (use first checkpoint as template)
    merged = unflatten_checkpoint(merged_flat, raw_checkpoints[0])

    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save merged checkpoint
    print(f"\nSaving merged checkpoint to: {output_path}")
    torch.save(merged, output_path)

    # Print statistics
    lora_keys = get_lora_keys(merged_flat)
    print(f"\nMerged checkpoint statistics:")
    print(f"  Total keys (flattened): {len(merged_flat)}")
    print(f"  LoRA/Emotion keys: {len(lora_keys)}")

    # Calculate size
    total_params = sum(
        v.numel() for v in merged_flat.values() if torch.is_tensor(v)
    )
    print(f"  Total parameters: {total_params:,}")

    # Show some key names
    if lora_keys:
        print(f"\n  Sample keys merged:")
        for k in lora_keys[:5]:
            print(f"    - {k}")
        if len(lora_keys) > 5:
            print(f"    ... and {len(lora_keys) - 5} more")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple emotion LoRA checkpoints"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=[
            "checkpoints/emotion_lora_ravdess/checkpoint_early_stop.pt",
            "checkpoints/emotion_lora_cremad/checkpoint_early_stop.pt",
            "checkpoints/emotion_lora_iesc/checkpoint_early_stop.pt",
        ],
        help="Paths to checkpoints to merge",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated weights for each checkpoint (e.g., '0.15,0.78,0.07')",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["weighted_average", "ties"],
        default="weighted_average",
        help="Merging method",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/emotion_merged/checkpoint_merged.pt",
        help="Output path for merged checkpoint",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="Base checkpoint for TIES merge",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="TIES density parameter (0.0-1.0)",
    )
    parser.add_argument(
        "--auto-weights",
        action="store_true",
        help="Automatically compute weights based on dataset sizes",
    )

    args = parser.parse_args()

    # Parse weights
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
    elif args.auto_weights:
        # Weights proportional to dataset sizes
        # RAVDESS: 1440, CREMA-D: 7442, IESC: 600
        dataset_sizes = [1440, 7442, 600]
        total = sum(dataset_sizes)
        weights = [s / total for s in dataset_sizes]
        print(f"Auto-computed weights based on dataset sizes: {weights}")
    else:
        weights = None

    # Filter to existing checkpoints
    existing_checkpoints = []
    existing_weights = []
    for i, ckpt_path in enumerate(args.checkpoints):
        if os.path.exists(ckpt_path):
            existing_checkpoints.append(ckpt_path)
            if weights:
                existing_weights.append(weights[i])
        else:
            print(f"Warning: Checkpoint not found, skipping: {ckpt_path}")

    if not existing_checkpoints:
        print("Error: No valid checkpoints found!")
        return

    # Use existing weights or default
    final_weights = existing_weights if existing_weights else None

    # Merge
    output_path = merge_checkpoints(
        checkpoint_paths=existing_checkpoints,
        weights=final_weights,
        method=args.method,
        output_path=args.output,
        base_checkpoint_path=args.base_checkpoint,
        density=args.density,
    )

    print(f"\n✓ Successfully merged {len(existing_checkpoints)} checkpoints!")
    print(f"  Output: {output_path}")

    # Print usage example
    print("\n" + "=" * 60)
    print("Usage example:")
    print("=" * 60)
    print("""
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model with merged checkpoint
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Load merged emotion weights
model.load_emotion_checkpoint("checkpoints/emotion_merged/checkpoint_merged.pt")

# Generate with enhanced emotions
audio = model.generate(
    text="I am so excited to see you!",
    language_id="en",
    emotion="excited",
    emotion_intensity=1.2
)
""")


if __name__ == "__main__":
    main()
