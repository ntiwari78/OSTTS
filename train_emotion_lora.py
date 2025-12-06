#!/usr/bin/env python3
"""
Fine-tune Chatterbox T3 model with LoRA/Adapter for emotion control.

This script fine-tunes the T3 model on emotion-labeled audio data (RAVDESS, CREMA-D, etc.)
using LoRA (Low-Rank Adaptation) or Adapter layers for efficient training.

Supported datasets:
- RAVDESS: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgusted, surprised)
- CREMA-D: 6 emotions (neutral, happy, sad, angry, fearful, disgusted)
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import os

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.lora_adapter import (
    apply_lora_to_linear,
    add_adapters_to_transformer,
    LoRALinear,
    AdapterLayer,
)
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.models.tokenizers import MTLTokenizer


class EmotionDataset(Dataset):
    """
    Dataset for emotion-labeled audio (RAVDESS, CREMA-D, or custom).

    Expected structure:
    data/
        emotion_happy/
            audio1.wav
            audio2.wav
        emotion_sad/
            audio1.wav
            ...
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: MTLTokenizer,
        emotion_mapping: Dict[str, str],
        language_id: str = "en",
        max_audio_length: float = 10.0,
        sample_rate: int = S3GEN_SR,
    ):
        """
        Args:
            data_dir: Directory containing emotion-labeled audio files
            tokenizer: Text tokenizer
            emotion_mapping: Mapping from emotion folder names to emotion types
            language_id: Language code for text tokenization (default: "en" for English)
            max_audio_length: Maximum audio length in seconds
            sample_rate: Target sample rate
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.emotion_mapping = emotion_mapping
        self.language_id = language_id
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate

        # Load data samples
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all audio samples with their emotion labels."""
        samples = []
        
        for emotion_folder, emotion_type in self.emotion_mapping.items():
            emotion_dir = self.data_dir / emotion_folder
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist, skipping")
                continue
            
            # Look for audio files
            audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(list(emotion_dir.glob(f"*{ext}")))
            
            for audio_file in audio_files:
                # Try to extract text from filename or metadata
                # Format: "text_emotion.wav" or use metadata
                text = self._extract_text_from_filename(audio_file)
                
                samples.append({
                    "audio_path": str(audio_file),
                    "emotion": emotion_type,
                    "text": text,
                })
        
        return samples
    
    def _extract_text_from_filename(self, audio_file: Path) -> str:
        """Extract text from filename or return default placeholder text.

        For RAVDESS/CREMA-D datasets, actual transcripts are not in filenames,
        so we use placeholder text that represents typical spoken content.
        """
        # RAVDESS uses two statements:
        # "Kids are talking by the door" and "Dogs are sitting by the door"
        # CREMA-D uses various sentences

        # Use a generic placeholder for training
        # The emotion embedding should learn to condition regardless of text
        return "This is a test sentence."
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio at original sample rate first
        audio, orig_sr = librosa.load(sample["audio_path"], sr=None)
        
        # Resample to 16kHz for tokenization (S3Gen tokenizer needs 16kHz)
        audio_16k = librosa.resample(audio, orig_sr=orig_sr, target_sr=S3_SR)
        
        # Also resample to 24kHz for reference (S3Gen uses 24kHz)
        audio_24k = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        
        # Truncate/pad to max length
        max_samples_16k = int(self.max_audio_length * S3_SR)
        max_samples_24k = int(self.max_audio_length * self.sample_rate)
        
        if len(audio_16k) > max_samples_16k:
            audio_16k = audio_16k[:max_samples_16k]
        else:
            audio_16k = np.pad(audio_16k, (0, max_samples_16k - len(audio_16k)))
        
        if len(audio_24k) > max_samples_24k:
            audio_24k = audio_24k[:max_samples_24k]
        else:
            audio_24k = np.pad(audio_24k, (0, max_samples_24k - len(audio_24k)))
        
        # Tokenize text
        text_tokens = self.tokenizer.text_to_tokens(
            sample["text"],
            language_id=self.language_id
        )
        text_tokens = text_tokens.squeeze(0)  # Remove batch dimension
        
        return {
            "audio_16k": torch.from_numpy(audio_16k).float(),  # For tokenization
            "audio_24k": torch.from_numpy(audio_24k).float(),  # For reference
            "text_tokens": text_tokens,
            "text_length": text_tokens.shape[0],  # Actual text length
            "emotion": sample["emotion"],
            "text": sample["text"],
        }


def download_hindi_emotion_data(output_dir: str = "data/hindi_emotions"):
    """
    Download or prepare Hindi emotion audio dataset.
    
    This function should be customized based on available datasets.
    Options:
    1. IndiTTS dataset
    2. IndicTTS dataset
    3. Custom dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up Hindi emotion dataset in {output_dir}")
    print("\nTo download datasets, you can:")
    print("1. Use IndicTTS: https://www.iitm.ac.in/donlab/tts/")
    print("2. Use IndiTTS: https://github.com/ai4bharat/IndicTTS")
    print("3. Create your own dataset with emotion labels")
    print("\nExpected structure:")
    print("  data/hindi_emotions/")
    print("    emotion_happy/")
    print("      *.wav")
    print("    emotion_sad/")
    print("      *.wav")
    print("    ...")
    
    # Create example structure
    for emotion in ["happy", "sad", "angry", "neutral", "excited"]:
        (output_path / f"emotion_{emotion}").mkdir(exist_ok=True)
    
    return output_path


def setup_model_with_lora(
    model: ChatterboxMultilingualTTS,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    use_adapter: bool = False,
    adapter_size: int = 64,
    freeze_base: bool = True,
) -> ChatterboxMultilingualTTS:
    """
    Apply LoRA or Adapter to the T3 model.

    Args:
        model: ChatterboxMultilingualTTS model
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        use_adapter: Whether to use adapters instead of LoRA
        adapter_size: Adapter bottleneck size
        freeze_base: Whether to freeze base model weights (recommended for LoRA)

    Returns:
        Model with LoRA/Adapter applied
    """
    print("Setting up model for emotion fine-tuning...")

    # IMPORTANT: Freeze all model parameters first
    if freeze_base:
        print("Freezing base model parameters...")
        for param in model.t3.parameters():
            param.requires_grad = False
        for param in model.s3gen.parameters():
            param.requires_grad = False
        for param in model.ve.parameters():
            param.requires_grad = False
        for param in model.emotion_embeddings.parameters():
            param.requires_grad = False

    if use_adapter:
        # Add adapters to transformer layers
        add_adapters_to_transformer(
            model.t3.tfmr,
            adapter_size=adapter_size,
        )
        print(f"Added adapters (size={adapter_size}) to transformer layers")
    else:
        # Apply LoRA to attention and feed-forward layers
        apply_lora_to_linear(
            model.t3.tfmr,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            rank=lora_rank,
            alpha=lora_alpha,
        )
        print(f"Applied LoRA (rank={lora_rank}, alpha={lora_alpha}) to transformer layers")

        # Enable training for LoRA parameters specifically
        lora_param_count = 0
        for name, param in model.t3.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                lora_param_count += param.numel()
        print(f"Enabled {lora_param_count:,} LoRA parameters for training")

    # Fine-tune emotion cross-attention layers (these are new, not pre-trained)
    if hasattr(model.t3.cond_enc, "emotion_cross_attn"):
        print("Enabling training for emotion cross-attention layers")
        for param in model.t3.cond_enc.emotion_cross_attn.parameters():
            param.requires_grad = True

    # Make emotion embeddings trainable (important!)
    print("Enabling training for emotion embeddings (64D)")
    for param in model.emotion_embeddings.parameters():
        param.requires_grad = True

    return model


def train_step(
    model: ChatterboxMultilingualTTS,
    batch: Dict,
    device: str,
    optimizer: torch.optim.Optimizer,
    batch_idx: int = 0,
) -> float:
    """Perform one training step with actual TTS loss."""
    model.t3.train()
    model.s3gen.eval()  # Freeze S3Gen during T3 training
    model.ve.eval()  # Freeze voice encoder
    
    optimizer.zero_grad()
    
    # Prepare inputs
    audio_16k = batch["audio_16k"].to(device)  # 16kHz for tokenization
    text_tokens = batch["text_tokens"].to(device)
    emotions = batch["emotion"]
    batch_size = audio_16k.shape[0]
    
    # Add start and stop tokens to text tokens
    # T3 expects start_text_token at beginning and stop_text_token at end
    start_token = model.t3.hp.start_text_token
    stop_token = model.t3.hp.stop_text_token
    
    # Pad text tokens with start/stop tokens
    text_tokens_with_special = []
    text_token_lens = []
    for i in range(batch_size):
        tokens = text_tokens[i]
        # Remove padding to get actual length
        actual_len = batch["text_length"][i]
        actual_tokens = tokens[:actual_len]
        
        # Add start and stop tokens
        tokens_with_special = torch.cat([
            torch.tensor([start_token], device=device),
            actual_tokens,
            torch.tensor([stop_token], device=device)
        ])
        text_tokens_with_special.append(tokens_with_special)
        text_token_lens.append(len(tokens_with_special))
    
    # Pad to same length
    max_text_len = max(len(t) for t in text_tokens_with_special)
    text_tokens_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
    for i, tokens in enumerate(text_tokens_with_special):
        text_tokens_padded[i, :len(tokens)] = tokens
    
    text_token_lens = torch.tensor(text_token_lens, device=device)
    
    # Get emotion embeddings
    emotion_embeds_list = []
    for emotion in emotions:
        emotion_embed = model.emotion_embeddings.get_emotion_embedding(emotion).to(device)
        # emotion_embed is (1, emotion_dim) from get_emotion_embedding
        # Squeeze to (emotion_dim,) then we'll stack
        emotion_embed = emotion_embed.squeeze(0)  # (emotion_dim,)
        emotion_embeds_list.append(emotion_embed)
    # Stack to (B, emotion_dim), then add sequence dimension
    emotion_embeds = torch.stack(emotion_embeds_list)  # (B, emotion_dim)
    emotion_embeds = emotion_embeds.unsqueeze(1)  # (B, 1, emotion_dim)
    
    # Extract speaker embeddings from audio (16kHz for voice encoder)
    # Move to CPU for voice encoder to avoid MPS issues
    speaker_embeds_list = []
    for i in range(batch_size):
        audio_single = audio_16k[i].cpu().numpy()
        # Voice encoder expects list of wavs
        try:
            ve_embed = model.ve.embeds_from_wavs([audio_single], sample_rate=S3_SR)
            ve_embed = torch.from_numpy(ve_embed).to(device)
            speaker_embeds_list.append(ve_embed.mean(dim=0, keepdim=True))  # (1, 256)
        except Exception as e:
            # Fallback to zero embedding if voice encoder fails
            print(f"Warning: Voice encoder failed, using zero embedding: {e}")
            speaker_embeds_list.append(torch.zeros(1, 256, device=device))
    
    speaker_emb = torch.stack(speaker_embeds_list)  # (B, 1, 256)
    
    # Tokenize audio to speech tokens using S3Gen tokenizer
    # S3Tokenizer expects list of numpy arrays (16kHz audio)
    audio_list = [audio_16k[i].cpu().numpy() for i in range(batch_size)]
    speech_tokens_list = []
    speech_token_lens_list = []
    
    with torch.no_grad():  # Tokenization doesn't need gradients
        for audio_single in audio_list:
            try:
                # Tokenize (S3Tokenizer.forward expects list of wavs)
                tokens, lens = model.s3gen.tokenizer([audio_single], max_len=None)
                # tokens is (1, T), lens is scalar tensor
                speech_tokens_list.append(tokens.squeeze(0).cpu())  # Remove batch dim, move to CPU
                speech_token_lens_list.append(lens.item() if hasattr(lens, 'item') else int(lens))
            except Exception as e:
                # Skip this sample if tokenization fails
                print(f"Warning: Tokenization failed, skipping sample: {e}")
                return 0.0  # Return zero loss to skip this batch
    
    # Pad speech tokens to same length
    if len(speech_tokens_list) == 0:
        return 0.0  # Skip if no valid tokens
    
    max_speech_len = max(len(t) for t in speech_tokens_list)
    if max_speech_len == 0:
        return 0.0  # Skip if empty
    
    # Ensure tokens are 1D tensors
    speech_tokens = torch.zeros(batch_size, max_speech_len, dtype=torch.long, device=device)
    for i, tokens in enumerate(speech_tokens_list):
        if len(tokens) > 0:
            # Ensure tokens is 1D
            if tokens.dim() > 1:
                tokens = tokens.flatten()
            tokens = tokens[:max_speech_len]  # Truncate if too long
            speech_tokens[i, :len(tokens)] = tokens.to(device)
            speech_token_lens_list[i] = min(len(tokens), max_speech_len)
    
    speech_token_lens = torch.tensor(speech_token_lens_list, device=device)
    
    # Create conditionals (note: emotion_adv removed in new API)
    t3_cond = T3Cond(
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=None,
        emotion_embed=emotion_embeds,  # (B, 1, 64) - 64D emotion embeddings
    )
    t3_cond = t3_cond.to(device=device)  # Use keyword argument
    
    # Calculate actual TTS loss using T3 model
    try:
        # Ensure emotion embeddings are in training mode
        model.emotion_embeddings.train()
        
        # Debug: Check input shapes
        if batch_idx == 0:  # Only print for first batch
            print(f"\nDebug - Batch shapes:")
            print(f"  text_tokens: {text_tokens_padded.shape}, lens: {text_token_lens}")
            print(f"  speech_tokens: {speech_tokens.shape}, lens: {speech_token_lens}")
            print(f"  emotion_embeds: {emotion_embeds.shape}")
            print(f"  speaker_emb: {speaker_emb.shape}")
        
        # T3 expects text_token_lens.max() == text_tokens.size(1)
        # This is checked in t3.loss() at line 184
        # We need to ensure this matches
        assert text_token_lens.max() == text_tokens_padded.size(1), \
            f"text_token_lens.max()={text_token_lens.max()} != text_tokens.size(1)={text_tokens_padded.size(1)}"
        assert speech_token_lens.max() == speech_tokens.size(1), \
            f"speech_token_lens.max()={speech_token_lens.max()} != speech_tokens.size(1)={speech_tokens.size(1)}"
        
        # Call forward to check output shapes before loss
        if batch_idx == 0:
            with torch.no_grad():
                out = model.t3.forward(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens_padded,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens,
                    training=True,
                )
                print(f"  text_logits shape: {out.text_logits.shape}")
                print(f"  speech_logits shape: {out.speech_logits.shape}")
                print(f"  text_tokens_padded shape: {text_tokens_padded.shape}")
                print(f"  speech_tokens shape: {speech_tokens.shape}")
        
        # Calculate loss manually to handle shape issues
        # T3's loss function has issues with F.cross_entropy and 3D tensors
        # We'll compute the loss manually
        out = model.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens_padded,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )
        
        # Reshape logits and targets for cross_entropy
        # logits: (B, N, C) -> (B*N, C)
        # targets: (B, N) -> (B*N,)
        text_logits_flat = out.text_logits.view(-1, out.text_logits.size(-1))  # (B*N, C)
        speech_logits_flat = out.speech_logits.view(-1, out.speech_logits.size(-1))  # (B*N, C)
        
        # Create masks
        IGNORE_ID = -100
        device = text_logits_flat.device
        len_text = text_tokens_padded.size(1)
        len_speech = speech_tokens.size(1)
        
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens_padded.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        
        text_targets_flat = masked_text.view(-1)  # (B*N,)
        speech_targets_flat = masked_speech.view(-1)  # (B*N,)
        
        # Debug: Check logits and targets before loss calculation
        if batch_idx < 3:
            print(f"\nBatch {batch_idx} - Loss calculation debug:")
            print(f"  text_logits_flat: min={text_logits_flat.min().item():.2f}, max={text_logits_flat.max().item():.2f}, mean={text_logits_flat.mean().item():.2f}")
            print(f"  text_targets_flat: unique values={torch.unique(text_targets_flat).numel()}, min={text_targets_flat.min().item()}, max={text_targets_flat.max().item()}")
            print(f"  text_targets_flat shape: {text_targets_flat.shape}, IGNORE_ID count: {(text_targets_flat == IGNORE_ID).sum().item()}")
            print(f"  Valid targets (not IGNORE_ID): {(text_targets_flat != IGNORE_ID).sum().item()}")
            print(f"  speech_logits_flat: min={speech_logits_flat.min().item():.2f}, max={speech_logits_flat.max().item():.2f}")
            print(f"  speech_targets_flat: unique values={torch.unique(speech_targets_flat).numel()}, min={speech_targets_flat.min().item()}, max={speech_targets_flat.max().item()}")
            print(f"  Valid speech targets: {(speech_targets_flat != IGNORE_ID).sum().item()}")
        
        # Compute losses
        loss_text = F.cross_entropy(text_logits_flat, text_targets_flat, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(speech_logits_flat, speech_targets_flat, ignore_index=IGNORE_ID)
        
        # Debug: Print loss components for first few batches
        if batch_idx < 3:
            print(f"  loss_text={loss_text.item():.4f}, loss_speech={loss_speech.item():.4f}")
        
        # Check for NaN
        if torch.isnan(loss_text) or torch.isnan(loss_speech):
            print(f"Warning: NaN loss - text: {loss_text.item()}, speech: {loss_speech.item()}")
            return 0.0
        
        # Combined loss (weighted sum)
        # Speech loss is more important for TTS
        loss = loss_text + 2.0 * loss_speech
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss (NaN/Inf): {loss.item()}, skipping batch")
            return 0.0
        
        # Only reject if both components are exactly 0 (which shouldn't happen in real training)
        if loss_text.item() == 0.0 and loss_speech.item() == 0.0:
            if batch_idx < 10:  # Only print first few
                print(f"Warning: Both losses are 0.0 (batch {batch_idx}), this might indicate an issue")
            return 0.0
        
        # Backward pass
        loss.backward()
        
        # Get trainable parameters from optimizer
        trainable_params = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
        
        # Check if gradients are flowing
        grad_norm = 0.0
        for param in trainable_params:
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if grad_norm == 0.0:
            if batch_idx < 5:  # Only print first few
                print(f"Warning: Zero gradient norm (batch {batch_idx}), loss: {loss.item():.4f}")
            optimizer.zero_grad()  # Clear gradients
            return 0.0
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            trainable_params, 
            max_norm=1.0
        )
        
        optimizer.step()
        
        return loss.item()
    except Exception as e:
        print(f"Error in training step: {e}")
        import traceback
        traceback.print_exc()
        # Return a dummy loss to continue
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Fine-tune T3 model with LoRA/Adapter for emotions")
    parser.add_argument("--data_dir", type=str, default="data/combined_emotions",
                       help="Directory containing emotion-labeled audio")
    parser.add_argument("--output_dir", type=str, default="checkpoints/emotion_lora",
                       help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA alpha")
    parser.add_argument("--use_adapter", action="store_true",
                       help="Use adapters instead of LoRA")
    parser.add_argument("--adapter_size", type=int, default=64,
                       help="Adapter bottleneck size")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--download_data", action="store_true",
                       help="Download/setup data directory")
    parser.add_argument("--language", type=str, default="en",
                       help="Language code for text tokenization (en, hi, etc.)")
    parser.add_argument("--early_stop_loss", type=float, default=0.4,
                       help="Stop training when average loss falls below this threshold")
    parser.add_argument("--early_stop_patience", type=int, default=50,
                       help="Number of consecutive batches below threshold before stopping")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Download/setup data
    if args.download_data:
        download_hindi_emotion_data(args.data_dir)
        print("\nPlease add your Hindi emotion audio files to the data directory")
        print("Then run the script again without --download_data")
        return
    
    # Load model
    print("Loading model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Apply LoRA/Adapter
    model = setup_model_with_lora(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_adapter=args.use_adapter,
        adapter_size=args.adapter_size,
    )
    # Move model components to device
    model.t3.to(device)
    model.s3gen.to(device)
    model.ve.to(device)
    model.emotion_embeddings.to(device)
    
    # Setup dataset
    # Combined RAVDESS + CREMA-D emotions
    emotion_mapping = {
        "emotion_angry": "angry",
        "emotion_calm": "calm",
        "emotion_disgusted": "disgusted",
        "emotion_fearful": "fearful",
        "emotion_happy": "happy",
        "emotion_neutral": "neutral",
        "emotion_sad": "sad",
        "emotion_surprised": "surprised",
        # Additional emotions (not in RAVDESS/CREMA-D but supported)
        "emotion_excited": "excited",
        "emotion_whisper": "whisper",
        "emotion_shout": "shout",
    }

    dataset = EmotionDataset(
        data_dir=args.data_dir,
        tokenizer=model.tokenizer,
        emotion_mapping=emotion_mapping,
        language_id=args.language,
    )
    
    if len(dataset) == 0:
        print(f"Error: No data found in {args.data_dir}")
        print("Run with --download_data to setup data directory structure")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Mac compatibility
    )
    
    # Setup optimizer (only train LoRA/Adapter parameters)
    trainable_params = []
    
    # Collect trainable parameters from model components
    # Only collect from T3 and emotion_embeddings (S3Gen should be frozen)
    for component_name, component in [
        ("t3", model.t3),
        ("emotion_embeddings", model.emotion_embeddings),
    ]:
        for name, param in component.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                if len(trainable_params) <= 20:  # Only print first 20 to avoid spam
                    print(f"Training parameter: {component_name}.{name}")
    
    if len(trainable_params) > 20:
        print(f"... and {len(trainable_params) - 20} more trainable parameters")
    
    if len(trainable_params) == 0:
        print("Warning: No trainable parameters found! Check LoRA/Adapter setup.")
        return
    
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping: loss < {args.early_stop_loss} for {args.early_stop_patience} consecutive batches")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Early stopping tracking
    early_stop_counter = 0
    early_stopped = False

    for epoch in range(args.epochs):
        if early_stopped:
            break

        total_loss = 0.0
        valid_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                loss = train_step(model, batch, device, optimizer, batch_idx=batch_idx)
                if loss > 0.0:  # Only count valid losses
                    total_loss += loss
                    valid_batches += 1

                    # Early stopping check
                    if loss < args.early_stop_loss:
                        early_stop_counter += 1
                        if early_stop_counter >= args.early_stop_patience:
                            print(f"\n\nEarly stopping triggered! Loss {loss:.4f} < {args.early_stop_loss} for {args.early_stop_patience} consecutive batches")
                            early_stopped = True
                            break
                    else:
                        early_stop_counter = 0  # Reset counter

                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}" if loss > 0 else "skip",
                    "valid": valid_batches,
                    "es": f"{early_stop_counter}/{args.early_stop_patience}"
                })
            except Exception as e:
                if batch_idx % 100 == 0 or batch_idx < 5:  # Print first 5 and every 100th
                    print(f"\nError in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f} (from {valid_batches}/{len(dataloader)} batches)")
        else:
            print(f"\nEpoch {epoch+1} - No valid batches processed!")
            print("Check data loading and tokenization")
            continue

        # Save checkpoint (save individual components)
        checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1}.pt"
        checkpoint = {
            "epoch": epoch + 1,
            "t3_state_dict": model.t3.state_dict(),
            "emotion_embeddings_state_dict": model.emotion_embeddings.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if early_stopped:
            # Save final checkpoint with early_stop marker
            final_path = output_path / "checkpoint_early_stop.pt"
            torch.save(checkpoint, final_path)
            print(f"Saved early stop checkpoint: {final_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()

