#!/usr/bin/env python3
"""
Test script to verify loss calculation is working correctly.
"""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from train_emotion_lora import HindiEmotionDataset
from torch.utils.data import DataLoader
from chatterbox.models.s3tokenizer import S3_SR
import librosa

device = 'cpu'
print("Loading model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

emotion_mapping = {
    'emotion_happy': 'happy',
    'emotion_sad': 'sad',
    'emotion_angry': 'angry',
    'emotion_neutral': 'neutral',
    'emotion_fearful': 'fearful',
}

print("Loading dataset...")
dataset = HindiEmotionDataset('data/hindi_emotions', model.tokenizer, emotion_mapping)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Get one batch
batch = next(iter(dataloader))
print(f"\nBatch info:")
print(f"  Text tokens shape: {batch['text_tokens'].shape}")
print(f"  Text length: {batch['text_length']}")
print(f"  Emotion: {batch['emotion']}")

# Prepare inputs like in train_step
audio_16k = batch["audio_16k"].to(device)
text_tokens = batch["text_tokens"].to(device)
emotions = batch["emotion"]
batch_size = audio_16k.shape[0]

# Add start and stop tokens
start_token = model.t3.hp.start_text_token
stop_token = model.t3.hp.stop_text_token

text_tokens_with_special = []
text_token_lens = []
for i in range(batch_size):
    tokens = text_tokens[i]
    actual_len = batch["text_length"][i]
    actual_tokens = tokens[:actual_len]
    
    tokens_with_special = torch.cat([
        torch.tensor([start_token], device=device),
        actual_tokens,
        torch.tensor([stop_token], device=device)
    ])
    text_tokens_with_special.append(tokens_with_special)
    text_token_lens.append(len(tokens_with_special))

max_text_len = max(len(t) for t in text_tokens_with_special)
text_tokens_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
for i, tokens in enumerate(text_tokens_with_special):
    text_tokens_padded[i, :len(tokens)] = tokens

text_token_lens = torch.tensor(text_token_lens, device=device)

# Get emotion embeddings
emotion_embeds_list = []
for emotion in emotions:
    emotion_embed = model.emotion_embeddings.get_emotion_embedding(emotion).to(device)
    emotion_embed = emotion_embed.squeeze(0)
    emotion_embeds_list.append(emotion_embed)
emotion_embeds = torch.stack(emotion_embeds_list).unsqueeze(1)

# Extract speaker embeddings
speaker_embeds_list = []
for i in range(batch_size):
    audio_single = audio_16k[i].cpu().numpy()
    try:
        ve_embed = model.ve.embeds_from_wavs([audio_single], sample_rate=S3_SR)
        ve_embed = torch.from_numpy(ve_embed).to(device)
        speaker_embeds_list.append(ve_embed.mean(dim=0, keepdim=True))
    except Exception as e:
        print(f"Warning: Voice encoder failed: {e}")
        speaker_embeds_list.append(torch.zeros(1, 256, device=device))

speaker_emb = torch.stack(speaker_embeds_list)

# Tokenize audio
audio_list = [audio_16k[i].cpu().numpy() for i in range(batch_size)]
speech_tokens_list = []
speech_token_lens_list = []

with torch.no_grad():
    for audio_single in audio_list:
        try:
            tokens, lens = model.s3gen.tokenizer([audio_single], max_len=None)
            speech_tokens_list.append(tokens.squeeze(0).cpu())
            speech_token_lens_list.append(lens.item() if hasattr(lens, 'item') else int(lens))
        except Exception as e:
            print(f"Warning: Tokenization failed: {e}")
            sys.exit(1)

max_speech_len = max(len(t) for t in speech_tokens_list)
speech_tokens = torch.zeros(batch_size, max_speech_len, dtype=torch.long, device=device)
for i, tokens in enumerate(speech_tokens_list):
    if len(tokens) > 0:
        if tokens.dim() > 1:
            tokens = tokens.flatten()
        tokens = tokens[:max_speech_len]
        speech_tokens[i, :len(tokens)] = tokens.to(device)
        speech_token_lens_list[i] = min(len(tokens), max_speech_len)

speech_token_lens = torch.tensor(speech_token_lens_list, device=device)

# Create conditionals
t3_cond = T3Cond(
    speaker_emb=speaker_emb,
    cond_prompt_speech_tokens=None,
    emotion_adv=None,
    emotion_embed=emotion_embeds,
)
t3_cond = t3_cond.to(device=device)

print(f"\nInput shapes:")
print(f"  text_tokens_padded: {text_tokens_padded.shape}")
print(f"  text_token_lens: {text_token_lens}")
print(f"  speech_tokens: {speech_tokens.shape}")
print(f"  speech_token_lens: {speech_token_lens}")

# Forward pass
model.t3.train()
out = model.t3.forward(
    t3_cond=t3_cond,
    text_tokens=text_tokens_padded,
    text_token_lens=text_token_lens,
    speech_tokens=speech_tokens,
    speech_token_lens=speech_token_lens,
    training=True,
)

print(f"\nOutput shapes:")
print(f"  text_logits: {out.text_logits.shape}")
print(f"  speech_logits: {out.speech_logits.shape}")

# Reshape for loss calculation
text_logits_flat = out.text_logits.view(-1, out.text_logits.size(-1))
speech_logits_flat = out.speech_logits.view(-1, out.speech_logits.size(-1))

# Create masks
IGNORE_ID = -100
len_text = text_tokens_padded.size(1)
len_speech = speech_tokens.size(1)

mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]
mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]
masked_text = text_tokens_padded.masked_fill(mask_text, IGNORE_ID)
masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)

text_targets_flat = masked_text.view(-1)
speech_targets_flat = masked_speech.view(-1)

print(f"\nLoss calculation inputs:")
print(f"  text_logits_flat: {text_logits_flat.shape}, min={text_logits_flat.min().item():.2f}, max={text_logits_flat.max().item():.2f}")
print(f"  text_targets_flat: {text_targets_flat.shape}, unique={torch.unique(text_targets_flat).numel()}, valid={(text_targets_flat != IGNORE_ID).sum().item()}")
print(f"  text_targets range: [{text_targets_flat[text_targets_flat != IGNORE_ID].min().item()}, {text_targets_flat[text_targets_flat != IGNORE_ID].max().item()}]")
print(f"  speech_logits_flat: {speech_logits_flat.shape}, min={speech_logits_flat.min().item():.2f}, max={speech_logits_flat.max().item():.2f}")
print(f"  speech_targets_flat: {speech_targets_flat.shape}, unique={torch.unique(speech_targets_flat).numel()}, valid={(speech_targets_flat != IGNORE_ID).sum().item()}")

# Compute losses
loss_text = F.cross_entropy(text_logits_flat, text_targets_flat, ignore_index=IGNORE_ID)
loss_speech = F.cross_entropy(speech_logits_flat, speech_targets_flat, ignore_index=IGNORE_ID)

print(f"\nLosses:")
print(f"  loss_text: {loss_text.item():.4f}")
print(f"  loss_speech: {loss_speech.item():.4f}")
print(f"  combined: {loss_text.item() + 2.0 * loss_speech.item():.4f}")

# Check if losses are valid
if loss_text.item() == 0.0 or loss_speech.item() == 0.0:
    print(f"\n⚠️  WARNING: One or both losses are 0.0!")
    print(f"  This could mean:")
    print(f"    1. All predictions are perfect (unlikely)")
    print(f"    2. Targets are out of vocabulary range")
    print(f"    3. Logits are all the same (model not initialized properly)")
    
    # Check vocabulary ranges
    print(f"\n  Vocabulary checks:")
    print(f"    Text vocab size: {model.t3.hp.text_tokens_dict_size}")
    print(f"    Speech vocab size: {model.t3.hp.speech_tokens_dict_size}")
    print(f"    Text targets max: {text_targets_flat[text_targets_flat != IGNORE_ID].max().item() if (text_targets_flat != IGNORE_ID).any() else 'N/A'}")
    print(f"    Speech targets max: {speech_targets_flat[speech_targets_flat != IGNORE_ID].max().item() if (speech_targets_flat != IGNORE_ID).any() else 'N/A'}")
else:
    print(f"\n✓ Losses are valid and non-zero!")

