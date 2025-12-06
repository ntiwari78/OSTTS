from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor

from .perceiver import Perceiver
from .t3_config import T3Config
from .emotion_cross_attention import EmotionCrossAttention


@dataclass
class T3Cond:
    """
    Dataclass container for conditioning information.

    Attributes:
        speaker_emb: Speaker embedding from voice encoder (B, 256)
        clap_emb: CLAP embedding (not yet implemented)
        cond_prompt_speech_tokens: Speech tokens from conditioning prompt
        cond_prompt_speech_emb: Speech embeddings from conditioning prompt
        emotion_embed: Emotion embedding vector (B, 64) - from EmotionEmbeddings
    """

    speaker_emb: Tensor
    clap_emb: Optional[Tensor] = None
    cond_prompt_speech_tokens: Optional[Tensor] = None
    cond_prompt_speech_emb: Optional[Tensor] = None
    emotion_embed: Optional[Tensor] = None  # (B, 64) emotion embedding

    def to(self, *, device=None, dtype=None):
        "Cast to a device and dtype. Dtype casting is ignored for long/int tensors."
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = type(v.view(-1)[0].item()) is not int
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self

    def save(self, fpath):
        torch.save(self.__dict__, fpath)

    @staticmethod
    def load(fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        # Handle legacy checkpoints that may have emotion_adv
        if 'emotion_adv' in kwargs:
            del kwargs['emotion_adv']
        return T3Cond(**kwargs)


class T3CondEnc(nn.Module):
    """
    Handle all non-text conditioning, like speaker embeddings / prompts, CLAP, emotion, etc.

    The emotion conditioning uses cross-attention to allow the emotion to modulate
    based on text context (when provided).
    """

    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp

        # Speaker embedding projection
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))

        # Emotion cross-attention conditioning
        self.emotion_cross_attn = EmotionCrossAttention(
            hidden_size=hp.n_channels,
            emotion_dim=hp.emotion_embed_dim,
            num_heads=hp.emotion_cross_attn_heads,
            num_query_tokens=hp.emotion_num_query_tokens,
        )

        # Perceiver resampler for speech prompts
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()

    def forward(
        self,
        cond: T3Cond,
        text_context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process conditioning inputs.

        Args:
            cond: T3Cond dataclass with conditioning tensors
            text_context: Optional text embeddings (B, L, n_channels) for cross-attention

        Returns:
            Concatenated conditioning embeddings (B, L_cond, n_channels)
        """
        # Validate
        assert (cond.cond_prompt_speech_tokens is None) == (cond.cond_prompt_speech_emb is None), \
            "no embeddings for cond_prompt_speech_tokens"

        # Speaker embedding projection
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]  # (B, 1, dim)
        empty = torch.zeros_like(cond_spkr[:, :0])  # (B, 0, dim)

        # CLAP (not yet implemented)
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)

        # Conditional prompt speech embeddings
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion cross-attention conditioning
        if cond.emotion_embed is not None:
            # Handle different input shapes
            emotion_embed = cond.emotion_embed
            if emotion_embed.dim() == 3:
                # (B, 1, emotion_dim) -> (B, emotion_dim)
                emotion_embed = emotion_embed.squeeze(1)

            # Handle batch size mismatch (e.g., when text_context has CFG batch size)
            # If text_context has larger batch size, expand emotion_embed to match
            if text_context is not None and text_context.size(0) > emotion_embed.size(0):
                expand_factor = text_context.size(0) // emotion_embed.size(0)
                emotion_embed = emotion_embed.repeat(expand_factor, 1)

            # Apply cross-attention (uses text_context if provided)
            cond_emotion = self.emotion_cross_attn(
                emotion_embed,
                context=text_context,
            )  # (B, num_query_tokens, n_channels)
        else:
            # No emotion conditioning - use empty
            cond_emotion = empty  # (B, 0, dim)

        # Handle batch size mismatch for concatenation
        # All tensors must have same batch size
        batch_size = cond_spkr.size(0)
        if cond_emotion.size(0) > batch_size:
            # cond_emotion has larger batch (e.g., from CFG), expand others to match
            batch_size = cond_emotion.size(0)
            if cond_spkr.size(0) < batch_size:
                cond_spkr = cond_spkr.expand(batch_size, -1, -1)
            if cond_clap.size(0) < batch_size:
                cond_clap = cond_clap.expand(batch_size, -1, -1)
            if cond_prompt_speech_emb.size(0) < batch_size:
                cond_prompt_speech_emb = cond_prompt_speech_emb.expand(batch_size, -1, -1)

        # Concat and return
        cond_embeds = torch.cat((
            cond_spkr,
            cond_clap,
            cond_prompt_speech_emb,
            cond_emotion,
        ), dim=1)

        return cond_embeds
