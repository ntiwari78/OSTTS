from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.emotion_embeddings import EmotionEmbeddings

# V0.4: Import intensity calibration
try:
    from .models.t3.modules.emotion_intensity_calibration import (
        get_calibrated_intensity,
        get_calibration_multiplier,
        CALIBRATED_INTENSITIES,
    )
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False
    CALIBRATED_INTENSITIES = {}


REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_embed
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        # Handle legacy checkpoints that may have emotion_adv
        if 'emotion_adv' in kwargs.get('t3', {}):
            del kwargs['t3']['emotion_adv']
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    """
    Multilingual Text-to-Speech model with emotion control.

    Features:
    - 23 language support
    - 11 emotion types with intensity control
    - Emotion blending/interpolation
    - Voice cloning via audio prompts

    Example:
        >>> model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        >>> # Basic usage
        >>> audio = model.generate("Hello world!", language_id="en", emotion="happy")
        >>> # With intensity
        >>> audio = model.generate("I'm excited!", language_id="en",
        ...                        emotion="excited", emotion_intensity=1.2)
        >>> # Emotion blending
        >>> audio = model.generate("Mixed feelings.", language_id="en",
        ...                        emotion_blend={"happy": 0.6, "sad": 0.4})
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

        # Initialize 64D emotion embeddings
        emotion_embed_dim = t3.hp.emotion_embed_dim if hasattr(t3.hp, 'emotion_embed_dim') else 64
        self.emotion_embeddings = EmotionEmbeddings(emotion_embed_dim=emotion_embed_dim)
        self.emotion_embeddings.to(device)

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    def get_supported_emotions(self) -> List[str]:
        """Return list of supported emotion types."""
        return self.emotion_embeddings.get_supported_emotions()

    def load_emotion_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """
        Load emotion-related weights from a fine-tuned or merged checkpoint.

        This method loads LoRA weights and emotion cross-attention parameters
        from a checkpoint file, enabling enhanced emotion capabilities.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            strict: If True, raise error on missing/unexpected keys

        Example:
            >>> model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            >>> model.load_emotion_checkpoint("checkpoints/emotion_merged/checkpoint_merged.pt")
            >>> audio = model.generate(text="Hello!", language_id="en", emotion="happy")
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading emotion checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Track loaded components
        loaded_components = []

        # Load emotion embeddings if present
        # First check for the wrapped format: emotion_embeddings_state_dict
        if "emotion_embeddings_state_dict" in checkpoint:
            self.emotion_embeddings.load_state_dict(checkpoint["emotion_embeddings_state_dict"], strict=False)
            loaded_components.append("emotion_embeddings")
        else:
            # Try unwrapped format: keys like "emotion_embeddings.weight", etc.
            emotion_embed_keys = [k for k in checkpoint.keys() if "emotion_embeddings" in k or k.startswith("embedding.")]
            if emotion_embed_keys:
                emotion_state = {}
                for key in emotion_embed_keys:
                    # Handle different key formats
                    if key.startswith("emotion_embeddings."):
                        new_key = key.replace("emotion_embeddings.", "")
                    else:
                        new_key = key
                    emotion_state[new_key] = checkpoint[key]

                if emotion_state:
                    self.emotion_embeddings.load_state_dict(emotion_state, strict=False)
                    loaded_components.append("emotion_embeddings")

        # Load T3 model weights (LoRA and emotion cross-attention)
        # First check for the wrapped format: t3_state_dict
        if "t3_state_dict" in checkpoint:
            missing, unexpected = self.t3.load_state_dict(checkpoint["t3_state_dict"], strict=False)
            if missing and strict:
                print(f"Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
            loaded_components.append("t3_lora_and_emotion")
        else:
            # Try unwrapped format: keys like "t3.layer.weight", etc.
            t3_keys = [k for k in checkpoint.keys() if k.startswith("t3.") or "cond_enc" in k or "lora" in k]
            if t3_keys:
                t3_state = {}
                for key in t3_keys:
                    # Remove "t3." prefix if present
                    if key.startswith("t3."):
                        new_key = key[3:]
                    else:
                        new_key = key
                    t3_state[new_key] = checkpoint[key]

                if t3_state:
                    missing, unexpected = self.t3.load_state_dict(t3_state, strict=False)
                    if missing and strict:
                        print(f"Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
                    loaded_components.append("t3_lora_and_emotion")

        # Also try loading full state dict directly (for some checkpoint formats)
        if not loaded_components:
            # Try loading as full model state
            try:
                missing, unexpected = self.t3.load_state_dict(checkpoint, strict=False)
                if not missing or len(missing) < len(checkpoint):
                    loaded_components.append("t3_full")
            except Exception:
                pass

        if loaded_components:
            print(f"✓ Loaded emotion checkpoint components: {loaded_components}")
        else:
            print("Warning: No compatible weights found in checkpoint")

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", map_location=device, weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        # Use strict=False to allow missing keys (new emotion cross-attention layers)
        missing_keys, unexpected_keys = t3.load_state_dict(t3_state, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in T3 model (will use random initialization): {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in T3 model state_dict: {unexpected_keys}")
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location=device, weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=device).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)

    def _get_emotion_embedding(
        self,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
        emotion_blend: Optional[Dict[str, float]] = None,
        use_calibration: bool = True,
    ) -> torch.Tensor:
        """
        Get emotion embedding based on provided parameters.

        Args:
            emotion: Single emotion name (e.g., "happy")
            emotion_intensity: Intensity multiplier (0.0 = neutral, 1.0 = full, >1.0 = exaggerated)
            emotion_blend: Dict of emotion names to weights for blending
            use_calibration: Whether to apply V0.4 intensity calibration (default: True)
                            Calibration adjusts intensity based on SER recognition patterns.
                            E.g., angry gets boosted to 1.5x for better recognition.

        Returns:
            Emotion embedding tensor (1, emotion_dim)
        """
        if emotion_blend is not None:
            # Blend multiple emotions
            # For blending, apply calibration to each emotion's contribution
            if use_calibration and HAS_CALIBRATION:
                calibrated_blend = {}
                for em, weight in emotion_blend.items():
                    calibration = get_calibration_multiplier(em)
                    # Apply calibration to the weight contribution
                    calibrated_blend[em] = weight * calibration
                # Renormalize weights
                total = sum(calibrated_blend.values())
                if total > 0:
                    calibrated_blend = {k: v / total for k, v in calibrated_blend.items()}
                emotion_blend = calibrated_blend
            return self.emotion_embeddings.interpolate_emotions(
                emotion_blend, device=self.device
            )
        elif emotion is not None:
            # Single emotion with intensity
            # V0.4: Apply calibration to intensity
            calibrated_intensity = emotion_intensity
            if use_calibration and HAS_CALIBRATION:
                calibrated_intensity = get_calibrated_intensity(
                    emotion=emotion,
                    user_intensity=emotion_intensity,
                    use_calibration=True,
                )
            return self.emotion_embeddings.get_emotion_embedding(
                emotion, intensity=calibrated_intensity, device=self.device
            )
        else:
            # Default to neutral
            return self.emotion_embeddings.get_emotion_embedding(
                "neutral", device=self.device
            )

    def prepare_conditionals(
        self,
        wav_fpath,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
        emotion_blend: Optional[Dict[str, float]] = None,
        use_calibration: bool = True,
    ):
        """
        Prepare conditioning data from a reference audio file.

        Args:
            wav_fpath: Path to reference audio file for voice cloning
            emotion: Emotion type (e.g., "happy", "sad"). Default: "neutral"
            emotion_intensity: Intensity of the emotion (0.0-1.5). Default: 1.0
            emotion_blend: Dict of emotion names to weights for blending multiple emotions
            use_calibration: Whether to apply V0.4 intensity calibration (default: True)
        """
        # Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        # Get emotion embedding (with V0.4 calibration)
        emotion_embed = self._get_emotion_embedding(
            emotion, emotion_intensity, emotion_blend, use_calibration=use_calibration
        )

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_embed=emotion_embed,
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        emotion: str = "neutral",
        emotion_intensity: float = 1.0,
        emotion_blend: Optional[Dict[str, float]] = None,
        use_calibration: bool = True,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate speech from text with emotion control.

        Args:
            text: Input text to synthesize
            language_id: Language code (e.g., "en", "hi", "zh")
            audio_prompt_path: Optional path to reference audio for voice cloning
            emotion: Emotion type. One of: neutral, happy, sad, angry, excited,
                    calm, surprised, fearful, disgusted, whisper, shout
            emotion_intensity: Emotion intensity (0.0 = neutral, 1.0 = full, >1.0 = exaggerated)
            emotion_blend: Dict of emotion names to weights for blending.
                          Example: {"happy": 0.7, "excited": 0.3}
            use_calibration: Whether to apply V0.4 intensity calibration (default: True).
                            Calibration automatically adjusts intensity based on SER analysis:
                            - angry: 1.5x boost (often misclassified as neutral)
                            - disgusted: 1.4x boost (needs clearer markers)
                            - calm: 0.8x reduction (to differentiate from neutral)
                            Set to False for raw intensity control.
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeated tokens
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold

        Returns:
            Audio tensor (1, num_samples) at 24kHz

        Examples:
            >>> # Basic generation (with V0.4 calibration)
            >>> audio = model.generate("Hello!", language_id="en", emotion="happy")

            >>> # With intensity control (calibration applied)
            >>> audio = model.generate("I'm so excited!", language_id="en",
            ...                        emotion="excited", emotion_intensity=1.3)

            >>> # Emotion blending
            >>> audio = model.generate("This is bittersweet.", language_id="en",
            ...                        emotion_blend={"happy": 0.4, "sad": 0.6})

            >>> # Disable calibration for raw intensity control
            >>> audio = model.generate("Hello!", language_id="en", emotion="angry",
            ...                        emotion_intensity=1.0, use_calibration=False)
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Validate emotion if not using blend
        if emotion_blend is None:
            supported_emotions = self.get_supported_emotions()
            if emotion.lower() not in supported_emotions:
                raise ValueError(
                    f"Unsupported emotion '{emotion}'. "
                    f"Supported emotions: {', '.join(supported_emotions)}"
                )
            emotion = emotion.lower()
        else:
            # Validate all emotions in blend
            supported_emotions = self.get_supported_emotions()
            for em in emotion_blend.keys():
                if em.lower() not in supported_emotions:
                    raise ValueError(
                        f"Unsupported emotion '{em}' in emotion_blend. "
                        f"Supported emotions: {', '.join(supported_emotions)}"
                    )
            # Normalize keys to lowercase
            emotion_blend = {k.lower(): v for k, v in emotion_blend.items()}

        # Prepare conditionals if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(
                audio_prompt_path,
                emotion=emotion if emotion_blend is None else None,
                emotion_intensity=emotion_intensity,
                emotion_blend=emotion_blend,
                use_calibration=use_calibration,
            )
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update emotion embedding if needed (with V0.4 calibration)
        _cond: T3Cond = self.conds.t3
        new_emotion_embed = self._get_emotion_embedding(
            emotion if emotion_blend is None else None,
            emotion_intensity,
            emotion_blend,
            use_calibration=use_calibration,
        )

        # Check if emotion changed
        needs_update = (
            _cond.emotion_embed is None or
            not torch.equal(_cond.emotion_embed, new_emotion_embed)
        )

        if needs_update:
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_embed=new_emotion_embed,
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
