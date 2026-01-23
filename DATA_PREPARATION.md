# Data Preparation for Emotion Fine-tuning

This document explains how to prepare emotion-labeled audio datasets for fine-tuning the Chatterbox TTS emotion system. It covers the three datasets used (RAVDESS, CREMA-D, IESC) and explains each script in detail.

## Table of Contents
- [Overview](#overview)
- [Dataset Requirements](#dataset-requirements)
- [Datasets Used](#datasets-used)
- [Data Preparation Scripts](#data-preparation-scripts)
- [Final Directory Structure](#final-directory-structure)
- [Quick Start](#quick-start)

---

## Overview

The emotion fine-tuning system requires audio data organized by emotion type. Each emotion should have its own folder containing audio files in WAV format at 24kHz sample rate.

### Requirements Summary

| Requirement | Value | Reason |
|-------------|-------|--------|
| Audio format | WAV (mono) | Compatible with librosa/soundfile |
| Sample rate | 24kHz | S3Gen model requirement |
| Max duration | 10 seconds | Memory constraints during training |
| Organization | `emotion_*/` folders | Dataset loader expectation |
| Minimum samples | ~50 per emotion | Sufficient for LoRA fine-tuning |

---

## Dataset Requirements

### Expected Directory Structure

```
data/
├── ravdess_emotions/           # RAVDESS dataset (English)
│   ├── emotion_angry/
│   │   ├── ravdess_03-01-05-01-01-01-01.wav
│   │   └── ...
│   ├── emotion_calm/
│   ├── emotion_disgusted/
│   ├── emotion_fearful/
│   ├── emotion_happy/
│   ├── emotion_neutral/
│   ├── emotion_sad/
│   ├── emotion_surprised/
│   └── metadata.json
│
├── cremad_emotions/            # CREMA-D dataset (English)
│   ├── emotion_angry/
│   │   ├── cremad_1001_DFA_ANG_XX.wav
│   │   └── ...
│   ├── emotion_disgusted/
│   ├── emotion_fearful/
│   ├── emotion_happy/
│   ├── emotion_neutral/
│   ├── emotion_sad/
│   └── metadata.json
│
└── hindi_emotions/             # IESC dataset (Hindi)
    ├── emotion_angry/
    │   ├── Speaker-1_audio1.wav
    │   └── ...
    ├── emotion_happy/
    ├── emotion_neutral/
    ├── emotion_sad/
    ├── emotion_surprised/
    └── data_summary.txt
```

### Audio Processing Requirements

1. **Sample Rate**: 24kHz (resampled from original)
2. **Channels**: Mono (converted from stereo if needed)
3. **Normalization**: Peak normalized to 0.95 to prevent clipping
4. **Format**: 16-bit PCM WAV

---

## Datasets Used

### 1. RAVDESS (Ryerson Audio-Visual Database)

**Source**: https://zenodo.org/record/1188976

**Statistics**:
- 1,440 audio files
- 24 actors (12 male, 12 female)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgusted, surprised
- 2 statements per emotion
- 2 intensity levels (normal, strong)

**Filename Format**: `MM-VV-EM-EI-ST-RE-AC.wav`
```
MM: Modality (01=full-AV, 02=video-only, 03=audio-only)
VV: Vocal channel (01=speech, 02=song)
EM: Emotion (01-08)
EI: Intensity (01=normal, 02=strong)
ST: Statement (01="Kids are talking", 02="Dogs are sitting")
RE: Repetition (01=1st, 02=2nd)
AC: Actor (01-24)
```

**Emotion Codes**:
| Code | Emotion |
|------|---------|
| 01 | neutral |
| 02 | calm |
| 03 | happy |
| 04 | sad |
| 05 | angry |
| 06 | fearful |
| 07 | disgusted |
| 08 | surprised |

### 2. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

**Source**: https://github.com/CheyneyComputerScience/CREMA-D

**Statistics**:
- 7,442 audio files
- 91 actors (48 male, 43 female)
- 6 emotions: neutral, happy, sad, angry, fearful, disgusted
- 12 sentences per actor
- 4 intensity levels (XX, LO, MD, HI)

**Filename Format**: `SSSS_UUU_EEE_II.wav`
```
SSSS: Speaker ID (1001-1091)
UUU:  Utterance code (DFA, IEO, IOM, etc.)
EEE:  Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
II:   Intensity (XX=unspecified, LO, MD, HI)
```

**Emotion Codes**:
| Code | Emotion |
|------|---------|
| ANG | angry |
| DIS | disgusted |
| FEA | fearful |
| HAP | happy |
| NEU | neutral |
| SAD | sad |

### 3. IESC (Indian Emotional Speech Corpora)

**Source**: Custom Hindi emotion speech corpus

**Statistics**:
- ~600 audio files
- Multiple speakers
- 5 emotions: neutral, happy, sad, angry, surprised (fearful in some versions)

**Original Structure**:
```
Indian Emotional Speech Corpora (IESC)/
├── Speaker-1/
│   ├── Anger/
│   │   └── *.wav
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Fear/
├── Speaker-2/
│   └── ...
```

**Emotion Mapping**:
| IESC Name | Our Name |
|-----------|----------|
| Anger | angry |
| Happy | happy |
| Neutral | neutral |
| Sad | sad |
| Fear | fearful |

---

## Data Preparation Scripts

### 1. prepare_emotion_data.py

**Purpose**: Processes RAVDESS and CREMA-D datasets into unified format.

**Location**: `prepare_emotion_data.py`

**Usage**:
```bash
# Process RAVDESS only
python prepare_emotion_data.py --ravdess /path/to/ravdess --output data/ravdess_emotions

# Process CREMA-D only
python prepare_emotion_data.py --cremad /path/to/cremad --output data/cremad_emotions

# Process both
python prepare_emotion_data.py \
    --ravdess /path/to/ravdess \
    --cremad /path/to/cremad \
    --output data/combined_emotions
```

**Code Walkthrough**:

```python
# Emotion mappings for each dataset
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}

CREMAD_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgusted",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
```

**Key Functions**:

#### `parse_ravdess_filename(filepath)`
Extracts emotion and metadata from RAVDESS filename.

```python
def parse_ravdess_filename(filepath: Path) -> Tuple[str, Dict]:
    """
    Parse RAVDESS filename: MM-VV-EM-EI-ST-RE-AC.wav

    Returns:
        emotion: Emotion name (e.g., "happy")
        metadata: Dict with actor, intensity, etc.
    """
    parts = filepath.stem.split("-")
    if len(parts) != 7:
        return None, {}

    emotion_code = parts[2]  # Position 3 (0-indexed: 2)
    emotion = RAVDESS_EMOTIONS.get(emotion_code)

    metadata = {
        "source": "ravdess",
        "modality": parts[0],
        "vocal_channel": "speech" if parts[1] == "01" else "song",
        "emotion_code": emotion_code,
        "intensity": "normal" if parts[3] == "01" else "strong",
        "actor": parts[6],
        "gender": "male" if int(parts[6]) % 2 == 1 else "female",
    }

    return emotion, metadata
```

#### `parse_cremad_filename(filepath)`
Extracts emotion and metadata from CREMA-D filename.

```python
def parse_cremad_filename(filepath: Path) -> Tuple[str, Dict]:
    """
    Parse CREMA-D filename: SSSS_UUU_EEE_II.wav

    Returns:
        emotion: Emotion name (e.g., "angry")
        metadata: Dict with speaker_id, utterance, etc.
    """
    parts = filepath.stem.split("_")
    if len(parts) != 4:
        return None, {}

    emotion_code = parts[2]  # Position 3 (0-indexed: 2)
    emotion = CREMAD_EMOTIONS.get(emotion_code)

    metadata = {
        "source": "cremad",
        "speaker_id": parts[0],
        "utterance": parts[1],
        "emotion_code": emotion_code,
        "intensity": parts[3],
    }

    return emotion, metadata
```

#### `process_audio_file(input_path, output_path, target_sr)`
Loads, resamples, normalizes, and saves audio.

```python
def process_audio_file(input_path, output_path, target_sr=24000):
    """
    Process audio file:
    1. Load at original sample rate
    2. Resample to target (24kHz)
    3. Normalize to prevent clipping
    4. Save as WAV
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Normalize to 0.95 peak
    max_val = abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    # Save
    sf.write(output_path, audio, target_sr)
    return True
```

#### `prepare_dataset(ravdess_dir, cremad_dir, output_dir)`
Main function that orchestrates the data preparation.

```python
def prepare_dataset(ravdess_dir, cremad_dir, output_dir):
    """
    Main preparation function:
    1. Create output directory structure
    2. Process RAVDESS files (if provided)
    3. Process CREMA-D files (if provided)
    4. Save metadata.json with statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: {"count": 0, "files": []})
    all_metadata = []

    # Process RAVDESS
    if ravdess_dir:
        ravdess_files = find_ravdess_files(ravdess_dir)
        for filepath in tqdm(ravdess_files, desc="Processing RAVDESS"):
            emotion, metadata = parse_ravdess_filename(filepath)
            if emotion is None:
                continue

            # Normalize emotion name
            emotion = EMOTION_NORMALIZATION.get(emotion, emotion)

            # Create emotion directory and process
            emotion_dir = output_dir / f"emotion_{emotion}"
            emotion_dir.mkdir(exist_ok=True)

            output_path = emotion_dir / f"ravdess_{filepath.stem}.wav"
            if process_audio_file(filepath, output_path):
                stats[emotion]["count"] += 1
                all_metadata.append(metadata)

    # Similar processing for CREMA-D...

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "total_files": sum(s["count"] for s in stats.values()),
            "emotions": {k: v["count"] for k, v in stats.items()},
            "target_sample_rate": 24000,
            "files": all_metadata,
        }, f, indent=2)
```

**Output**:
```
============================================================
Dataset Preparation Complete!
============================================================

Output directory: data/ravdess_emotions
Target sample rate: 24000 Hz

Emotion distribution:
  angry       :   192 files
  calm        :   192 files
  disgusted   :   192 files
  fearful     :   192 files
  happy       :   192 files
  neutral     :    96 files
  sad         :   192 files
  surprised   :   192 files
  TOTAL       :  1440 files

Metadata saved to: data/ravdess_emotions/metadata.json
```

---

### 2. reorganize_iesc_data.py

**Purpose**: Reorganizes IESC Hindi emotion data from speaker-based to emotion-based structure.

**Location**: `reorganize_iesc_data.py`

**Usage**:
```bash
python reorganize_iesc_data.py \
    --source "/path/to/Indian Emotional Speech Corpora (IESC)" \
    --target data/hindi_emotions
```

**Code Walkthrough**:

```python
# Emotion mapping from IESC folder names to our emotion names
emotion_mapping = {
    "Anger": "angry",
    "Happy": "happy",
    "Sad": "sad",
    "Neutral": "neutral",
    "Fear": "fearful",
}
```

**Key Function**:

#### `reorganize_iesc_data(source_dir, target_dir, copy_files)`
Reorganizes speaker-based structure to emotion-based.

```python
def reorganize_iesc_data(source_dir, target_dir, copy_files=True):
    """
    Reorganizes IESC data:

    FROM:
        Speaker-1/
            Anger/
                audio1.wav
            Happy/
                audio1.wav
        Speaker-2/
            ...

    TO:
        emotion_angry/
            Speaker-1_audio1.wav
            Speaker-2_audio1.wav
        emotion_happy/
            Speaker-1_audio1.wav
            ...
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directories
    for emotion in emotion_mapping.values():
        (target_path / f"emotion_{emotion}").mkdir(parents=True, exist_ok=True)

    # Find all speaker directories
    speaker_dirs = sorted([
        d for d in source_path.iterdir()
        if d.is_dir() and d.name.startswith("Speaker-")
    ])

    # Process each speaker
    for speaker_dir in speaker_dirs:
        speaker_name = speaker_dir.name

        # Process each emotion folder
        for emotion_folder in speaker_dir.iterdir():
            if not emotion_folder.is_dir():
                continue

            emotion_name = emotion_folder.name
            if emotion_name not in emotion_mapping:
                continue

            target_emotion = emotion_mapping[emotion_name]
            target_emotion_dir = target_path / f"emotion_{target_emotion}"

            # Process all audio files
            audio_files = list(emotion_folder.glob("*.wav"))

            for audio_file in audio_files:
                # Create unique filename: speaker_originalname
                new_filename = f"{speaker_name}_{audio_file.name}"
                target_file = target_emotion_dir / new_filename

                if copy_files:
                    shutil.copy2(audio_file, target_file)
                else:
                    target_file.symlink_to(audio_file.absolute())

    # Print statistics and save summary
    ...
```

**Output**:
```
Scanning source directory: /path/to/IESC
Target directory: data/hindi_emotions
============================================================
Found 10 speaker directories

Processing Speaker-1...
  Processed Speaker-1
Processing Speaker-2...
  ...

============================================================
Reorganization complete!
Total files processed: 600

Files per emotion:
  emotion_angry: 120 files
  emotion_fearful: 120 files
  emotion_happy: 120 files
  emotion_neutral: 120 files
  emotion_sad: 120 files

Summary saved to: data/hindi_emotions/data_summary.txt
```

---

### 3. download_hindi_emotion_data.py

**Purpose**: Sets up directory structure and provides guidance for Hindi emotion datasets.

**Location**: `download_hindi_emotion_data.py`

**Usage**:
```bash
# Create custom dataset structure
python download_hindi_emotion_data.py --output_dir data/hindi_emotions --dataset custom

# Setup for IndicTTS
python download_hindi_emotion_data.py --dataset indic_tts

# Setup for AI4Bharat
python download_hindi_emotion_data.py --dataset ai4bharat
```

**Code Walkthrough**:

#### `setup_indic_tts_structure(output_dir)`
Creates the standard emotion folder structure.

```python
def setup_indic_tts_structure(output_dir: Path):
    """
    Creates directory structure for emotion-labeled data:

    output_dir/
        emotion_happy/
        emotion_sad/
        emotion_angry/
        emotion_neutral/
        emotion_excited/
        emotion_calm/
        emotion_surprised/
        emotion_fearful/
    """
    emotions = ["happy", "sad", "angry", "neutral",
                "excited", "calm", "surprised", "fearful"]

    for emotion in emotions:
        (output_dir / f"emotion_{emotion}").mkdir(parents=True, exist_ok=True)
```

#### `create_sample_metadata(output_dir)`
Creates a metadata file with emotion descriptions in Hindi.

```python
def create_sample_metadata(output_dir: Path):
    """Creates metadata.json with emotion descriptions."""
    metadata = {
        "emotions": {
            "happy": {
                "description": "Joyful, upbeat speech",
                "examples": ["खुशी", "आनंद", "प्रसन्न"]  # Hindi examples
            },
            "sad": {
                "description": "Melancholic, subdued speech",
                "examples": ["दुख", "उदासी", "निराशा"]
            },
            # ... more emotions
        },
        "data_format": {
            "audio": "WAV format, 16kHz or 24kHz sample rate",
            "text": "Devanagari script (UTF-8)",
            "naming": "text_emotion.wav or use metadata.json"
        }
    }

    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
```

#### `create_labeling_script(output_dir)`
Creates a helper script for manual labeling.

```python
def create_labeling_script(output_dir: Path):
    """
    Creates label_audio.py helper script:

    Usage:
        python label_audio.py audio_file.wav happy

    This moves/copies the audio file to the appropriate emotion folder.
    """
    script_content = '''#!/usr/bin/env python3
import sys
from pathlib import Path
import shutil

audio_file = Path(sys.argv[1])
emotion = sys.argv[2].lower()

valid_emotions = ["happy", "sad", "angry", "neutral", ...]
if emotion not in valid_emotions:
    print(f"Invalid emotion. Must be one of: {valid_emotions}")
    sys.exit(1)

emotion_dir = Path("data/hindi_emotions") / f"emotion_{emotion}"
emotion_dir.mkdir(parents=True, exist_ok=True)

dest_file = emotion_dir / audio_file.name
shutil.copy2(audio_file, dest_file)
print(f"Labeled {audio_file.name} as {emotion}")
'''

    script_path = output_dir / "label_audio.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)  # Make executable
```

---

### 4. EmotionDataset Class (in train_emotion_lora.py)

**Purpose**: PyTorch Dataset class for loading prepared emotion data during training.

**Location**: `train_emotion_lora.py` (lines 47-177)

**Code Walkthrough**:

```python
class EmotionDataset(Dataset):
    """
    Dataset for emotion-labeled audio.

    Expected structure:
        data/
            emotion_happy/
                audio1.wav
            emotion_sad/
                audio1.wav
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: MTLTokenizer,
        emotion_mapping: Dict[str, str],
        language_id: str = "en",
        max_audio_length: float = 10.0,
        sample_rate: int = 24000,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.emotion_mapping = emotion_mapping
        self.language_id = language_id
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate

        # Load samples
        self.samples = self._load_samples()
```

#### `_load_samples()`
Scans emotion folders and builds sample list.

```python
def _load_samples(self) -> List[Dict]:
    """
    Scans emotion_* folders and collects all audio files.

    Returns list of dicts:
        [
            {"audio_path": "...", "emotion": "happy", "text": "..."},
            {"audio_path": "...", "emotion": "sad", "text": "..."},
            ...
        ]
    """
    samples = []

    for emotion_folder, emotion_type in self.emotion_mapping.items():
        emotion_dir = self.data_dir / emotion_folder
        if not emotion_dir.exists():
            print(f"Warning: {emotion_dir} does not exist")
            continue

        # Find audio files
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(emotion_dir.glob(f"*{ext}"))

        for audio_file in audio_files:
            text = self._extract_text_from_filename(audio_file)
            samples.append({
                "audio_path": str(audio_file),
                "emotion": emotion_type,
                "text": text,
            })

    return samples
```

#### `__getitem__(idx)`
Returns a single training sample with processed audio.

```python
def __getitem__(self, idx):
    """
    Returns:
        audio_16k: Tensor for S3 tokenizer (16kHz)
        audio_24k: Tensor for S3Gen (24kHz)
        text_tokens: Tokenized text
        emotion: Emotion label
    """
    sample = self.samples[idx]

    # Load audio at original rate
    audio, orig_sr = librosa.load(sample["audio_path"], sr=None)

    # Resample to 16kHz (for tokenization)
    audio_16k = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)

    # Resample to 24kHz (for S3Gen)
    audio_24k = librosa.resample(audio, orig_sr=orig_sr, target_sr=24000)

    # Truncate/pad to max length
    max_samples_16k = int(self.max_audio_length * 16000)
    max_samples_24k = int(self.max_audio_length * 24000)

    if len(audio_16k) > max_samples_16k:
        audio_16k = audio_16k[:max_samples_16k]
    else:
        audio_16k = np.pad(audio_16k, (0, max_samples_16k - len(audio_16k)))

    # Similar for audio_24k...

    # Tokenize text
    text_tokens = self.tokenizer.text_to_tokens(
        sample["text"],
        language_id=self.language_id
    )

    return {
        "audio_16k": torch.from_numpy(audio_16k).float(),
        "audio_24k": torch.from_numpy(audio_24k).float(),
        "text_tokens": text_tokens.squeeze(0),
        "emotion": sample["emotion"],
        "text": sample["text"],
    }
```

---

## Final Directory Structure

After running all preparation scripts, you should have:

```
chatterbox/
├── data/
│   ├── ravdess_emotions/              # 1,440 files
│   │   ├── emotion_angry/             # 192 files
│   │   ├── emotion_calm/              # 192 files
│   │   ├── emotion_disgusted/         # 192 files
│   │   ├── emotion_fearful/           # 192 files
│   │   ├── emotion_happy/             # 192 files
│   │   ├── emotion_neutral/           # 96 files
│   │   ├── emotion_sad/               # 192 files
│   │   ├── emotion_surprised/         # 192 files
│   │   └── metadata.json
│   │
│   ├── cremad_emotions/               # 7,442 files
│   │   ├── emotion_angry/             # 1,271 files
│   │   ├── emotion_disgusted/         # 1,271 files
│   │   ├── emotion_fearful/           # 1,271 files
│   │   ├── emotion_happy/             # 1,271 files
│   │   ├── emotion_neutral/           # 1,087 files
│   │   ├── emotion_sad/               # 1,271 files
│   │   └── metadata.json
│   │
│   └── hindi_emotions/                # ~600 files
│       ├── emotion_angry/             # ~120 files
│       ├── emotion_fearful/           # ~120 files
│       ├── emotion_happy/             # ~120 files
│       ├── emotion_neutral/           # ~120 files
│       ├── emotion_sad/               # ~120 files
│       └── data_summary.txt
│
├── prepare_emotion_data.py            # RAVDESS/CREMA-D processor
├── reorganize_iesc_data.py            # IESC reorganizer
├── download_hindi_emotion_data.py     # Hindi data setup
└── train_emotion_lora.py              # Training script
```

---

## Quick Start

### Step 1: Download Raw Datasets

```bash
# RAVDESS (download from Zenodo)
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
unzip Audio_Speech_Actors_01-24.zip -d /path/to/ravdess

# CREMA-D (clone from GitHub)
git clone https://github.com/CheyneyComputerScience/CREMA-D.git /path/to/cremad

# IESC (obtain from source)
# Place in /path/to/iesc
```

### Step 2: Prepare Each Dataset

```bash
# Prepare RAVDESS
python prepare_emotion_data.py \
    --ravdess /path/to/ravdess \
    --output data/ravdess_emotions

# Prepare CREMA-D
python prepare_emotion_data.py \
    --cremad /path/to/cremad \
    --output data/cremad_emotions

# Prepare IESC (Hindi)
python reorganize_iesc_data.py \
    --source "/path/to/iesc" \
    --target data/hindi_emotions
```

### Step 3: Verify Data

```bash
# Check file counts
find data/ravdess_emotions -name "*.wav" | wc -l   # Should be 1440
find data/cremad_emotions -name "*.wav" | wc -l    # Should be 7442
find data/hindi_emotions -name "*.wav" | wc -l     # Should be ~600
```

### Step 4: Train

```bash
# Train on RAVDESS
python train_emotion_lora.py \
    --data_dir data/ravdess_emotions \
    --output_dir checkpoints/emotion_lora_ravdess \
    --epochs 3 --batch_size 2 --lr 5e-5

# Train on CREMA-D
python train_emotion_lora.py \
    --data_dir data/cremad_emotions \
    --output_dir checkpoints/emotion_lora_cremad \
    --epochs 3 --batch_size 2 --lr 5e-5

# Train on IESC (Hindi)
python train_emotion_lora.py \
    --data_dir data/hindi_emotions \
    --output_dir checkpoints/emotion_lora_iesc \
    --language hi \
    --epochs 3 --batch_size 2 --lr 5e-5
```

### Step 5: Merge Checkpoints

```bash
python merge_emotion_checkpoints.py \
    --auto-weights \
    --output checkpoints/emotion_merged/checkpoint_merged.pt
```

---

## Troubleshooting

### Common Issues

**1. "No audio files found"**
- Check that audio files are in the correct format (.wav)
- Verify folder names start with `emotion_`
- Ensure files are not in nested subdirectories

**2. "Sample rate mismatch"**
- The scripts automatically resample to 24kHz
- Original files can be any sample rate

**3. "Out of memory during training"**
- Reduce `--batch_size` to 1
- Reduce `--max_audio_length` (default 10s)
- Use `--lora_rank 4` instead of 8

**4. "Emotion not recognized"**
- Check `emotion_mapping` in the script matches your folder names
- Ensure emotion names are normalized (lowercase, no spaces)

---

## References

- RAVDESS: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLOS ONE.
- CREMA-D: Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.
- IESC: Indian Emotional Speech Corpora for Hindi language emotion recognition.
