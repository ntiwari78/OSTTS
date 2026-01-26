# train_emotion_lora.py - Line-by-Line Explanation

This document explains `train_emotion_lora.py` in plain English, making each code section readable as English statements.

## File Header and Imports (Lines 1-88)

**Lines 1-18**: This is a Python script that fine-tunes the Chatterbox T3 model for emotion control. It uses LoRA (Low-Rank Adaptation) to efficiently train on emotion-labeled audio from RAVDESS, CREMA-D, and IESC datasets.

**Lines 20-28**: Import standard Python libraries for file paths, command-line arguments, JSON handling, data structures, and time tracking.

**Lines 31-34**: Add the `src` directory to Python's import path so we can import chatterbox modules.

**Lines 36-43**: Import PyTorch libraries for neural networks, data loading, and audio processing tools (librosa, soundfile).

**Lines 45-56**: Import Chatterbox-specific modules:
- The main TTS model (`ChatterboxMultilingualTTS`)
- LoRA adapter functions for efficient training
- Emotion loss functions
- Tokenizers and audio processing components

**Lines 58-67**: Try to import prosody enhancement modules (Phase 2 feature). If not available, set a flag to False.

**Lines 69-77**: Try to import contrastive learning modules (Phase 3 feature). If not available, set a flag to False.

**Lines 79-87**: Try to import SER data filtering modules (Phase 3 feature). If not available, set a flag to False.

## Dataset Configuration (Lines 90-172)

**Lines 94-103**: Define a data class `DatasetConfig` that stores:
- Dataset name
- Default file path
- Mapping from folder names to emotion types
- Expected number of samples
- Language code
- Description text
- List of unique emotions in this dataset

**Lines 107-157**: Create configuration objects for three datasets:
- **RAVDESS** (lines 108-125): 1,440 English samples with 8 emotions (neutral, calm, happy, sad, angry, fearful, disgusted, surprised)
- **CREMA-D** (lines 126-141): 7,442 English samples with 6 emotions (neutral, happy, sad, angry, fearful, disgusted)
- **IESC** (lines 142-156): 600 Hindi samples with 5 emotions (neutral, happy, sad, angry, fearful)

**Lines 160-172**: Create a combined emotion mapping that includes all emotions from all datasets, plus additional emotions like "excited", "whisper", and "shout".

## Training Log Structure (Lines 175-197)

**Lines 175-191**: Define a `TrainingLog` data class that tracks:
- Which dataset was used
- Start and end times
- How many samples were loaded vs expected
- Which files were missed
- How many epochs completed
- Final loss value
- Whether training stopped early
- Distribution of emotions in the dataset
- List of all batch losses and epoch losses
- Training configuration settings

**Lines 193-196**: Define a method to save the training log as a JSON file.

## Data Validation (Lines 199-245)

**Lines 199-237**: Function `validate_dataset_coverage` checks if all expected audio files were loaded:
- Count how many samples are actually in the dataset
- Scan all emotion folders to find any audio files that weren't loaded
- Check if the count matches expected (allowing 5% tolerance for missing files)
- Return whether validation passed, the actual count, and list of missed files

**Lines 240-245**: Function `get_emotion_distribution` counts how many samples exist for each emotion type in the dataset.

## Emotion Dataset Class (Lines 248-386)

**Lines 248-260**: Define `EmotionDataset` class that loads emotion-labeled audio files. Expected folder structure: `data/emotion_happy/audio1.wav`, `data/emotion_sad/audio2.wav`, etc.

**Lines 262-288**: Initialize the dataset:
- Store the data directory path
- Store the tokenizer for text processing
- Store the emotion mapping (folder name → emotion type)
- Store language ID (en/hi)
- Store maximum audio length (10 seconds default)
- Store target sample rate (24kHz)
- Store dataset source name
- Load all samples by calling `_load_samples()`

**Lines 290-293**: Print how many samples were loaded.

**Lines 295-327**: Method `_load_samples()`:
- For each emotion folder in the mapping:
  - Check if the folder exists
  - Find all audio files (.wav, .mp3, .flac, .m4a)
  - For each audio file:
    - Extract text from filename (or use placeholder)
    - Create a sample dictionary with audio path, emotion label, text, and metadata
- Return the list of all samples

**Lines 329-341**: Method `_extract_text_from_filename()`:
- For RAVDESS/CREMA-D, actual transcripts aren't in filenames
- Return a generic placeholder text "This is a test sentence."
- The emotion embedding should learn to work regardless of text content

**Lines 343-344**: Method `__len__()` returns the number of samples in the dataset.

**Lines 346-386**: Method `__getitem__()` loads one sample:
- Get the sample at the given index
- Load audio file at its original sample rate
- Resample audio to 16kHz (for tokenization) and 24kHz (for reference)
- Truncate or pad audio to maximum length
- Tokenize the text using the tokenizer
- Return a dictionary with:
  - Audio at 16kHz and 24kHz
  - Text tokens
  - Text length
  - Emotion label
  - Original text

## Balanced Sampling (Lines 389-511)

**Lines 393-451**: Class `BalancedEmotionSampler` ensures equal representation of each emotion:
- Group all samples by emotion type
- Find the emotion with the most samples
- For each emotion, sample enough to match the maximum count (with replacement if needed)
- Shuffle all sampled indices
- This prevents the model from overfitting to dominant emotions

**Lines 454-511**: Class `DatasetWeightedSampler` balances across multiple datasets:
- Group samples by source dataset (RAVDESS, CREMA-D, IESC)
- Sample equally from each dataset source
- This ensures each dataset contributes equally despite size differences

## Data Download Helper (Lines 514-544)

**Lines 514-544**: Function `download_hindi_emotion_data()`:
- Creates the directory structure for Hindi emotion data
- Prints instructions on where to get Hindi datasets
- Creates example emotion folders
- Returns the output path

## Model Setup with LoRA (Lines 547-619)

**Lines 547-619**: Function `setup_model_with_lora()`:
- If `freeze_base` is True, freeze all parameters in T3, S3Gen, voice encoder, and emotion embeddings
- If using adapters: add adapter layers to transformer
- If using LoRA: apply LoRA to attention and feed-forward layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Enable training only for LoRA parameters (count how many)
- Enable training for emotion cross-attention layers (these are new, not pre-trained)
- Enable training for emotion embeddings (these need to learn)
- Return the modified model

## Training Step Function (Lines 622-973)

**Lines 622-634**: Function `train_step()` performs one training step. It takes:
- The model
- A batch of data
- Device (CPU/GPU)
- Optimizer
- Batch index
- Optional loss functions (combined, prosody, contrastive)
- Optional weights for each loss type

**Lines 636-638**: Set model to training mode for T3, evaluation mode for S3Gen and voice encoder (they're frozen).

**Lines 640-646**: Clear gradients from previous step. Get audio, text tokens, emotions, and batch size from the batch.

**Lines 648-677**: Prepare text tokens:
- Get start and stop token IDs from model config
- For each sample in batch:
  - Remove padding to get actual text length
  - Add start token at beginning
  - Add stop token at end
- Pad all text sequences to the same length

**Lines 679-689**: Get emotion embeddings:
- For each emotion in the batch, get its 64-dimensional embedding
- Stack all embeddings into a batch tensor
- Add sequence dimension: (B, emotion_dim) → (B, 1, emotion_dim)

**Lines 691-706**: Extract speaker embeddings from audio:
- For each audio sample, use voice encoder to get speaker embedding
- Handle errors by using zero embedding if voice encoder fails
- Stack all speaker embeddings

**Lines 708-746**: Tokenize audio to speech tokens:
- Convert audio to list of numpy arrays
- For each audio sample:
  - Use S3Gen tokenizer to convert audio to discrete tokens
  - Handle errors by skipping the sample if tokenization fails
- Pad all speech token sequences to the same length

**Lines 748-754**: Create conditioning object:
- Combine speaker embeddings and emotion embeddings
- Move to the correct device

**Lines 756-793**: Calculate TTS loss:
- Set emotion embeddings to training mode
- Print debug info for first batch (shapes of all tensors)
- Check that token lengths match tensor sizes
- Call T3 forward pass to get text and speech logits

**Lines 805-823**: Reshape logits and targets for loss calculation:
- Flatten logits from (B, N, C) to (B*N, C)
- Create masks to ignore padding tokens
- Flatten targets from (B, N) to (B*N,)

**Lines 825-843**: Compute losses:
- Calculate cross-entropy loss for text prediction
- Calculate cross-entropy loss for speech prediction
- Print debug info for first few batches

**Lines 844-862**: Check for invalid losses:
- If loss is NaN, return 0.0 to skip this batch
- If loss is infinite, return 0.0 to skip this batch
- If both losses are exactly 0.0, return 0.0 (this shouldn't happen)

**Lines 850-851**: Combine text and speech losses with weights (speech loss is 2x more important).

**Lines 864-894**: Apply SER loss if enabled:
- Get audio for SER evaluation
- Remove sequence dimension from emotion embeddings
- Call combined loss function with TTS loss, emotion embeddings, audio, and target emotions
- Get total loss (TTS loss + SER loss)
- Log SER accuracy every 50 batches

**Lines 896-913**: Apply prosody loss if enabled:
- Get emotion embeddings for prosody prediction
- Calculate prosody loss
- Add to total loss with weight
- Log prosody loss every 100 batches

**Lines 914-938**: Apply contrastive loss if enabled:
- Convert emotion names to numeric labels
- Calculate contrastive loss on emotion embeddings
- Add to total loss with weight
- Log contrastive loss every 100 batches

**Lines 940-965**: Backward pass and optimization:
- Compute gradients with `loss.backward()`
- Check if gradients are flowing (calculate gradient norm)
- If gradient norm is zero, skip this batch
- Clip gradients to maximum norm of 1.0 for stability
- Update parameters with optimizer

**Lines 967-973**: Return the loss value, or 0.0 if there was an error.

## Main Function - Argument Parsing (Lines 976-1096)

**Lines 976-997**: Create argument parser with description and examples of how to use the script.

**Lines 999-1012**: Add arguments for dataset selection:
- `--dataset`: Choose which dataset to train on (ravdess, cremad, iesc, combined, custom)
- `--data_dir`: Custom data directory path
- `--ravdess_dir`, `--cremad_dir`, `--iesc_dir`: Per-dataset directories for combined training

**Lines 1014-1016**: Add argument for output directory (auto-generated if not specified).

**Lines 1018-1028**: Add arguments for device and model settings:
- `--device`: Which device to use (auto, cuda, mps, cpu)
- `--lora_rank`: LoRA rank (default 8)
- `--lora_alpha`: LoRA alpha (default 16.0)
- `--use_adapter`: Use adapters instead of LoRA
- `--adapter_size`: Adapter bottleneck size (default 64)

**Lines 1030-1036**: Add arguments for training hyperparameters:
- `--batch_size`: Batch size (default 4)
- `--epochs`: Number of epochs (default 10)
- `--lr`: Learning rate (default 1e-4)

**Lines 1038-1046**: Add arguments for sampling strategy:
- `--balanced_sampling`: Use balanced emotion sampling
- `--balanced_datasets`: Balance across datasets in combined mode
- `--language`: Language code (auto-detected if not specified)

**Lines 1048-1058**: Add arguments for early stopping:
- `--early_stop_loss`: Stop when loss falls below this threshold (default 0.4)
- `--early_stop_patience`: Number of consecutive batches below threshold (default 50)
- `--validate_data`: Validate dataset coverage (default True)
- `--skip_validation`: Skip data validation

**Lines 1060-1066**: Add arguments for SER Integration Loss:
- `--use_ser_loss`: Enable SER loss during training
- `--ser_weight`: Weight for SER loss (default 0.3)
- `--consistency_weight`: Weight for emotion consistency loss (default 0.5)

**Lines 1068-1074**: Add arguments for Prosody Enhancement (Phase 2):
- `--use_prosody_loss`: Enable prosody prediction loss
- `--prosody_weight`: Weight for prosody loss (default 0.2)
- `--expressiveness_scale`: Expressiveness scale for emotions (default 1.0)

**Lines 1076-1082**: Add arguments for Contrastive Learning (Phase 3):
- `--use_contrastive_loss`: Enable contrastive loss
- `--contrastive_weight`: Weight for contrastive loss (default 0.1)
- `--contrastive_temperature`: Temperature for contrastive loss (default 0.07)

**Lines 1084-1090**: Add arguments for SER-guided Data Filtering (Phase 3):
- `--use_ser_filtering`: Enable SER data filtering
- `--min_agreement`: Minimum SER model agreement (default 0.5)
- `--filter_cache_path`: Path to cache filtered dataset

**Lines 1092-1094**: Add argument for data download:
- `--download_data`: Download/setup data directory

**Line 1096**: Parse all arguments.

## Main Function - Setup (Lines 1098-1176)

**Lines 1098-1101**: Auto-generate output directory if not specified:
- Format: `checkpoints/emotion_lora_{dataset_name}`

**Lines 1103-1109**: Auto-detect language:
- If dataset is IESC, set language to Hindi ("hi")
- Otherwise, set language to English ("en")

**Lines 1111-1122**: Setup device:
- If "auto", check for CUDA first, then MPS (Apple Silicon), then fall back to CPU
- Otherwise use the specified device

**Lines 1124-1129**: If `--download_data` flag is set:
- Call download function
- Print instructions
- Exit

**Lines 1131-1158**: Initialize training log:
- Store dataset name
- Store start time
- Store all configuration settings (LoRA params, batch size, epochs, learning rate, etc.)

**Lines 1160-1176**: Load and setup model:
- Load pre-trained ChatterboxMultilingualTTS model
- Apply LoRA/Adapter modifications
- Move all model components to the specified device

## Main Function - Dataset Setup (Lines 1178-1274)

**Lines 1181-1204**: If dataset is one of the predefined configs (ravdess, cremad, iesc):
- Get the dataset configuration
- Use custom data directory if provided, otherwise use default
- Get emotion mapping and expected sample count
- Create EmotionDataset with the configuration
- Print dataset information

**Lines 1206-1252**: If dataset is "combined":
- Load samples from all three datasets (RAVDESS, CREMA-D, IESC)
- For each dataset:
  - Check if directory exists
  - Load samples from that dataset
  - Add to combined list
- Create combined dataset with all samples
- Calculate total expected samples

**Lines 1254-1269**: If dataset is "custom":
- Require `--data_dir` to be specified
- Use combined emotion mapping
- Create dataset from custom directory

**Lines 1271-1274**: Check if any data was loaded. If not, print error and exit.

**Lines 1276-1277**: Update training log with number of samples loaded.

## Main Function - Data Validation (Lines 1279-1306)

**Lines 1282-1299**: If validation is enabled and config exists:
- Call `validate_dataset_coverage()` to check all files were loaded
- Store missed files in training log
- Print warning if files were missed
- Print validation result (PASSED or FAILED)

**Lines 1301-1306**: Get emotion distribution:
- Count samples per emotion
- Store in training log
- Print distribution

## Main Function - SER Data Filtering (Lines 1308-1363)

**Lines 1311-1363**: If SER filtering is enabled:
- Check if SER filter module is available
- Create SERDataFilter with minimum agreement threshold
- Get all audio paths and expected emotions from dataset
- Filter dataset to keep only samples where SER models agree
- Create filtered subset of dataset
- Print filtering statistics
- Update training log with filtered sample count

## Main Function - DataLoader Setup (Lines 1365-1392)

**Lines 1368-1376**: If balanced emotion sampling is enabled:
- Create BalancedEmotionSampler
- Create DataLoader with the sampler

**Lines 1377-1385**: If balanced datasets is enabled (for combined dataset):
- Create DatasetWeightedSampler
- Create DataLoader with the sampler

**Lines 1386-1392**: Otherwise:
- Create DataLoader with random shuffling

## Main Function - Optimizer Setup (Lines 1394-1417)

**Lines 1394-1410**: Collect trainable parameters:
- Loop through T3 model and emotion embeddings
- Find all parameters that require gradients
- Print first 20 parameter names
- Print total count if more than 20

**Lines 1412-1414**: Check if any trainable parameters exist. If not, print warning and exit.

**Line 1416**: Print total number of trainable parameters.

**Line 1417**: Create AdamW optimizer with learning rate and weight decay.

## Main Function - Loss Function Setup (Lines 1419-1481)

**Lines 1419-1437**: If SER loss is enabled:
- Create CombinedEmotionLoss with SER integration
- Move to device
- Print setup confirmation

**Lines 1439-1462**: If prosody loss is enabled:
- Check if prosody module is available
- Create prosody predictor and loss function
- Move to device
- Add prosody predictor parameters to optimizer
- Print setup confirmation

**Lines 1464-1481**: If contrastive loss is enabled:
- Check if contrastive module is available
- Create SupervisedContrastiveLoss
- Move to device
- Print setup confirmation

## Main Function - Training Loop (Lines 1483-1577)

**Lines 1484-1491**: Print training start information:
- Number of epochs
- Dataset name and sample count
- Early stopping criteria

**Lines 1493-1495**: Create output directory and initialize early stopping counter.

**Lines 1497-1577**: For each epoch:
- If early stopped, break out of loop
- Initialize loss tracking variables
- Create progress bar
- For each batch:
  - Call `train_step()` to process the batch
  - If loss is valid (> 0.0):
    - Add to total loss
    - Increment valid batch counter
    - Store loss in training log
    - Check early stopping condition:
      - If loss < threshold, increment counter
      - If counter >= patience, stop training
      - Otherwise, reset counter
  - Update progress bar with current loss
  - Handle errors by printing and continuing
- Calculate average loss for the epoch
- Store epoch loss in training log
- Save checkpoint with:
  - Epoch number
  - Dataset name
  - T3 model state
  - Emotion embeddings state
  - Optimizer state
  - Loss value
  - Configuration
- If early stopped, save special early stop checkpoint

## Main Function - Finalization (Lines 1579-1599)

**Lines 1579-1583**: Save training log:
- Set end time
- Save to JSON file
- Print confirmation

**Lines 1585-1599**: Print training summary:
- Dataset name
- Samples loaded vs expected
- Coverage percentage
- Epochs completed
- Final loss
- Whether early stopped
- Checkpoint location

## Entry Point (Lines 1602-1603)

**Lines 1602-1603**: If script is run directly (not imported), call the `main()` function.

---

## Key Concepts Explained

### LoRA (Low-Rank Adaptation)
Instead of training all model parameters, LoRA adds small trainable matrices to existing layers. This is much more efficient and requires less memory.

### Emotion Embeddings
64-dimensional vectors that represent different emotions. The model learns to associate these embeddings with the corresponding emotional characteristics in speech.

### SER Loss
Speech Emotion Recognition loss uses an external model to verify that generated audio matches the intended emotion. This provides additional training signal.

### Balanced Sampling
Ensures each emotion is represented equally during training, preventing the model from overfitting to dominant emotions.

### Early Stopping
Stops training when loss falls below a threshold for a certain number of consecutive batches, preventing overfitting.

### Prosody Loss (Phase 2)
Predicts prosodic features (pitch, energy, tempo) from emotion embeddings and compares to actual audio prosody.

### Contrastive Loss (Phase 3)
Pushes embeddings of different emotions apart and pulls embeddings of the same emotion together, improving emotion separation.

### SER Data Filtering (Phase 3)
Uses SER models to filter out training samples where the audio doesn't clearly match the emotion label, improving data quality.
