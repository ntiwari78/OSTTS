# Train on each dataset separately (creates separate checkpoints)
python train_emotion_lora.py --dataset ravdess --output_dir checkpoints/emotion_lora_ravdess
python train_emotion_lora.py --dataset cremad --output_dir checkpoints/emotion_lora_cremad
python train_emotion_lora.py --dataset iesc --output_dir checkpoints/emotion_lora_iesc

# With balanced sampling
python train_emotion_lora.py --dataset ravdess --balanced_sampling --output_dir checkpoints/emotion_lora_ravdess

# Merge checkpoints
python merge_emotion_checkpoints.py --method dataset_adaptive --output checkpoints/emotion_merged/checkpoint_merged.pt

# Run tests
python test_emotion_system.py
