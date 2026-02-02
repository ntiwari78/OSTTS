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

# ====================================================================
# V06 IMPROVEMENT EXPERIMENTS
# ====================================================================
# Based on V03-SER (69%) vs V05 (51.7%) root cause analysis.
# V03-SER used: ser=0.3, prosody=0.2, contrastive=0.1, SER filtering,
#   no scheduler, weight_decay=1e-5 (old default), stopped at epoch 2.
# V05 used: ser=0.5, NO prosody/contrastive/filtering,
#   cosine scheduler, weight_decay=0.01, trained all 15 epochs.

# Experiment A: Replicate V03-SER recipe exactly
python train_emotion_lora.py \
  --dataset ravdess \
  --output_dir checkpoints/exp_a_v03_replica \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment B: V03 recipe + gentle cosine scheduler
python train_emotion_lora.py \
  --dataset ravdess \
  --output_dir checkpoints/exp_b_cosine \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler cosine --warmup_steps 100 --min_lr 5e-5 \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment C1: SER weight = 0.2 (lower)
python train_emotion_lora.py \
  --dataset ravdess \
  --output_dir checkpoints/exp_c1_ser02 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.2 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment C3: SER weight = 0.5 (V05's setting, but with all losses)
python train_emotion_lora.py \
  --dataset ravdess \
  --output_dir checkpoints/exp_c3_ser05 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.5 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment D2: weight_decay = 5e-3
python train_emotion_lora.py \
  --dataset ravdess \
  --output_dir checkpoints/exp_d2_wd5e3 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 5e-3 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment D3: weight_decay = 0.01 (V05's setting)
python train_emotion_lora.py \
  --dataset ravdess \
  --output_dir checkpoints/exp_d3_wd01 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 0.01 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# ====================================================================
# CREMA-D EXPERIMENTS (V03-SER recipe variants)
# ====================================================================
# Baseline: cremad_v05 = V03-SER recipe, 15 epochs → 55.6% emotion2vec
# CREMA-D has 7442 samples (5x RAVDESS), 6 emotions.
# Goal: find best recipe/epoch for CREMA-D.

# Experiment CA: V03-SER recipe, 5 epochs (find sweet spot epoch)
python train_emotion_lora.py \
  --dataset cremad \
  --output_dir checkpoints/exp_ca_v03_replica \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment CB: V03-SER recipe + gentle cosine scheduler
python train_emotion_lora.py \
  --dataset cremad \
  --output_dir checkpoints/exp_cb_cosine \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler cosine --warmup_steps 100 --min_lr 5e-5 \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment CC1: SER weight = 0.2 (lower)
python train_emotion_lora.py \
  --dataset cremad \
  --output_dir checkpoints/exp_cc1_ser02 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.2 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment CC3: SER weight = 0.5 (V05's setting, but with all losses)
python train_emotion_lora.py \
  --dataset cremad \
  --output_dir checkpoints/exp_cc3_ser05 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.5 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment CD2: weight_decay = 5e-3
python train_emotion_lora.py \
  --dataset cremad \
  --output_dir checkpoints/exp_cd2_wd5e3 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 5e-3 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment CD3: weight_decay = 0.01 (V05's setting)
python train_emotion_lora.py \
  --dataset cremad \
  --output_dir checkpoints/exp_cd3_wd01 \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 0.01 \
  --scheduler none \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# Experiment CE: V03-SER recipe + balanced sampling
python train_emotion_lora.py \
  --dataset cremad \
  --output_dir checkpoints/exp_ce_balanced \
  --epochs 5 --batch_size 4 --lr 1e-4 --weight_decay 1e-5 \
  --scheduler none --balanced_sampling \
  --use_ser_loss --ser_weight 0.3 --consistency_weight 0.5 \
  --use_prosody_loss --prosody_weight 0.2 --expressiveness_scale 1.3 \
  --use_contrastive_loss --contrastive_weight 0.1 --contrastive_temperature 0.07 \
  --use_ser_filtering --min_agreement 0.5

# ====================================================================
# BENCHMARKING — RAVDESS experiments (run for each experiment's best epoch)
# ====================================================================
# python benchmark_v03.py --checkpoint exp_a_epoch2 --output benchmark_output/exp_a_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_a_epoch2

# ====================================================================
# BENCHMARKING — CREMA-D experiments
# ====================================================================
# For each CREMA-D experiment, benchmark epochs 2 and 3 (sweet spot range):
#
# python benchmark_v03.py --checkpoint exp_ca_epoch2 --output benchmark_output/exp_ca_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_ca_epoch2
#
# python benchmark_v03.py --checkpoint exp_ca_epoch3 --output benchmark_output/exp_ca_epoch3
# python benchmark_llm_emotions.py --checkpoint exp_ca_epoch3
#
# python benchmark_v03.py --checkpoint exp_cb_epoch2 --output benchmark_output/exp_cb_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_cb_epoch2
#
# python benchmark_v03.py --checkpoint exp_cc1_epoch2 --output benchmark_output/exp_cc1_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_cc1_epoch2
#
# python benchmark_v03.py --checkpoint exp_cc3_epoch2 --output benchmark_output/exp_cc3_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_cc3_epoch2
#
# python benchmark_v03.py --checkpoint exp_cd2_epoch2 --output benchmark_output/exp_cd2_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_cd2_epoch2
#
# python benchmark_v03.py --checkpoint exp_cd3_epoch2 --output benchmark_output/exp_cd3_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_cd3_epoch2
#
# python benchmark_v03.py --checkpoint exp_ce_epoch2 --output benchmark_output/exp_ce_epoch2
# python benchmark_llm_emotions.py --checkpoint exp_ce_epoch2

# Compare LLM results
python - <<'PY'
import json
from collections import defaultdict

v05_path = "/Users/ntiwari/IITP/attempt4/chatterbox/benchmark_output/comparison/BENCHMARK_LLM_RESULTS_V05.json"
v03_path = "/Users/ntiwari/IITP/attempt4/chatterbox/benchmark_output/comparison/BENCHMARK_LLM_RESULTS_V03.json"

with open(v05_path) as f:
    v05 = json.load(f)
with open(v03_path) as f:
    v03 = json.load(f)

# get model results
v05_models = next(iter(v05.values()))["model_results"]
v03_models = next(iter(v03.values()))["model_results"]

# compare file sets per model
for model in ["emotion2vec_base", "wav2vec2_ehcalabres", "dpngtm"]:
    v05_files = {p["file"] for p in v05_models[model]["predictions"]}
    v03_files = {p["file"] for p in v03_models[model]["predictions"]}
    only_v05 = sorted(v05_files - v03_files)
    only_v03 = sorted(v03_files - v05_files)
    print(model)
    print("  v05 only:", only_v05)
    print("  v03 only:", only_v03)

# compare correctness changes for emotion2vec (since main drop)
model = "emotion2vec_base"
v05_pred = {p["file"]: p for p in v05_models[model]["predictions"]}
v03_pred = {p["file"]: p for p in v03_models[model]["predictions"]}
common = sorted(set(v05_pred) & set(v03_pred))
flips = []
for f in common:
    c03 = v03_pred[f]["correct"]
    c05 = v05_pred[f]["correct"]
    if c03 != c05:
        flips.append((f, c03, c05, v03_pred[f]["predicted"], v05_pred[f]["predicted"], v03_pred[f]["expected"]))

print("\nEmotion2vec correctness flips:")
for row in flips:
    print(" ", row)

# per-model correctness flip counts
for model in ["emotion2vec_base", "wav2vec2_ehcalabres", "dpngtm"]:
    v05_pred = {p["file"]: p for p in v05_models[model]["predictions"]}
    v03_pred = {p["file"]: p for p in v03_models[model]["predictions"]}
    common = set(v05_pred) & set(v03_pred)
    up = down = 0
    for f in common:
        c03 = v03_pred[f]["correct"]
        c05 = v05_pred[f]["correct"]
        if c03 and not c05:
            down += 1
        elif not c03 and c05:
            up += 1
    print(f"{model}: +{up} / -{down} flips")
PY