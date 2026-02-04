# Emotion TTS Benchmark Results v0.3

**Generated**: 2026-01-31 20:08:24

## Summary

| Checkpoint | Dataset | Tests | Passed | Failed | Success Rate |
|------------|---------|-------|--------|--------|--------------|
| RAVDESS | RAVDESS | 29 | 29 | 0 | 100.0% |
| CREMAD | CREMA-D | 27 | 27 | 0 | 100.0% |
| IESC | IESC | 9 | 9 | 0 | 100.0% |
| IESC_SER | IESC-SER | 9 | 9 | 0 | 100.0% |
| RAVDESS_SER | RAVDESS-SER | 29 | 29 | 0 | 100.0% |
| V04_FULL | Combined-V04 | 29 | 29 | 0 | 100.0% |
| V05_RAVDESS | RAVDESS-V05 | 29 | 29 | 0 | 100.0% |

## Prosody Analysis by Emotion

### Pitch (Mean Hz)

| Emotion | RAVDESS | CREMAD | IESC | RAVDESS-SER | CREMAD-SER | IESC-SER |
|---------|---------|---------|---------|---------|---------|---------|
| affectionate | 142.8 | 141.6 | - | 157.7 | - | - |
| angry | 228.7 | 215.9 | 171.3 | 132.5 | - | 230.7 |
| awed | 172.4 | 141.8 | - | 175.4 | - | - |
| bored | 128.9 | 145.1 | - | 178.4 | - | - |
| calm | 129.2 | - | - | 149.1 | - | - |
| contemptuous | 155.0 | 139.0 | - | 143.9 | - | - |
| disgusted | 183.9 | 126.8 | - | 197.6 | - | - |
| excited | 145.5 | 134.4 | - | 152.9 | - | - |
| fearful | 125.4 | 137.6 | - | 235.0 | - | - |
| happy | 210.8 | 110.9 | 198.4 | 178.0 | - | 144.9 |
| neutral | 154.5 | 139.6 | 153.2 | 194.1 | - | 176.8 |
| sad | 140.8 | 179.5 | 236.5 | 171.5 | - | 137.2 |
| sarcastic | 174.0 | 146.2 | - | 162.3 | - | - |
| surprised | 181.6 | - | 157.1 | 144.4 | - | 170.4 |

### Energy (Mean RMS)

| Emotion | RAVDESS | CREMAD | IESC | RAVDESS-SER | CREMAD-SER | IESC-SER |
|---------|---------|---------|---------|---------|---------|---------|
| affectionate | 0.1110 | 0.1322 | - | 0.1083 | - | - |
| angry | 0.0721 | 0.0550 | 0.0930 | 0.0737 | - | 0.0679 |
| awed | 0.1106 | 0.1033 | - | 0.0951 | - | - |
| bored | 0.0751 | 0.0816 | - | 0.0622 | - | - |
| calm | 0.0810 | - | - | 0.0951 | - | - |
| contemptuous | 0.1087 | 0.1141 | - | 0.0796 | - | - |
| disgusted | 0.1110 | 0.0730 | - | 0.0797 | - | - |
| excited | 0.0948 | 0.0801 | - | 0.1043 | - | - |
| fearful | 0.1135 | 0.1022 | - | 0.0693 | - | - |
| happy | 0.0530 | 0.0351 | 0.0913 | 0.1136 | - | 0.0692 |
| neutral | 0.0921 | 0.0740 | 0.1064 | 0.0802 | - | 0.0952 |
| sad | 0.0790 | 0.0795 | 0.0860 | 0.0901 | - | 0.0518 |
| sarcastic | 0.1064 | 0.0971 | - | 0.1241 | - | - |
| surprised | 0.1149 | - | 0.0982 | 0.0851 | - | 0.0914 |

## Test Details

### Basic Emotion Tests

#### RAVDESS

| Emotion | Status | Duration | Pitch (Hz) |
|---------|--------|----------|------------|
| neutral | ✓ | 2.76s | 154.5 |
| happy | ✓ | 4.48s | 210.8 |
| sad | ✓ | 2.72s | 140.8 |
| angry | ✓ | 1.76s | 228.7 |
| fearful | ✓ | 2.72s | 125.4 |
| surprised | ✓ | 2.16s | 181.6 |
| calm | ✓ | 2.04s | 129.2 |
| disgusted | ✓ | 2.48s | 183.9 |
| excited | ✓ | 2.96s | 145.5 |

#### CREMAD

| Emotion | Status | Duration | Pitch (Hz) |
|---------|--------|----------|------------|
| neutral | ✓ | 4.08s | 139.6 |
| happy | ✓ | 4.72s | 110.9 |
| sad | ✓ | 6.48s | 179.5 |
| angry | ✓ | 4.16s | 215.9 |
| fearful | ✓ | 2.68s | 137.6 |
| disgusted | ✓ | 2.60s | 126.8 |
| excited | ✓ | 3.52s | 134.4 |

#### IESC

| Emotion | Status | Duration | Pitch (Hz) |
|---------|--------|----------|------------|
| neutral | ✓ | 2.44s | 153.2 |
| happy | ✓ | 2.44s | 198.4 |
| sad | ✓ | 1.80s | 236.5 |
| angry | ✓ | 1.80s | 171.3 |
| surprised | ✓ | 4.92s | 157.1 |

#### IESC_SER

| Emotion | Status | Duration | Pitch (Hz) |
|---------|--------|----------|------------|
| neutral | ✓ | 2.32s | 176.8 |
| happy | ✓ | 3.00s | 144.9 |
| sad | ✓ | 3.24s | 137.2 |
| angry | ✓ | 4.08s | 230.7 |
| surprised | ✓ | 3.12s | 170.4 |

#### RAVDESS_SER

| Emotion | Status | Duration | Pitch (Hz) |
|---------|--------|----------|------------|
| neutral | ✓ | 3.08s | 194.1 |
| happy | ✓ | 3.40s | 178.0 |
| sad | ✓ | 2.92s | 171.5 |
| angry | ✓ | 2.72s | 132.5 |
| fearful | ✓ | 2.84s | 235.0 |
| surprised | ✓ | 3.24s | 144.4 |
| calm | ✓ | 2.68s | 149.1 |
| disgusted | ✓ | 2.28s | 197.6 |
| excited | ✓ | 2.32s | 152.9 |

#### V04_FULL

| Emotion | Status | Duration | Pitch (Hz) |
|---------|--------|----------|------------|
| neutral | ✓ | 2.60s | 140.0 |
| happy | ✓ | 2.88s | 170.2 |
| sad | ✓ | 2.12s | 152.9 |
| angry | ✓ | 2.32s | 137.6 |
| fearful | ✓ | 2.04s | 106.1 |
| surprised | ✓ | 3.36s | 199.3 |
| calm | ✓ | 2.48s | 158.5 |
| disgusted | ✓ | 2.24s | 179.0 |
| excited | ✓ | 3.08s | 158.5 |

#### V05_RAVDESS

| Emotion | Status | Duration | Pitch (Hz) |
|---------|--------|----------|------------|
| neutral | ✓ | 2.92s | 147.5 |
| happy | ✓ | 4.36s | 141.4 |
| sad | ✓ | 2.96s | 123.6 |
| angry | ✓ | 2.48s | 126.1 |
| fearful | ✓ | 2.44s | 133.9 |
| surprised | ✓ | 4.12s | 160.4 |
| calm | ✓ | 2.68s | 173.5 |
| disgusted | ✓ | 2.20s | 155.6 |
| excited | ✓ | 3.64s | 173.0 |

## Notes

- All benchmarks run with `checkpoint_epoch_10.pt`
- Audio generated at 24kHz sample rate
- Prosody extracted using librosa
- New emotions (v0.3): sarcastic, bored, affectionate, contemptuous, awed
