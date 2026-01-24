# Emotion TTS Benchmark Results v0.3

**Generated**: 2026-01-24 00:00:36

## Summary

| Checkpoint | Dataset | Tests | Passed | Failed | Success Rate |
|------------|---------|-------|--------|--------|--------------|
| RAVDESS | RAVDESS | 29 | 29 | 0 | 100.0% |
| CREMAD | CREMA-D | 27 | 27 | 0 | 100.0% |
| IESC | IESC | 9 | 9 | 0 | 100.0% |

## Prosody Analysis by Emotion

### Pitch (Mean Hz)

| Emotion | RAVDESS | CREMA-D | IESC |
|---------|---------|---------|------|
| affectionate | 142.8 | 141.6 | - |
| angry | 228.7 | 215.9 | 171.3 |
| awed | 172.4 | 141.8 | - |
| bored | 128.9 | 145.1 | - |
| calm | 129.2 | - | - |
| contemptuous | 155.0 | 139.0 | - |
| disgusted | 183.9 | 126.8 | - |
| excited | 145.5 | 134.4 | - |
| fearful | 125.4 | 137.6 | - |
| happy | 210.8 | 110.9 | 198.4 |
| neutral | 154.5 | 139.6 | 153.2 |
| sad | 140.8 | 179.5 | 236.5 |
| sarcastic | 174.0 | 146.2 | - |
| surprised | 181.6 | - | 157.1 |

### Energy (Mean RMS)

| Emotion | RAVDESS | CREMA-D | IESC |
|---------|---------|---------|------|
| affectionate | 0.1110 | 0.1322 | - |
| angry | 0.0721 | 0.0550 | 0.0930 |
| awed | 0.1106 | 0.1033 | - |
| bored | 0.0751 | 0.0816 | - |
| calm | 0.0810 | - | - |
| contemptuous | 0.1087 | 0.1141 | - |
| disgusted | 0.1110 | 0.0730 | - |
| excited | 0.0948 | 0.0801 | - |
| fearful | 0.1135 | 0.1022 | - |
| happy | 0.0530 | 0.0351 | 0.0913 |
| neutral | 0.0921 | 0.0740 | 0.1064 |
| sad | 0.0790 | 0.0795 | 0.0860 |
| sarcastic | 0.1064 | 0.0971 | - |
| surprised | 0.1149 | - | 0.0982 |

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

## Notes

- All benchmarks run with `checkpoint_epoch_10.pt`
- Audio generated at 24kHz sample rate
- Prosody extracted using librosa
- New emotions (v0.3): sarcastic, bored, affectionate, contemptuous, awed
