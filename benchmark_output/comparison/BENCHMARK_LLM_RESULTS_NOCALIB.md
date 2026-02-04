# Emotion Recognition Benchmark Results (LLM-based)

**Generated**: 2026-02-01 08:19:16

## Overview

This report evaluates the generated TTS emotions using multiple emotion recognition models:
1. **emotion2vec+ (base)** - Foundation model, best accuracy, multilingual
2. **wav2vec2 (ehcalabres)** - 7 emotions, well-established
3. **wav2vec2 (Dpngtm)** - 8 emotions, trained on multiple datasets

## Summary

### Model Accuracy by Checkpoint

| Checkpoint | Dataset | emotion2vec | wav2vec2 (ehcalabres) | wav2vec2 (Dpngtm) |
|------------|---------|-------------|----------------------|-------------------|
| RAVDESS_SER_NOCALIB | RAVDESS-SER-NOCALIB | 58.6% | 0.0% | 27.6% |

## Detailed Results

### RAVDESS_SER_NOCALIB (RAVDESS-SER-NOCALIB)

#### emotion2vec_base

**Overall Accuracy**: 58.6% (17/29)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| affectionate | 1 | 1 | 100.0% |
| angry | 3 | 5 | 60.0% |
| awed | 0 | 1 | 0.0% |
| bored | 1 | 1 | 100.0% |
| calm | 0 | 1 | 0.0% |
| contemptuous | 0 | 1 | 0.0% |
| disgusted | 0 | 1 | 0.0% |
| excited | 1 | 2 | 50.0% |
| fearful | 1 | 1 | 100.0% |
| happy | 5 | 7 | 71.4% |
| neutral | 1 | 1 | 100.0% |
| sad | 4 | 5 | 80.0% |
| sarcastic | 0 | 1 | 0.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | happy | 1.00 | ✓ |
| intensity_sad_1.5.wav | sad | happy | 1.00 | ✗ |
| intensity_sad_1.0.wav | sad | sad | 1.00 | ✓ |
| basic_surprised_1.0.wav | surprised | neutral | 0.99 | ✗ |
| new_contemptuous_1.0.wav | contemptuous | neutral | 1.00 | ✗ |
| intensity_sad_0.5.wav | sad | sad | 1.00 | ✓ |
| basic_fearful_1.0.wav | fearful | fearful | 0.99 | ✓ |
| transition_sad_happy.wav | happy | sad | 1.00 | ✗ |
| basic_disgusted_1.0.wav | disgusted | neutral | 0.85 | ✗ |
| intensity_sad_0.0.wav | sad | sad | 1.00 | ✓ |

#### wav2vec2_ehcalabres

**Overall Accuracy**: 0.0% (0/29)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| affectionate | 0 | 1 | 0.0% |
| angry | 0 | 5 | 0.0% |
| awed | 0 | 1 | 0.0% |
| bored | 0 | 1 | 0.0% |
| calm | 0 | 1 | 0.0% |
| contemptuous | 0 | 1 | 0.0% |
| disgusted | 0 | 1 | 0.0% |
| excited | 0 | 2 | 0.0% |
| fearful | 0 | 1 | 0.0% |
| happy | 0 | 7 | 0.0% |
| neutral | 0 | 1 | 0.0% |
| sad | 0 | 5 | 0.0% |
| sarcastic | 0 | 1 | 0.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | angry | 0.13 | ✗ |
| intensity_sad_1.5.wav | sad | angry | 0.13 | ✗ |
| intensity_sad_1.0.wav | sad | angry | 0.13 | ✗ |
| basic_surprised_1.0.wav | surprised | angry | 0.13 | ✗ |
| new_contemptuous_1.0.wav | contemptuous | neutral | 0.13 | ✗ |
| intensity_sad_0.5.wav | sad | angry | 0.13 | ✗ |
| basic_fearful_1.0.wav | fearful | angry | 0.13 | ✗ |
| transition_sad_happy.wav | happy | angry | 0.13 | ✗ |
| basic_disgusted_1.0.wav | disgusted | angry | 0.13 | ✗ |
| intensity_sad_0.0.wav | sad | angry | 0.13 | ✗ |

#### dpngtm

**Overall Accuracy**: 27.6% (8/29)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| affectionate | 0 | 1 | 0.0% |
| angry | 1 | 5 | 20.0% |
| awed | 1 | 1 | 100.0% |
| bored | 0 | 1 | 0.0% |
| calm | 0 | 1 | 0.0% |
| contemptuous | 1 | 1 | 100.0% |
| disgusted | 0 | 1 | 0.0% |
| excited | 1 | 2 | 50.0% |
| fearful | 0 | 1 | 0.0% |
| happy | 4 | 7 | 57.1% |
| neutral | 0 | 1 | 0.0% |
| sad | 0 | 5 | 0.0% |
| sarcastic | 0 | 1 | 0.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | calm | 0.87 | ✗ |
| intensity_sad_1.5.wav | sad | happy | 0.52 | ✗ |
| intensity_sad_1.0.wav | sad | calm | 0.89 | ✗ |
| basic_surprised_1.0.wav | surprised | happy | 0.66 | ✗ |
| new_contemptuous_1.0.wav | contemptuous | angry | 0.97 | ✓ |
| intensity_sad_0.5.wav | sad | happy | 0.85 | ✗ |
| basic_fearful_1.0.wav | fearful | happy | 0.95 | ✗ |
| transition_sad_happy.wav | happy | happy | 0.89 | ✓ |
| basic_disgusted_1.0.wav | disgusted | happy | 0.96 | ✗ |
| intensity_sad_0.0.wav | sad | calm | 0.49 | ✗ |

## Notes

### Model Limitations

1. **emotion2vec+**: Recognizes 9 emotions (Angry, Disgusted, Fearful, Happy, Neutral, Other, Sad, Surprised, Unknown)
2. **wav2vec2 (ehcalabres)**: Recognizes 7 emotions (angry, calm, disgust, fear, happy, neutral, sad)
3. **wav2vec2 (Dpngtm)**: Recognizes 8 emotions (angry, calm, disgust, fearful, happy, neutral, sad, surprised)

### New Emotions (v0.3)

The new emotions (sarcastic, bored, affectionate, contemptuous, awed) are not directly recognized by these models.
They are mapped to closest emotions:
- **sarcastic** -> angry, neutral, or other
- **bored** -> neutral, sad, or other
- **affectionate** -> happy, neutral, or other
- **contemptuous** -> angry, disgusted, or other
- **awed** -> surprised, happy, or other

### Evaluation Criteria

- **Correct**: Model prediction matches expected emotion (or close match for unsupported emotions)
- **Confidence**: Model's confidence score for the prediction
- **Close matches**: Emotions like "excited" accepted as "happy", "fearful" as "fear", etc.

## References

1. [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec)
2. [wav2vec2 ehcalabres](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
3. [wav2vec2 Dpngtm](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition)
