# Emotion Recognition Benchmark Results (LLM-based)

**Generated**: 2026-02-03 18:09:03

## Overview

This report evaluates the generated TTS emotions using multiple emotion recognition models:
1. **emotion2vec+ (base)** - Foundation model, best accuracy, multilingual
2. **wav2vec2 (ehcalabres)** - 7 emotions, well-established
3. **wav2vec2 (Dpngtm)** - 8 emotions, trained on multiple datasets

## Summary

### Model Accuracy by Checkpoint

| Checkpoint | Dataset | emotion2vec | wav2vec2 (ehcalabres) | wav2vec2 (Dpngtm) | SpeechBrain (IEMOCAP) |
|------------|---------|-------------|----------------------|-------------------|-----------------------|
| CREMAD_V05_CORE7 | CREMAD-V05-CORE7 | 85.0% | 10.0% | 20.0% | 20.0% |

## Detailed Results

### CREMAD_V05_CORE7 (CREMAD-V05-CORE7)

#### emotion2vec_base

**Overall Accuracy**: 85.0% (17/20)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| angry | 3 | 5 | 60.0% |
| disgusted | 1 | 1 | 100.0% |
| excited | 1 | 1 | 100.0% |
| fearful | 1 | 1 | 100.0% |
| happy | 5 | 5 | 100.0% |
| neutral | 1 | 1 | 100.0% |
| sad | 5 | 5 | 100.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | happy | 1.00 | ✓ |
| intensity_sad_1.5.wav | sad | sad | 1.00 | ✓ |
| intensity_sad_1.0.wav | sad | sad | 1.00 | ✓ |
| basic_surprised_1.0.wav | surprised | neutral | 0.99 | ✗ |
| intensity_sad_0.5.wav | sad | sad | 1.00 | ✓ |
| basic_fearful_1.0.wav | fearful | fearful | 1.00 | ✓ |
| basic_disgusted_1.0.wav | disgusted | disgusted | 0.97 | ✓ |
| intensity_sad_0.0.wav | sad | sad | 1.00 | ✓ |
| basic_angry_1.0.wav | angry | neutral | 1.00 | ✗ |
| intensity_angry_1.0.wav | angry | neutral | 0.96 | ✗ |

#### wav2vec2_ehcalabres

**Overall Accuracy**: 10.0% (2/20)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| angry | 0 | 5 | 0.0% |
| disgusted | 0 | 1 | 0.0% |
| excited | 0 | 1 | 0.0% |
| fearful | 0 | 1 | 0.0% |
| happy | 0 | 5 | 0.0% |
| neutral | 0 | 1 | 0.0% |
| sad | 2 | 5 | 40.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | surprised | 0.13 | ✗ |
| intensity_sad_1.5.wav | sad | angry | 0.13 | ✗ |
| intensity_sad_1.0.wav | sad | sad | 0.13 | ✓ |
| basic_surprised_1.0.wav | surprised | sad | 0.13 | ✗ |
| intensity_sad_0.5.wav | sad | sad | 0.13 | ✓ |
| basic_fearful_1.0.wav | fearful | neutral | 0.13 | ✗ |
| basic_disgusted_1.0.wav | disgusted | neutral | 0.13 | ✗ |
| intensity_sad_0.0.wav | sad | angry | 0.13 | ✗ |
| basic_angry_1.0.wav | angry | neutral | 0.13 | ✗ |
| intensity_angry_1.0.wav | angry | neutral | 0.13 | ✗ |

#### dpngtm

**Overall Accuracy**: 20.0% (4/20)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| angry | 0 | 5 | 0.0% |
| disgusted | 0 | 1 | 0.0% |
| excited | 1 | 1 | 100.0% |
| fearful | 0 | 1 | 0.0% |
| happy | 3 | 5 | 60.0% |
| neutral | 0 | 1 | 0.0% |
| sad | 0 | 5 | 0.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | disgusted | 0.65 | ✗ |
| intensity_sad_1.5.wav | sad | happy | 0.85 | ✗ |
| intensity_sad_1.0.wav | sad | happy | 0.96 | ✗ |
| basic_surprised_1.0.wav | surprised | disgusted | 0.41 | ✗ |
| intensity_sad_0.5.wav | sad | happy | 0.56 | ✗ |
| basic_fearful_1.0.wav | fearful | angry | 0.68 | ✗ |
| basic_disgusted_1.0.wav | disgusted | angry | 0.74 | ✗ |
| intensity_sad_0.0.wav | sad | happy | 0.96 | ✗ |
| basic_angry_1.0.wav | angry | happy | 0.95 | ✗ |
| intensity_angry_1.0.wav | angry | disgusted | 0.85 | ✗ |

#### speechbrain_iemocap

**Overall Accuracy**: 20.0% (4/20)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| angry | 2 | 5 | 40.0% |
| disgusted | 0 | 1 | 0.0% |
| excited | 0 | 1 | 0.0% |
| fearful | 0 | 1 | 0.0% |
| happy | 1 | 5 | 20.0% |
| neutral | 1 | 1 | 100.0% |
| sad | 0 | 5 | 0.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | happy | 1.00 | ✓ |
| intensity_sad_1.5.wav | sad | angry | 1.00 | ✗ |
| intensity_sad_1.0.wav | sad | happy | 1.00 | ✗ |
| basic_surprised_1.0.wav | surprised | neutral | 0.67 | ✗ |
| intensity_sad_0.5.wav | sad | happy | 1.00 | ✗ |
| basic_fearful_1.0.wav | fearful | neutral | 1.00 | ✗ |
| basic_disgusted_1.0.wav | disgusted | angry | 1.00 | ✗ |
| intensity_sad_0.0.wav | sad | angry | 1.00 | ✗ |
| basic_angry_1.0.wav | angry | neutral | 1.00 | ✗ |
| intensity_angry_1.0.wav | angry | happy | 1.00 | ✗ |

## Notes

### Model Limitations

1. **emotion2vec+**: Recognizes 9 emotions (Angry, Disgusted, Fearful, Happy, Neutral, Other, Sad, Surprised, Unknown)
2. **wav2vec2 (ehcalabres)**: Recognizes 7 emotions (angry, calm, disgust, fear, happy, neutral, sad)
3. **wav2vec2 (Dpngtm)**: Recognizes 8 emotions (angry, calm, disgust, fearful, happy, neutral, sad, surprised)
4. **SpeechBrain IEMOCAP**: Recognizes 4 emotions (angry, happy, sad, neutral)

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
