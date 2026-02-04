# Emotion Recognition Benchmark Results (LLM-based)

**Generated**: 2026-02-03 20:32:02

## Overview

This report evaluates the generated TTS emotions using emotion2vec+ (base) — the only SER model
that reliably generalizes to synthetic TTS audio.

## Summary

### Model Accuracy by Checkpoint

| Checkpoint | Dataset | emotion2vec |
|------------|---------|-------------|
| COMBINED_V07 | Combined-V07 | 85.0% |

## Detailed Results

### COMBINED_V07 (Combined-V07)

#### emotion2vec_base

**Overall Accuracy**: 85.0% (17/20)

| Emotion | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| angry | 4 | 5 | 80.0% |
| disgusted | 1 | 1 | 100.0% |
| excited | 1 | 1 | 100.0% |
| fearful | 1 | 1 | 100.0% |
| happy | 4 | 5 | 80.0% |
| neutral | 1 | 1 | 100.0% |
| sad | 5 | 5 | 100.0% |
| surprised | 0 | 1 | 0.0% |

**Sample Predictions**:

| File | Expected | Predicted | Confidence | Correct |
|------|----------|-----------|------------|--------|
| basic_happy_1.0.wav | happy | happy | 1.00 | ✓ |
| intensity_sad_1.5.wav | sad | sad | 1.00 | ✓ |
| intensity_sad_1.0.wav | sad | sad | 1.00 | ✓ |
| basic_surprised_1.0.wav | surprised | neutral | 1.00 | ✗ |
| intensity_sad_0.5.wav | sad | sad | 1.00 | ✓ |
| basic_fearful_1.0.wav | fearful | fearful | 0.98 | ✓ |
| basic_disgusted_1.0.wav | disgusted | disgusted | 0.92 | ✓ |
| intensity_sad_0.0.wav | sad | sad | 1.00 | ✓ |
| basic_angry_1.0.wav | angry | angry | 0.68 | ✓ |
| intensity_angry_1.0.wav | angry | angry | 0.59 | ✓ |

## Notes

### Model Details

- **emotion2vec+**: Recognizes 9 emotions (Angry, Disgusted, Fearful, Happy, Neutral, Other, Sad, Surprised, Unknown)

### Evaluation Criteria

- **Correct**: Model prediction matches expected emotion (or close match for unsupported emotions)
- **Confidence**: Model's confidence score for the prediction
- **Close matches**: Emotions like "excited" accepted as "happy", "fearful" as "fear", etc.

## References

1. [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec)
