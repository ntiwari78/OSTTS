"""
Emotion Intensity Calibration (V0.4)

This module provides pre-calibrated intensity multipliers per emotion
based on SER (Speech Emotion Recognition) analysis.

The calibration compensates for emotions that need stronger/weaker
expression to be correctly recognized by SER models.

Key findings:
- Angry at default intensity (1.0) is often misclassified as neutral
- Disgusted needs stronger markers for recognition
- Calm needs reduction to differentiate from neutral
"""

from typing import Dict, Optional


# =============================================================================
# Pre-calibrated Intensity Multipliers
# =============================================================================

CALIBRATED_INTENSITIES: Dict[str, float] = {
    # High-energy emotions need intensity boost
    'angry': 1.5,       # Critical fix: default intensity → neutral
    'disgusted': 1.4,   # Needs clearer markers for recognition
    'excited': 1.3,     # Needs energy boost
    'happy': 1.2,       # Slight pitch/energy boost
    'fearful': 1.1,     # Minor enhancement for tremor
    'shout': 1.3,       # Needs energy boost

    # Well-performing emotions (no change)
    'surprised': 1.0,   # Works well at default
    'sad': 1.0,         # Works well at default
    'neutral': 1.0,     # Baseline reference

    # Low-arousal emotions need reduction to differentiate from neutral
    'calm': 0.8,        # Reduce to avoid neutral confusion
    'whisper': 0.9,     # Slight reduction
    'bored': 0.9,       # Slight reduction

    # New emotions (v0.2.1) - conservative calibration
    'sarcastic': 1.1,   # Slight boost for intonation
    'affectionate': 1.0, # Works at default
    'contemptuous': 1.2, # Needs clearer markers
    'awed': 1.1,        # Slight boost
}


# =============================================================================
# Calibration Functions
# =============================================================================

def get_calibrated_intensity(
    emotion: str,
    user_intensity: float = 1.0,
    use_calibration: bool = True,
    custom_calibration: Optional[Dict[str, float]] = None,
) -> float:
    """
    Apply calibration to user-specified intensity.

    The calibration multiplier adjusts the intensity to optimize
    for SER recognition while respecting user intent.

    Args:
        emotion: Emotion name (case-insensitive)
        user_intensity: User-specified intensity (0.0 - 2.0)
        use_calibration: Whether to apply calibration (default: True)
        custom_calibration: Optional custom calibration dict to override defaults

    Returns:
        Calibrated intensity value, clamped to [0.1, 2.5]

    Example:
        >>> get_calibrated_intensity('angry', 1.0)
        1.5
        >>> get_calibrated_intensity('calm', 1.0)
        0.8
        >>> get_calibrated_intensity('happy', 1.5)
        1.8
    """
    if not use_calibration:
        return user_intensity

    # Use custom calibration if provided, otherwise use defaults
    calibration_map = custom_calibration or CALIBRATED_INTENSITIES

    # Get calibration multiplier (default to 1.0 for unknown emotions)
    calibration = calibration_map.get(emotion.lower(), 1.0)

    # Apply calibration
    calibrated = user_intensity * calibration

    # Clamp to reasonable range
    return max(0.1, min(2.5, calibrated))


def get_calibration_multiplier(emotion: str) -> float:
    """
    Get the calibration multiplier for an emotion.

    Args:
        emotion: Emotion name (case-insensitive)

    Returns:
        Calibration multiplier (1.0 if unknown)
    """
    return CALIBRATED_INTENSITIES.get(emotion.lower(), 1.0)


def update_calibration(emotion: str, multiplier: float) -> None:
    """
    Update calibration for an emotion (for runtime tuning).

    Args:
        emotion: Emotion name
        multiplier: New calibration multiplier

    Note:
        This modifies the global calibration dict.
        Use with caution in production.
    """
    if 0.1 <= multiplier <= 3.0:
        CALIBRATED_INTENSITIES[emotion.lower()] = multiplier
    else:
        raise ValueError(f"Multiplier must be in [0.1, 3.0], got {multiplier}")


def get_all_calibrations() -> Dict[str, float]:
    """
    Get all calibration multipliers.

    Returns:
        Dictionary mapping emotion names to calibration multipliers
    """
    return CALIBRATED_INTENSITIES.copy()


# =============================================================================
# Intensity Ranges for Different Use Cases
# =============================================================================

INTENSITY_PRESETS = {
    'subtle': {
        # For natural, understated emotional expression
        'multiplier': 0.7,
        'description': 'Subtle, natural emotional expression',
    },
    'normal': {
        # Standard calibrated intensity
        'multiplier': 1.0,
        'description': 'Normal calibrated intensity',
    },
    'expressive': {
        # For clear, recognizable emotional expression
        'multiplier': 1.3,
        'description': 'Clear, expressive emotional expression',
    },
    'dramatic': {
        # For highly exaggerated emotional expression
        'multiplier': 1.6,
        'description': 'Dramatic, exaggerated emotional expression',
    },
}


def get_preset_intensity(
    emotion: str,
    preset: str = 'normal',
    user_intensity: float = 1.0,
) -> float:
    """
    Get intensity using a preset level.

    Args:
        emotion: Emotion name
        preset: One of 'subtle', 'normal', 'expressive', 'dramatic'
        user_intensity: Base user intensity

    Returns:
        Calibrated intensity with preset applied
    """
    if preset not in INTENSITY_PRESETS:
        preset = 'normal'

    preset_multiplier = INTENSITY_PRESETS[preset]['multiplier']
    base_calibrated = get_calibrated_intensity(emotion, user_intensity)

    return max(0.1, min(2.5, base_calibrated * preset_multiplier))


# =============================================================================
# Emotion-Specific Intensity Recommendations
# =============================================================================

INTENSITY_RECOMMENDATIONS = {
    'angry': {
        'min': 1.2,
        'recommended': 1.5,
        'max': 2.0,
        'notes': 'Needs strong energy and tension for recognition',
    },
    'sad': {
        'min': 0.8,
        'recommended': 1.0,
        'max': 1.3,
        'notes': 'Works well across intensity range',
    },
    'happy': {
        'min': 1.0,
        'recommended': 1.2,
        'max': 1.6,
        'notes': 'Slight boost helps with pitch and energy',
    },
    'fearful': {
        'min': 0.9,
        'recommended': 1.1,
        'max': 1.5,
        'notes': 'Needs tremor and tension markers',
    },
    'disgusted': {
        'min': 1.2,
        'recommended': 1.4,
        'max': 1.8,
        'notes': 'Needs strong nasal and tension markers',
    },
    'surprised': {
        'min': 0.8,
        'recommended': 1.0,
        'max': 1.4,
        'notes': 'Works well at default, high pitch range',
    },
    'neutral': {
        'min': 0.8,
        'recommended': 1.0,
        'max': 1.0,
        'notes': 'Baseline, avoid going above 1.0',
    },
    'calm': {
        'min': 0.6,
        'recommended': 0.8,
        'max': 1.0,
        'notes': 'Reduce to differentiate from neutral',
    },
    'excited': {
        'min': 1.1,
        'recommended': 1.3,
        'max': 1.8,
        'notes': 'Needs high energy and fast rate',
    },
}


def get_intensity_recommendation(emotion: str) -> Dict:
    """
    Get intensity recommendation for an emotion.

    Args:
        emotion: Emotion name

    Returns:
        Dictionary with min, recommended, max intensities and notes
    """
    default = {
        'min': 0.8,
        'recommended': 1.0,
        'max': 1.5,
        'notes': 'No specific recommendation available',
    }
    return INTENSITY_RECOMMENDATIONS.get(emotion.lower(), default)
