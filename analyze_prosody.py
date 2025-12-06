#!/usr/bin/env python3
"""
analyze_prosody.py

Analyze prosodic features (pitch, energy, duration) for emotion validation.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse


def extract_prosodic_features(audio_path, sr=24000):
    """
    Extract prosodic features from audio file.

    Returns:
        Dict with pitch, energy, and duration features
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Duration
    duration = len(y) / sr

    # Pitch (F0) using PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr
    )
    f0_clean = f0[~np.isnan(f0)]

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # Zero crossing rate (correlates with pitch perception)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    features = {
        "duration_s": float(duration),
        "pitch": {
            "mean_hz": float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0,
            "std_hz": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0,
            "min_hz": float(np.min(f0_clean)) if len(f0_clean) > 0 else 0,
            "max_hz": float(np.max(f0_clean)) if len(f0_clean) > 0 else 0,
            "range_hz": float(np.max(f0_clean) - np.min(f0_clean)) if len(f0_clean) > 0 else 0,
            "voiced_fraction": float(np.mean(voiced_flag)),
        },
        "energy": {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "max": float(np.max(rms)),
            "dynamic_range_db": float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))),
        },
        "spectral": {
            "centroid_mean": float(np.mean(spectral_centroid)),
            "centroid_std": float(np.std(spectral_centroid)),
            "rolloff_mean": float(np.mean(spectral_rolloff)),
        },
        "tempo_bpm": float(tempo) if isinstance(tempo, (int, float, np.floating)) else float(tempo[0]) if len(tempo) > 0 else 0,
        "zcr_mean": float(np.mean(zcr)),
    }

    return features


def analyze_emotion_prosody(audio_dir, patterns=None):
    """
    Analyze prosodic features for all generated emotion audio.
    """
    audio_dir = Path(audio_dir)
    results = {}

    if patterns is None:
        patterns = ["benchmark_*_*.wav"]

    for pattern in patterns:
        for audio_file in sorted(audio_dir.glob(pattern)):
            # Extract emotion from filename
            stem = audio_file.stem
            parts = stem.split("_")

            # Handle different filename patterns
            if "blend" in parts:
                emotion = f"blend_{parts[-1]}"  # blend_bittersweet
            elif "intensity" in parts:
                emotion = f"intensity_{parts[2]}_{parts[3]}"  # intensity_excited_0.5
            elif len(parts) >= 2:
                # benchmark_happy_positive -> happy
                emotion = parts[1]
            else:
                emotion = "unknown"

            print(f"Analyzing: {audio_file.name}...")
            try:
                features = extract_prosodic_features(audio_file)
                features["file"] = audio_file.name
                results[stem] = features
            except Exception as e:
                print(f"  Error: {e}")
                results[stem] = {"error": str(e), "file": audio_file.name}

    return results


def plot_prosody_comparison(results, output_path="prosody_comparison.png"):
    """
    Create visualization comparing prosodic features across emotions.
    """
    # Filter only basic emotion results (not intensity or blend)
    basic_results = {}
    for key, val in results.items():
        if "intensity" not in key and "blend" not in key and "error" not in val:
            parts = key.split("_")
            if len(parts) >= 2:
                emotion = parts[1]
                basic_results[emotion] = val

    if not basic_results:
        print("No basic emotion results to plot")
        return

    emotions = list(basic_results.keys())

    # Extract features for plotting
    pitch_means = [basic_results[e]["pitch"]["mean_hz"] for e in emotions]
    pitch_stds = [basic_results[e]["pitch"]["std_hz"] for e in emotions]
    energy_means = [basic_results[e]["energy"]["mean"] for e in emotions]
    tempos = [basic_results[e]["tempo_bpm"] for e in emotions]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pitch mean
    ax1 = axes[0, 0]
    bars1 = ax1.bar(emotions, pitch_means, color='steelblue')
    ax1.set_ylabel('Hz')
    ax1.set_title('Mean Pitch (F0)')
    ax1.tick_params(axis='x', rotation=45)

    # Pitch variation
    ax2 = axes[0, 1]
    bars2 = ax2.bar(emotions, pitch_stds, color='coral')
    ax2.set_ylabel('Hz')
    ax2.set_title('Pitch Variation (Std)')
    ax2.tick_params(axis='x', rotation=45)

    # Energy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(emotions, energy_means, color='green')
    ax3.set_ylabel('RMS')
    ax3.set_title('Mean Energy')
    ax3.tick_params(axis='x', rotation=45)

    # Tempo
    ax4 = axes[1, 1]
    bars4 = ax4.bar(emotions, tempos, color='purple')
    ax4.set_ylabel('BPM')
    ax4.set_title('Tempo')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_intensity_comparison(results, output_path="intensity_comparison.png"):
    """
    Plot how features change with intensity.
    """
    # Filter intensity results
    intensity_results = []
    for key, val in results.items():
        if "intensity" in key and "error" not in val:
            parts = key.split("_")
            if len(parts) >= 4:
                try:
                    intensity = float(parts[3])
                    intensity_results.append((intensity, val))
                except ValueError:
                    continue

    if not intensity_results:
        print("No intensity results to plot")
        return

    # Sort by intensity
    intensity_results.sort(key=lambda x: x[0])
    intensities = [r[0] for r in intensity_results]

    pitch_means = [r[1]["pitch"]["mean_hz"] for r in intensity_results]
    pitch_stds = [r[1]["pitch"]["std_hz"] for r in intensity_results]
    energy_means = [r[1]["energy"]["mean"] for r in intensity_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    ax1.plot(intensities, pitch_means, 'o-', markersize=10, linewidth=2)
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Hz')
    ax1.set_title('Mean Pitch vs Intensity')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(intensities, pitch_stds, 'o-', markersize=10, linewidth=2, color='coral')
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Hz')
    ax2.set_title('Pitch Variation vs Intensity')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(intensities, energy_means, 'o-', markersize=10, linewidth=2, color='green')
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('RMS')
    ax3.set_title('Mean Energy vs Intensity')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nIntensity plot saved to: {output_path}")
    plt.close()


def generate_prosody_report(audio_dir, output_path="prosody_report.json"):
    """Generate comprehensive prosody analysis report."""
    results = analyze_emotion_prosody(audio_dir)

    if not results:
        print("No audio files found to analyze!")
        return None

    # Save JSON report
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {output_path}")

    # Create visualizations
    plot_dir = Path(audio_dir)
    plot_prosody_comparison(results, plot_dir / "prosody_comparison.png")
    plot_intensity_comparison(results, plot_dir / "intensity_comparison.png")

    # Print summary
    print("\n" + "=" * 60)
    print("PROSODIC FEATURE SUMMARY BY EMOTION")
    print("=" * 60)

    # Group by basic emotion
    basic_emotions = {}
    for key, features in results.items():
        if "intensity" not in key and "blend" not in key and "error" not in features:
            parts = key.split("_")
            if len(parts) >= 2:
                emotion = parts[1]
                basic_emotions[emotion] = features

    for emotion, features in sorted(basic_emotions.items()):
        print(f"\n{emotion.upper()}:")
        print(f"  Pitch: {features['pitch']['mean_hz']:.1f} Hz (std: {features['pitch']['std_hz']:.1f})")
        print(f"  Energy: {features['energy']['mean']:.4f} (range: {features['energy']['dynamic_range_db']:.1f} dB)")
        print(f"  Tempo: {features['tempo_bpm']:.1f} BPM")
        print(f"  Duration: {features['duration_s']:.2f}s")

    # Print intensity analysis
    print("\n" + "=" * 60)
    print("INTENSITY VARIATION ANALYSIS")
    print("=" * 60)

    intensity_data = []
    for key, features in results.items():
        if "intensity" in key and "error" not in features:
            parts = key.split("_")
            if len(parts) >= 4:
                try:
                    intensity = float(parts[3])
                    intensity_data.append((intensity, features))
                except ValueError:
                    continue

    if intensity_data:
        intensity_data.sort(key=lambda x: x[0])
        for intensity, features in intensity_data:
            print(f"\nIntensity {intensity:.1f}:")
            print(f"  Pitch: {features['pitch']['mean_hz']:.1f} Hz (std: {features['pitch']['std_hz']:.1f})")
            print(f"  Energy: {features['energy']['mean']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="benchmark_output",
                       help="Directory containing generated audio files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for prosody report (default: audio_dir/prosody_report.json)")
    args = parser.parse_args()

    output_path = args.output if args.output else Path(args.audio_dir) / "prosody_report.json"
    generate_prosody_report(args.audio_dir, output_path)
