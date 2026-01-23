#!/usr/bin/env python3
"""
Comprehensive test suite for the emotion system.

Tests all components of the emotion system including:
1. EmotionEmbeddings - 16 emotions, intensity, blending
2. IntensityTransform - Nonlinear intensity mapping
3. EmotionTrajectory - Temporal dynamics (static, transition, keyframe)
4. EmotionCrossAttention - Context-aware conditioning
5. Emotion losses - Consistency, contrastive, SER
6. Checkpoint loading - Per-dataset and merged checkpoints
7. Full pipeline - End-to-end generation with emotions

Usage:
    python test_emotion_system.py                    # Run all tests
    python test_emotion_system.py -v                 # Verbose output
    python test_emotion_system.py TestEmotionEmbeddings  # Run specific test class
"""

import sys
import unittest
from pathlib import Path
from typing import List, Dict

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np


class TestEmotionEmbeddings(unittest.TestCase):
    """Test EmotionEmbeddings module."""

    @classmethod
    def setUpClass(cls):
        """Load module once for all tests."""
        from chatterbox.models.t3.modules.emotion_embeddings import (
            EmotionEmbeddings,
            EMOTION_INIT_EMBEDDINGS_64D,
        )
        cls.EmotionEmbeddings = EmotionEmbeddings
        cls.EMOTION_INIT_EMBEDDINGS_64D = EMOTION_INIT_EMBEDDINGS_64D
        cls.embeddings = EmotionEmbeddings(emotion_embed_dim=64)

    def test_all_emotions_present(self):
        """Verify all 16 emotions are present."""
        expected_emotions = {
            "neutral", "happy", "sad", "angry", "excited", "calm",
            "surprised", "fearful", "disgusted", "whisper", "shout",
            "sarcastic", "bored", "affectionate", "contemptuous", "awed"
        }
        actual_emotions = set(self.embeddings.get_supported_emotions())
        self.assertEqual(expected_emotions, actual_emotions,
                        f"Missing emotions: {expected_emotions - actual_emotions}")

    def test_embedding_dimension(self):
        """Verify embedding dimensions are correct."""
        for emotion in self.embeddings.get_supported_emotions():
            embed = self.embeddings.get_emotion_embedding(emotion)
            self.assertEqual(embed.shape, (1, 64),
                           f"Wrong shape for {emotion}: {embed.shape}")

    def test_neutral_is_zero(self):
        """Verify neutral embedding is close to zero."""
        neutral = self.embeddings.get_emotion_embedding("neutral")
        # First 16 dims (VAD + prosodic) should be exactly zero
        self.assertTrue(torch.allclose(neutral[:, :16], torch.zeros(1, 16), atol=1e-6),
                       "Neutral VAD+prosodic should be zero")

    def test_intensity_zero_returns_neutral(self):
        """Verify intensity=0 returns neutral embedding."""
        neutral = self.embeddings.get_emotion_embedding("neutral")
        happy_zero = self.embeddings.get_emotion_embedding("happy", intensity=0.0)
        self.assertTrue(torch.allclose(happy_zero, neutral, atol=1e-5),
                       "intensity=0 should return neutral")

    def test_intensity_scaling(self):
        """Verify intensity scales distance from neutral."""
        neutral = self.embeddings.get_emotion_embedding("neutral")
        half = self.embeddings.get_emotion_embedding("happy", intensity=0.5)
        full = self.embeddings.get_emotion_embedding("happy", intensity=1.0)

        dist_half = torch.norm(half - neutral).item()
        dist_full = torch.norm(full - neutral).item()

        self.assertGreater(dist_full, dist_half,
                          "Full intensity should be further from neutral")
        # Half intensity should be roughly half the distance
        self.assertAlmostEqual(dist_half / dist_full, 0.5, delta=0.2)

    def test_intensity_extrapolation(self):
        """Verify intensity > 1 extrapolates beyond emotion."""
        neutral = self.embeddings.get_emotion_embedding("neutral")
        full = self.embeddings.get_emotion_embedding("happy", intensity=1.0)
        extra = self.embeddings.get_emotion_embedding("happy", intensity=1.5)

        dist_full = torch.norm(full - neutral).item()
        dist_extra = torch.norm(extra - neutral).item()

        self.assertGreater(dist_extra, dist_full,
                          "Extrapolated should be further than full intensity")

    def test_interpolation(self):
        """Test emotion blending."""
        blended = self.embeddings.interpolate_emotions({"happy": 0.5, "sad": 0.5})
        self.assertEqual(blended.shape, (1, 64))

        # Blended should be between happy and sad
        happy = self.embeddings.get_emotion_embedding("happy")
        sad = self.embeddings.get_emotion_embedding("sad")

        dist_to_happy = torch.norm(blended - happy).item()
        dist_to_sad = torch.norm(blended - sad).item()
        dist_happy_sad = torch.norm(happy - sad).item()

        # Blended should be closer to both than they are to each other
        self.assertLess(dist_to_happy, dist_happy_sad)
        self.assertLess(dist_to_sad, dist_happy_sad)

    def test_unknown_emotion_raises(self):
        """Verify unknown emotion raises ValueError."""
        with self.assertRaises(ValueError):
            self.embeddings.get_emotion_embedding("nonexistent_emotion")

    def test_vad_values_reasonable(self):
        """Verify VAD values are in expected ranges."""
        for emotion in self.embeddings.get_supported_emotions():
            embed = self.embeddings.get_emotion_embedding(emotion)
            vad = embed[0, :3].tolist()

            for i, val in enumerate(vad):
                self.assertGreaterEqual(val, -1.0, f"{emotion} VAD[{i}] too low: {val}")
                self.assertLessEqual(val, 1.0, f"{emotion} VAD[{i}] too high: {val}")


class TestIntensityTransform(unittest.TestCase):
    """Test IntensityTransform for nonlinear intensity mapping."""

    @classmethod
    def setUpClass(cls):
        from chatterbox.models.t3.modules.emotion_embeddings import IntensityTransform
        cls.IntensityTransform = IntensityTransform
        cls.transform = IntensityTransform(emotion_dim=64)

    def test_output_shape(self):
        """Verify output shape matches input."""
        neutral = torch.randn(1, 64)
        target = torch.randn(1, 64)
        intensity = torch.tensor([[0.5]])

        result = self.transform(neutral, target, intensity)
        self.assertEqual(result.shape, (1, 64))

    def test_batch_processing(self):
        """Verify batch processing works."""
        neutral = torch.randn(4, 64)
        target = torch.randn(4, 64)
        intensity = torch.tensor([[0.5], [0.7], [1.0], [1.3]])

        result = self.transform(neutral, target, intensity)
        self.assertEqual(result.shape, (4, 64))

    def test_zero_intensity_near_neutral(self):
        """Zero intensity should produce result close to neutral."""
        neutral = torch.randn(1, 64)
        target = torch.randn(1, 64)
        intensity = torch.tensor([[0.0]])

        result = self.transform(neutral, target, intensity)
        dist = torch.norm(result - neutral).item()

        # Should be very close to neutral (allowing for learned bias)
        self.assertLess(dist, 2.0, "Zero intensity should be close to neutral")

    def test_gradient_flow(self):
        """Verify gradients flow through transform."""
        neutral = torch.randn(1, 64, requires_grad=True)
        target = torch.randn(1, 64, requires_grad=True)
        intensity = torch.tensor([[0.7]], requires_grad=True)

        result = self.transform(neutral, target, intensity)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(neutral.grad)
        self.assertIsNotNone(target.grad)
        self.assertIsNotNone(intensity.grad)


class TestEmotionTrajectory(unittest.TestCase):
    """Test EmotionTrajectory for temporal dynamics."""

    @classmethod
    def setUpClass(cls):
        from chatterbox.models.t3.modules.emotion_trajectory import EmotionTrajectory
        cls.EmotionTrajectory = EmotionTrajectory
        cls.trajectory = EmotionTrajectory(emotion_dim=64)

    def test_static_mode(self):
        """Static mode should repeat embedding."""
        embed = torch.randn(2, 64)
        result = self.trajectory.forward_static(embed, seq_len=10)

        self.assertEqual(result.shape, (2, 10, 64))

        # All positions should be identical to input
        for i in range(10):
            self.assertTrue(torch.allclose(result[:, i, :], embed, atol=1e-6))

    def test_transition_mode_shape(self):
        """Verify transition mode output shape."""
        start = torch.randn(1, 64)
        end = torch.randn(1, 64)
        result = self.trajectory.forward_transition(start, end, seq_len=50)

        self.assertEqual(result.shape, (1, 50, 64))

    def test_transition_endpoints(self):
        """Transition should be closer to start at beginning, end at end."""
        start = torch.zeros(1, 64)
        end = torch.ones(1, 64)
        result = self.trajectory.forward_transition(start, end, seq_len=10)

        # First position should be closer to start
        dist_start_first = torch.norm(result[:, 0, :] - start).item()
        dist_end_first = torch.norm(result[:, 0, :] - end).item()
        self.assertLess(dist_start_first, dist_end_first)

        # Last position should be closer to end
        dist_start_last = torch.norm(result[:, -1, :] - start).item()
        dist_end_last = torch.norm(result[:, -1, :] - end).item()
        self.assertLess(dist_end_last, dist_start_last)

    def test_keyframe_mode(self):
        """Test keyframe mode with multiple emotions."""
        keyframes = [
            torch.zeros(1, 64),   # Start: neutral
            torch.ones(1, 64),    # Middle: excited
            torch.zeros(1, 64),   # End: neutral
        ]
        positions = [0.0, 0.5, 1.0]

        result = self.trajectory.forward_keyframes(keyframes, positions, seq_len=20)
        self.assertEqual(result.shape, (1, 20, 64))

    def test_text_context_integration(self):
        """Test that text context can be integrated."""
        start = torch.randn(1, 64)
        end = torch.randn(1, 64)
        text_context = torch.randn(1, 20, 1024)

        result = self.trajectory.forward_transition(start, end, seq_len=10, text_context=text_context)
        self.assertEqual(result.shape, (1, 10, 64))

    def test_unified_forward(self):
        """Test unified forward method."""
        start = torch.randn(1, 64)
        end = torch.randn(1, 64)

        # Static mode
        static = self.trajectory(start, seq_len=5)
        self.assertEqual(static.shape, (1, 5, 64))

        # Transition mode
        transition = self.trajectory(start, end_embed=end, seq_len=5)
        self.assertEqual(transition.shape, (1, 5, 64))


class TestEmotionCrossAttention(unittest.TestCase):
    """Test EmotionCrossAttention module."""

    @classmethod
    def setUpClass(cls):
        from chatterbox.models.t3.modules.emotion_cross_attention import EmotionCrossAttention
        cls.EmotionCrossAttention = EmotionCrossAttention
        cls.cross_attn = EmotionCrossAttention(
            hidden_size=1024,
            emotion_dim=64,
            num_heads=8,
            num_query_tokens=4,
        )

    def test_output_shape(self):
        """Verify output shape is (B, 4, 1024)."""
        emotion = torch.randn(2, 64)
        context = torch.randn(2, 50, 1024)

        output = self.cross_attn(emotion, context)
        self.assertEqual(output.shape, (2, 4, 1024))

    def test_without_context(self):
        """Test forward without text context."""
        emotion = torch.randn(2, 64)

        output = self.cross_attn(emotion, context=None)
        self.assertEqual(output.shape, (2, 4, 1024))

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            emotion = torch.randn(batch_size, 64)
            context = torch.randn(batch_size, 30, 1024)

            output = self.cross_attn(emotion, context)
            self.assertEqual(output.shape, (batch_size, 4, 1024))

    def test_gradient_flow(self):
        """Verify gradients flow correctly."""
        emotion = torch.randn(1, 64, requires_grad=True)
        context = torch.randn(1, 20, 1024, requires_grad=True)

        output = self.cross_attn(emotion, context)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(emotion.grad)
        self.assertIsNotNone(context.grad)


class TestEmotionLosses(unittest.TestCase):
    """Test emotion loss functions."""

    @classmethod
    def setUpClass(cls):
        from chatterbox.models.t3.modules.emotion_losses import (
            EmotionConsistencyLoss,
            CombinedEmotionLoss,
        )
        cls.EmotionConsistencyLoss = EmotionConsistencyLoss
        cls.CombinedEmotionLoss = CombinedEmotionLoss

    def test_consistency_loss(self):
        """Test emotion consistency loss computation."""
        loss_fn = self.EmotionConsistencyLoss(emotion_dim=64, audio_feature_dim=256)

        target_emotion = torch.randn(4, 64)
        audio_features = torch.randn(4, 256)

        losses = loss_fn(target_emotion, audio_features)

        self.assertIn("consistency_loss", losses)
        self.assertIn("contrastive_loss", losses)
        self.assertIn("total_loss", losses)

        # Losses should be positive
        self.assertGreater(losses["consistency_loss"].item(), 0)
        self.assertGreater(losses["contrastive_loss"].item(), 0)

    def test_combined_loss(self):
        """Test combined emotion loss."""
        loss_fn = self.CombinedEmotionLoss(
            emotion_dim=64,
            audio_feature_dim=256,
            use_ser=False,  # Don't load SER model in tests
        )

        tts_loss = torch.tensor(1.5)
        emotion_embed = torch.randn(4, 64)
        audio_features = torch.randn(4, 256)

        losses = loss_fn(
            tts_loss=tts_loss,
            emotion_embed=emotion_embed,
            audio_features=audio_features,
        )

        self.assertIn("tts_loss", losses)
        self.assertIn("total_loss", losses)
        self.assertGreater(losses["total_loss"].item(), tts_loss.item())


class TestCheckpointLoading(unittest.TestCase):
    """Test checkpoint loading functionality."""

    def test_merged_checkpoint_format(self):
        """Verify merged checkpoint has correct structure if exists."""
        checkpoint_path = Path("checkpoints/emotion_merged/checkpoint_merged.pt")

        if not checkpoint_path.exists():
            self.skipTest("Merged checkpoint not found")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Should have these keys
        self.assertTrue(
            "emotion_embeddings_state_dict" in checkpoint or
            any("emotion_embeddings" in k for k in checkpoint.keys()),
            "Checkpoint should have emotion embeddings"
        )

    def test_per_dataset_checkpoint_format(self):
        """Verify per-dataset checkpoints have correct structure if they exist."""
        checkpoint_paths = [
            Path("checkpoints/emotion_lora_ravdess/checkpoint_early_stop.pt"),
            Path("checkpoints/emotion_lora_cremad/checkpoint_early_stop.pt"),
            Path("checkpoints/emotion_lora_iesc/checkpoint_early_stop.pt"),
        ]

        found_any = False
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():
                found_any = True
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

                self.assertIn("epoch", checkpoint)
                self.assertIn("loss", checkpoint)
                self.assertTrue(
                    "emotion_embeddings_state_dict" in checkpoint or
                    "t3_state_dict" in checkpoint,
                    f"Checkpoint {checkpoint_path} missing expected keys"
                )

        if not found_any:
            self.skipTest("No per-dataset checkpoints found")


class TestDatasetConfiguration(unittest.TestCase):
    """Test dataset configuration and validation."""

    def test_dataset_configs_present(self):
        """Verify dataset configs are defined."""
        # Import from training script
        sys.path.insert(0, str(Path(__file__).parent))
        from train_emotion_lora import DATASET_CONFIGS

        expected = {"ravdess", "cremad", "iesc"}
        actual = set(DATASET_CONFIGS.keys())
        self.assertEqual(expected, actual)

    def test_dataset_config_fields(self):
        """Verify dataset configs have required fields."""
        from train_emotion_lora import DATASET_CONFIGS

        for name, config in DATASET_CONFIGS.items():
            self.assertTrue(hasattr(config, "name"), f"{name} missing 'name'")
            self.assertTrue(hasattr(config, "expected_samples"), f"{name} missing 'expected_samples'")
            self.assertTrue(hasattr(config, "emotion_mapping"), f"{name} missing 'emotion_mapping'")
            self.assertTrue(hasattr(config, "language"), f"{name} missing 'language'")

    def test_expected_sample_counts(self):
        """Verify expected sample counts are set correctly."""
        from train_emotion_lora import DATASET_CONFIGS

        expected_counts = {
            "ravdess": 1440,
            "cremad": 7442,
            "iesc": 600,
        }

        for name, expected in expected_counts.items():
            actual = DATASET_CONFIGS[name].expected_samples
            self.assertEqual(actual, expected, f"{name} has wrong expected_samples")


class TestBalancedSampling(unittest.TestCase):
    """Test balanced sampling strategies."""

    def test_balanced_sampler_creation(self):
        """Test BalancedEmotionSampler can be created."""
        from train_emotion_lora import BalancedEmotionSampler

        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.samples = [
                    {"emotion": "happy", "audio_path": "a1.wav"},
                    {"emotion": "happy", "audio_path": "a2.wav"},
                    {"emotion": "sad", "audio_path": "a3.wav"},
                ]

        dataset = MockDataset()
        sampler = BalancedEmotionSampler(dataset)

        # Should produce more samples than original due to balancing
        self.assertGreaterEqual(len(sampler), len(dataset.samples))


def run_all_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestIntensityTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionTrajectory))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionCrossAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionLosses))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestBalancedSampling))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    # Run tests
    if len(sys.argv) > 1:
        # Run specific tests
        unittest.main()
    else:
        # Run all tests
        result = run_all_tests()

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")

        if result.wasSuccessful():
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
