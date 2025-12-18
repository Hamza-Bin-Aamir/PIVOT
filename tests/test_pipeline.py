"""Tests for evaluation pipeline."""

import numpy as np
import pytest

from src.eval.pipeline import (
    DetectionPrediction,
    EvaluationPipeline,
    EvaluationResults,
    GroundTruth,
)


class TestDetectionPrediction:
    """Test DetectionPrediction dataclass."""

    def test_minimal_prediction(self) -> None:
        """Test prediction with only required fields."""
        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
        )
        assert pred.centers.shape == (1, 3)
        assert pred.confidences.shape == (1,)
        assert pred.sizes is None
        assert pred.segmentation_mask is None
        assert pred.triage_score is None

    def test_full_prediction(self) -> None:
        """Test prediction with all fields."""
        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
            sizes=np.array([5.0]),
            segmentation_mask=np.ones((10, 10, 10), dtype=bool),
            triage_score=0.8,
        )
        assert pred.sizes is not None
        assert pred.segmentation_mask is not None
        assert pred.triage_score == 0.8


class TestGroundTruth:
    """Test GroundTruth dataclass."""

    def test_minimal_ground_truth(self) -> None:
        """Test ground truth with only required fields."""
        gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]))
        assert gt.centers.shape == (1, 3)
        assert gt.sizes is None
        assert gt.segmentation_mask is None
        assert gt.triage_score is None
        assert gt.is_urgent is None

    def test_full_ground_truth(self) -> None:
        """Test ground truth with all fields."""
        gt = GroundTruth(
            centers=np.array([[10.0, 20.0, 30.0]]),
            sizes=np.array([5.0]),
            segmentation_mask=np.ones((10, 10, 10), dtype=bool),
            triage_score=0.8,
            is_urgent=True,
        )
        assert gt.sizes is not None
        assert gt.segmentation_mask is not None
        assert gt.triage_score == 0.8
        assert gt.is_urgent is True


class TestEvaluationPipeline:
    """Test EvaluationPipeline class."""

    def test_initialization(self) -> None:
        """Test pipeline initialization."""
        pipeline = EvaluationPipeline()
        assert pipeline.spacing is not None
        assert pipeline.detection_distance_threshold == 10.0
        assert len(pipeline.predictions) == 0
        assert len(pipeline.ground_truths) == 0

    def test_custom_initialization(self) -> None:
        """Test pipeline with custom parameters."""
        spacing = np.array([0.7, 0.7, 2.5])
        pipeline = EvaluationPipeline(
            spacing=spacing,
            detection_distance_threshold=5.0,
            froc_thresholds=[0.1, 0.5, 0.9],
        )
        np.testing.assert_array_equal(pipeline.spacing, spacing)
        assert pipeline.detection_distance_threshold == 5.0
        assert pipeline.froc_thresholds == [0.1, 0.5, 0.9]

    def test_add_case(self) -> None:
        """Test adding cases to pipeline."""
        pipeline = EvaluationPipeline()

        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
        )
        gt = GroundTruth(centers=np.array([[10.5, 20.5, 30.5]]))

        pipeline.add_case(pred, gt)
        assert len(pipeline.predictions) == 1
        assert len(pipeline.ground_truths) == 1

    def test_evaluate_empty_pipeline(self) -> None:
        """Test error when evaluating empty pipeline."""
        pipeline = EvaluationPipeline()
        with pytest.raises(ValueError, match="No cases added"):
            pipeline.evaluate()

    def test_evaluate_perfect_detection(self) -> None:
        """Test evaluation with perfect detections."""
        pipeline = EvaluationPipeline(detection_distance_threshold=1.0)

        # Add perfect predictions
        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0], [50.0, 60.0, 70.0]]),
            confidences=np.array([0.9, 0.8]),
        )
        gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0], [50.0, 60.0, 70.0]]))

        pipeline.add_case(pred, gt)
        results = pipeline.evaluate()

        assert isinstance(results, EvaluationResults)
        assert results.num_images == 1
        assert results.num_predictions == 2
        assert results.num_ground_truths == 2

        # Check detection metrics
        assert results.detection_metrics["precision"] == 1.0
        assert results.detection_metrics["recall"] == 1.0
        assert results.detection_metrics["num_matched"] == 2

        # Check FROC metrics
        assert results.froc_metrics["average_sensitivity"] > 0.9

    def test_evaluate_with_false_positives(self) -> None:
        """Test evaluation with false positive detections."""
        pipeline = EvaluationPipeline(detection_distance_threshold=1.0)

        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0], [100.0, 100.0, 100.0]]),  # Second is FP
            confidences=np.array([0.9, 0.8]),
        )
        gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]))

        pipeline.add_case(pred, gt)
        results = pipeline.evaluate()

        # Precision should be < 1.0 (1 TP, 1 FP)
        assert results.detection_metrics["precision"] < 1.0
        assert results.detection_metrics["recall"] == 1.0  # All GTs found
        assert results.detection_metrics["num_matched"] == 1

    def test_evaluate_with_missed_detections(self) -> None:
        """Test evaluation with missed ground truths."""
        pipeline = EvaluationPipeline(detection_distance_threshold=1.0)

        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
        )
        gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0], [100.0, 100.0, 100.0]]))  # Second is missed

        pipeline.add_case(pred, gt)
        results = pipeline.evaluate()

        # Recall should be < 1.0 (1 TP, 1 FN)
        assert results.detection_metrics["precision"] == 1.0  # No FPs
        assert results.detection_metrics["recall"] < 1.0
        assert results.detection_metrics["num_matched"] == 1

    def test_evaluate_segmentation(self) -> None:
        """Test segmentation evaluation."""
        pipeline = EvaluationPipeline()

        # Create masks
        pred_mask = np.zeros((10, 10, 10), dtype=bool)
        pred_mask[0:5, 0:5, 0:5] = True

        gt_mask = np.zeros((10, 10, 10), dtype=bool)
        gt_mask[0:5, 0:5, 0:5] = True  # Perfect overlap

        pred = DetectionPrediction(
            centers=np.array([[2.5, 2.5, 2.5]]),
            confidences=np.array([0.9]),
            segmentation_mask=pred_mask,
        )
        gt = GroundTruth(centers=np.array([[2.5, 2.5, 2.5]]), segmentation_mask=gt_mask)

        pipeline.add_case(pred, gt)
        results = pipeline.evaluate()

        # Should have perfect Dice and IoU
        assert results.segmentation_metrics["mean_dice"] == 1.0
        assert results.segmentation_metrics["mean_iou"] == 1.0
        assert results.segmentation_metrics["num_masks"] == 1

    def test_evaluate_segmentation_no_masks(self) -> None:
        """Test segmentation evaluation without masks."""
        pipeline = EvaluationPipeline()

        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
        )
        gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]))

        pipeline.add_case(pred, gt)
        results = pipeline.evaluate()

        assert np.isnan(results.segmentation_metrics["mean_dice"])
        assert results.segmentation_metrics["num_masks"] == 0

    def test_evaluate_size(self) -> None:
        """Test size estimation evaluation."""
        pipeline = EvaluationPipeline()

        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
            sizes=np.array([5.0]),
        )
        gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]), sizes=np.array([5.0]))

        pipeline.add_case(pred, gt)
        results = pipeline.evaluate()

        # Perfect size prediction
        assert results.size_metrics["detection_size_metrics"]["mae"] == 0.0
        assert results.size_metrics["num_detections"] == 1

    def test_evaluate_triage_correlation(self) -> None:
        """Test triage score correlation evaluation."""
        pipeline = EvaluationPipeline()

        # Add cases with increasing triage scores
        for i in range(5):
            score = i / 4.0  # 0.0, 0.25, 0.5, 0.75, 1.0
            pred = DetectionPrediction(
                centers=np.array([[10.0, 20.0, 30.0]]),
                confidences=np.array([0.9]),
                triage_score=score,
            )
            gt = GroundTruth(
                centers=np.array([[10.0, 20.0, 30.0]]),
                triage_score=score,
                is_urgent=score > 0.5,
            )
            pipeline.add_case(pred, gt)

        results = pipeline.evaluate()

        # Should have perfect correlation (with floating point tolerance)
        assert abs(results.triage_metrics["correlation_metrics"]["pearson_correlation"] - 1.0) < 0.001
        assert results.triage_metrics["num_scores"] == 5
        assert results.triage_metrics["num_binary_labels"] == 5

    def test_evaluate_triage_classification(self) -> None:
        """Test triage binary classification evaluation."""
        pipeline = EvaluationPipeline()

        # Perfect classification
        pred1 = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
            triage_score=0.9,
        )
        gt1 = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]), is_urgent=True)

        pred2 = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
            triage_score=0.1,
        )
        gt2 = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]), is_urgent=False)

        pipeline.add_case(pred1, gt1)
        pipeline.add_case(pred2, gt2)
        results = pipeline.evaluate()

        # Should have perfect AUC
        assert results.triage_metrics["classification_metrics"]["auc_roc"] == 1.0

    def test_reset_pipeline(self) -> None:
        """Test pipeline reset functionality."""
        pipeline = EvaluationPipeline()

        pred = DetectionPrediction(
            centers=np.array([[10.0, 20.0, 30.0]]),
            confidences=np.array([0.9]),
        )
        gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]))

        pipeline.add_case(pred, gt)
        assert len(pipeline.predictions) == 1

        pipeline.reset()
        assert len(pipeline.predictions) == 0
        assert len(pipeline.ground_truths) == 0

    def test_multiple_images(self) -> None:
        """Test evaluation across multiple images."""
        pipeline = EvaluationPipeline(detection_distance_threshold=1.0)

        # Add 3 images
        for i in range(3):
            pred = DetectionPrediction(
                centers=np.array([[10.0 + i, 20.0, 30.0]]),
                confidences=np.array([0.9]),
            )
            gt = GroundTruth(centers=np.array([[10.0 + i, 20.0, 30.0]]))
            pipeline.add_case(pred, gt)

        results = pipeline.evaluate()

        assert results.num_images == 3
        assert results.num_predictions == 3
        assert results.num_ground_truths == 3
        assert results.detection_metrics["num_matched"] == 3


class TestIntegrationPipeline:
    """Integration tests for evaluation pipeline."""

    def test_comprehensive_evaluation(self) -> None:
        """Test full pipeline with all metrics."""
        pipeline = EvaluationPipeline(
            spacing=np.array([1.0, 1.0, 1.0]),
            detection_distance_threshold=2.0,
        )

        # Create comprehensive prediction and ground truth
        pred_mask = np.zeros((20, 20, 20), dtype=bool)
        pred_mask[5:15, 5:15, 5:15] = True

        gt_mask = np.zeros((20, 20, 20), dtype=bool)
        gt_mask[5:15, 5:15, 5:15] = True

        pred = DetectionPrediction(
            centers=np.array([[10.0, 10.0, 10.0], [50.0, 50.0, 50.0]]),
            confidences=np.array([0.95, 0.85]),
            sizes=np.array([1000.0, 800.0]),
            segmentation_mask=pred_mask,
            triage_score=0.8,
        )

        gt = GroundTruth(
            centers=np.array([[10.1, 10.1, 10.1], [50.2, 50.2, 50.2]]),
            sizes=np.array([1000.0, 800.0]),
            segmentation_mask=gt_mask,
            triage_score=0.8,
            is_urgent=True,
        )

        pipeline.add_case(pred, gt)
        results = pipeline.evaluate()

        # All metrics should be computed
        assert "average_sensitivity" in results.froc_metrics
        assert "precision" in results.detection_metrics
        assert "mean_dice" in results.segmentation_metrics
        assert "detection_size_metrics" in results.size_metrics
        assert "correlation_metrics" in results.triage_metrics

        # Should have good performance
        assert results.detection_metrics["precision"] > 0.9
        assert results.segmentation_metrics["mean_dice"] == 1.0

    def test_batch_processing(self) -> None:
        """Test processing a batch of images."""
        pipeline = EvaluationPipeline(detection_distance_threshold=1.0)

        # Simulate a batch of 10 images with varying quality
        for i in range(10):
            # Vary the quality
            offset = 0.1 * i if i < 5 else 0.5  # First 5 good, rest bad

            pred = DetectionPrediction(
                centers=np.array([[10.0 + offset, 20.0, 30.0]]),
                confidences=np.array([0.9 - i * 0.05]),
            )
            gt = GroundTruth(centers=np.array([[10.0, 20.0, 30.0]]))
            pipeline.add_case(pred, gt)

        results = pipeline.evaluate()

        assert results.num_images == 10
        # With threshold of 1.0, offsets 0.1-0.4 match, but 0.5 doesn't
        # First 5 have offsets 0.0, 0.1, 0.2, 0.3, 0.4 (all < 1.0)
        # Last 5 all have offset 0.5 (just at threshold)
        # Actual distance = sqrt(offset^2) so 0.5 matches
        assert results.detection_metrics["num_matched"] == 10
