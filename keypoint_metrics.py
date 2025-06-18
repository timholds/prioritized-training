import tensorflow as tf
import numpy as np


class OKSMetric(tf.keras.metrics.Metric):
    """
    Object Keypoint Similarity (OKS) metric for COCO keypoint evaluation.
    
    OKS measures the similarity between predicted and ground truth keypoints,
    taking into account the scale of the person and per-keypoint difficulty.
    
    Formula: OKS = Σᵢ[exp(-dᵢ²/2s²κᵢ²)·δ(vᵢ>0)] / Σᵢ[δ(vᵢ>0)]
    where:
        - dᵢ: Euclidean distance between predicted and ground truth for keypoint i
        - s: Object scale (square root of object area)
        - κᵢ: Per-keypoint constant that controls falloff
        - vᵢ: Visibility flag for keypoint i
    """
    
    def __init__(self, name='oks', use_area=False, **kwargs):
        """
        Args:
            name: Metric name
            use_area: If True, use actual object area for scale. If False, use adaptive scale.
                     Note: For direct regression without bbox, we estimate scale from keypoint spread.
        """
        super().__init__(name=name, **kwargs)
        
        # COCO keypoint sigmas (κ values)
        self.sigmas = tf.constant([
            0.026,  # nose
            0.025,  # left_eye  
            0.025,  # right_eye
            0.035,  # left_ear
            0.035,  # right_ear
            0.079,  # left_shoulder
            0.079,  # right_shoulder
            0.072,  # left_elbow
            0.072,  # right_elbow
            0.062,  # left_wrist
            0.062,  # right_wrist
            0.107,  # left_hip
            0.107,  # right_hip
            0.087,  # left_knee
            0.087,  # right_knee
            0.089,  # left_ankle
            0.089,  # right_ankle
        ], dtype=tf.float32)
        
        self.use_area = use_area
        self.total_oks = self.add_weight(name='total_oks', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def _estimate_scale_from_keypoints(self, coords_true, visibility):
        """
        Estimate person scale from the spread of visible keypoints.
        
        Args:
            coords_true: Ground truth coordinates (batch_size, 17, 2)
            visibility: Visibility flags (batch_size, 17)
            
        Returns:
            scale: Estimated scale for each person (batch_size,)
        """
        batch_size = tf.shape(coords_true)[0]
        
        # Mask out invisible keypoints by setting them to a large value for min, small for max
        vis_expanded = tf.expand_dims(visibility, axis=2)  # (batch, 17, 1)
        
        # For invisible keypoints, set coordinates to extreme values so they don't affect min/max
        large_val = tf.constant(999.0, dtype=tf.float32)
        small_val = tf.constant(-999.0, dtype=tf.float32)
        
        coords_for_min = tf.where(vis_expanded > 0, coords_true, large_val)
        coords_for_max = tf.where(vis_expanded > 0, coords_true, small_val)
        
        # Compute bounding box for each person
        min_coords = tf.reduce_min(coords_for_min, axis=1)  # (batch, 2)
        max_coords = tf.reduce_max(coords_for_max, axis=1)  # (batch, 2)
        
        # Handle case where no keypoints are visible
        num_visible = tf.reduce_sum(visibility, axis=1)  # (batch,)
        has_visible = num_visible >= 2
        
        # Compute bounding box dimensions
        bbox_width = max_coords[:, 0] - min_coords[:, 0]   # (batch,)
        bbox_height = max_coords[:, 1] - min_coords[:, 1]  # (batch,)
        
        # Use max dimension as scale proxy
        estimated_scale = tf.maximum(bbox_width, bbox_height)
        
        # Apply minimum scale and handle cases with insufficient visible keypoints
        min_scale = 0.05
        default_scale = 0.15  # For people with <2 visible keypoints
        
        scale = tf.where(
            has_visible,
            tf.maximum(estimated_scale, min_scale),
            default_scale
        )
        
        return scale
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state with batch of predictions.
        
        Args:
            y_true: Ground truth with shape (batch_size, 51) - coords + visibility
            y_pred: Predictions with shape (batch_size, 34) - coords only
            sample_weight: Not used
        """
        # Extract coordinates and visibility
        coords_true = y_true[:, :34]
        visibility = y_true[:, 34:]
        
        # Ensure consistent dtypes
        coords_true = tf.cast(coords_true, tf.float32)
        coords_pred = tf.cast(y_pred, tf.float32)
        visibility = tf.cast(visibility, tf.float32)
        
        # Reshape for easier manipulation
        batch_size = tf.shape(coords_true)[0]
        coords_true = tf.reshape(coords_true, [batch_size, 17, 2])
        coords_pred = tf.reshape(coords_pred, [batch_size, 17, 2])
        visibility = tf.reshape(visibility, [batch_size, 17])
        
        # Compute Euclidean distances
        distances = tf.sqrt(tf.reduce_sum(tf.square(coords_true - coords_pred), axis=2))  # (batch, 17)
        
        # Compute scale based on keypoint spread (adaptive scale estimation)
        if self.use_area:
            # Would need bbox info to compute actual area - use adaptive scale instead
            scale = self._estimate_scale_from_keypoints(coords_true, visibility)
        else:
            # Estimate scale from the spread of visible keypoints
            # This is more accurate than using a fixed scale for all people
            scale = self._estimate_scale_from_keypoints(coords_true, visibility)
        
        # Expand dimensions for broadcasting
        scale = tf.expand_dims(scale, axis=1)  # (batch, 1)
        sigmas_expanded = tf.expand_dims(self.sigmas, axis=0)  # (1, 17)
        
        # Compute OKS for each keypoint
        # OKS_i = exp(-d_i^2 / (2 * s^2 * kappa_i^2))
        oks_per_keypoint = tf.exp(-tf.square(distances) / (2 * tf.square(scale) * tf.square(sigmas_expanded)))
        
        # Apply visibility mask
        oks_per_keypoint = oks_per_keypoint * visibility
        
        # Compute OKS per sample
        num_visible = tf.reduce_sum(visibility, axis=1)  # (batch,)
        num_visible = tf.maximum(num_visible, 1.0)  # Avoid division by zero
        
        oks_per_sample = tf.reduce_sum(oks_per_keypoint, axis=1) / num_visible  # (batch,)
        
        # Update running average
        batch_oks = tf.reduce_mean(oks_per_sample)
        self.total_oks.assign_add(batch_oks)
        self.count.assign_add(1.0)
    
    def result(self):
        """Return the average OKS across all batches."""
        return tf.divide(self.total_oks, self.count)
    
    def reset_state(self):
        """Reset metric state."""
        self.total_oks.assign(0.0)
        self.count.assign(0.0)


class PCKMetric(tf.keras.metrics.Metric):
    """
    Percentage of Correct Keypoints (PCK) metric.
    
    A keypoint is considered correct if the distance between prediction and
    ground truth is less than a threshold (typically 0.05 or 0.1 times the
    person scale).
    """
    
    def __init__(self, threshold=0.05, name='pck', **kwargs):
        """
        Args:
            threshold: Distance threshold as fraction of person scale
            name: Metric name
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct_keypoints = self.add_weight(name='correct', initializer='zeros')
        self.total_keypoints = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state with batch of predictions."""
        # Extract coordinates and visibility
        coords_true = y_true[:, :34]
        visibility = y_true[:, 34:]
        
        # Ensure consistent dtypes
        coords_true = tf.cast(coords_true, tf.float32)
        coords_pred = tf.cast(y_pred, tf.float32)
        visibility = tf.cast(visibility, tf.float32)
        
        # Reshape for easier manipulation  
        batch_size = tf.shape(coords_true)[0]
        coords_true = tf.reshape(coords_true, [batch_size, 17, 2])
        coords_pred = tf.reshape(coords_pred, [batch_size, 17, 2])
        visibility = tf.reshape(visibility, [batch_size, 17])
        
        # Compute Euclidean distances
        distances = tf.sqrt(tf.reduce_sum(tf.square(coords_true - coords_pred), axis=2))
        
        # For normalized coordinates, use fixed scale
        scale_threshold = self.threshold * 0.25  # 0.25 is approximate person scale
        
        # Count correct keypoints (distance < threshold and visible)
        correct = tf.cast(distances < scale_threshold, tf.float32) * visibility
        
        # Update counts
        self.correct_keypoints.assign_add(tf.reduce_sum(correct))
        self.total_keypoints.assign_add(tf.reduce_sum(visibility))
    
    def result(self):
        """Return PCK percentage."""
        return tf.divide(self.correct_keypoints, tf.maximum(self.total_keypoints, 1.0))
    
    def reset_state(self):
        """Reset metric state."""
        self.correct_keypoints.assign(0.0)
        self.total_keypoints.assign(0.0)


def create_keypoint_metric(metric_type='oks', **kwargs):
    """
    Factory function to create keypoint metrics.
    
    Args:
        metric_type: One of ['oks', 'pck', 'pck@0.05', 'pck@0.1']
        **kwargs: Additional arguments for the metric
        
    Returns:
        Metric instance
    """
    if metric_type == 'oks':
        return OKSMetric(**kwargs)
    elif metric_type == 'pck' or metric_type == 'pck@0.05':
        return PCKMetric(threshold=0.05, **kwargs)
    elif metric_type == 'pck@0.1':
        return PCKMetric(threshold=0.1, **kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")