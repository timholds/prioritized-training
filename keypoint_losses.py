import tensorflow as tf


def smooth_l1_loss(y_true, y_pred, delta=1.0):
    """
    Smooth L1 (Huber) loss - more robust to outliers than MSE.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        delta: Threshold at which to change from L2 to L1 loss
        
    Returns:
        Smooth L1 loss value
    """
    # Ensure consistent dtypes
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    error = y_true - y_pred
    abs_error = tf.abs(error)
    
    # Smooth L1: L2 when |error| < delta, L1 - delta/2 otherwise
    loss = tf.where(
        abs_error < delta,
        0.5 * tf.square(error),
        delta * abs_error - 0.5 * tf.square(delta)
    )
    
    return loss


def keypoint_loss_with_visibility(y_true, y_pred, reduction='mean', delta=1.0):
    """
    Visibility-aware keypoint loss function using Smooth L1.
    
    Expects y_true to have shape (batch_size, 51) where:
    - First 34 values are x,y coordinates for 17 keypoints
    - Last 17 values are visibility flags (0 or 1)
    
    If y_true has shape (batch_size, 34), assumes all keypoints are visible.
    
    Only computes loss on visible keypoints (visibility > 0).
    
    Args:
        y_true: Shape (batch_size, 51) or (batch_size, 34) - coordinates and optional visibility
        y_pred: Shape (batch_size, 34) - predicted coordinates only
        reduction: 'mean' for scalar loss, 'none' for per-sample losses
        
    Returns:
        Average loss per visible keypoint (scalar) if reduction='mean'
        Per-sample losses (batch_size,) if reduction='none'
    """
    # Handle both cases: with and without visibility information
    if y_true.shape[-1] == 51:
        # Extract coordinates and visibility from ground truth
        coords_true = y_true[:, :34]  # Shape: (batch_size, 34)
        visibility = y_true[:, 34:]    # Shape: (batch_size, 17)
    elif y_true.shape[-1] == 34:
        # No visibility information provided, assume all keypoints are visible
        coords_true = y_true  # Shape: (batch_size, 34)
        batch_size = tf.shape(y_true)[0]
        visibility = tf.ones([batch_size, 17], dtype=tf.float32)  # All visible
    else:
        raise ValueError(f"Expected y_true shape [..., 34] or [..., 51], got {y_true.shape}")
    
    # Ensure consistent dtypes
    coords_true = tf.cast(coords_true, tf.float32)
    coords_pred = tf.cast(y_pred, tf.float32)
    visibility = tf.cast(visibility, tf.float32)
    
    # Reshape for easier manipulation
    batch_size = tf.shape(coords_true)[0]
    coords_true = tf.reshape(coords_true, [batch_size, 17, 2])  # (batch, 17, 2)
    coords_pred = tf.reshape(y_pred, [batch_size, 17, 2])       # (batch, 17, 2)
    visibility = tf.reshape(visibility, [batch_size, 17, 1])     # (batch, 17, 1)
    
    # Compute Smooth L1 loss for each coordinate
    coord_loss = smooth_l1_loss(coords_true, coords_pred, delta=delta)  # (batch, 17, 2)
    
    # Average loss per keypoint (over x,y)
    keypoint_loss = tf.reduce_mean(coord_loss, axis=2)  # (batch, 17)
    
    # Apply visibility mask - set loss to 0 for invisible keypoints
    visibility_mask = tf.squeeze(visibility, axis=2)  # (batch, 17)
    masked_loss = keypoint_loss * visibility_mask
    
    # Count visible keypoints per sample
    visible_keypoints_per_sample = tf.reduce_sum(visibility_mask, axis=1)  # (batch,)
    
    # Debug: Print visibility statistics occasionally (every 100th call)
    # This helps identify if visibility masking is working correctly
    # Note: Commented out to avoid potential TensorArray warnings from tf.cond
    # debug_condition = tf.equal(tf.cast(tf.timestamp() * 1000, tf.int64) % 100, 0)
    # tf.cond(
    #     debug_condition,
    #     lambda: tf.print("Visibility stats - min:", tf.reduce_min(visible_keypoints_per_sample), 
    #                     "max:", tf.reduce_max(visible_keypoints_per_sample),
    #                     "mean:", tf.reduce_mean(visible_keypoints_per_sample)),
    #     lambda: tf.constant(0)
    # )
    
    # Avoid division by zero
    visible_keypoints_per_sample = tf.maximum(visible_keypoints_per_sample, 1.0)
    
    # Average loss per visible keypoint for each sample
    loss_per_sample = tf.reduce_sum(masked_loss, axis=1) / visible_keypoints_per_sample  # (batch,)
    
    # Return based on reduction mode
    if reduction == 'none':
        return loss_per_sample
    else:
        final_loss = tf.reduce_mean(loss_per_sample)
        # Scale up the loss for better gradient flow with small coordinate values
        # This doesn't change the optimization but makes the loss values more readable
        return final_loss * 10.0


def keypoint_mse_with_visibility(y_true, y_pred, reduction='mean'):
    """
    Visibility-aware keypoint loss function using MSE.
    Same as above but with MSE instead of Smooth L1, for comparison.
    """
    # Extract coordinates and visibility from ground truth
    coords_true = y_true[:, :34]
    visibility = y_true[:, 34:]
    
    # Reshape for easier manipulation
    batch_size = tf.shape(coords_true)[0]
    coords_true = tf.reshape(coords_true, [batch_size, 17, 2])
    coords_pred = tf.reshape(y_pred, [batch_size, 17, 2])
    visibility = tf.reshape(visibility, [batch_size, 17, 1])
    
    # Compute MSE for each coordinate
    coord_loss = tf.square(coords_true - coords_pred)  # (batch, 17, 2)
    
    # Average loss per keypoint (over x,y)
    keypoint_loss = tf.reduce_mean(coord_loss, axis=2)  # (batch, 17)
    
    # Apply visibility mask
    visibility_mask = tf.squeeze(visibility, axis=2)
    masked_loss = keypoint_loss * visibility_mask
    
    # Count visible keypoints per sample
    visible_keypoints_per_sample = tf.reduce_sum(visibility_mask, axis=1)
    visible_keypoints_per_sample = tf.maximum(visible_keypoints_per_sample, 1.0)
    
    # Average loss per visible keypoint for each sample
    loss_per_sample = tf.reduce_sum(masked_loss, axis=1) / visible_keypoints_per_sample
    
    # Return based on reduction mode
    if reduction == 'none':
        return loss_per_sample
    else:
        return tf.reduce_mean(loss_per_sample)


# COCO keypoint sigmas for OKS calculation (used for per-keypoint weighting if needed)
COCO_KEYPOINT_SIGMAS = [
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
]


def create_keypoint_loss(loss_type='smooth_l1_with_visibility', **kwargs):
    """
    Factory function to create keypoint loss functions.
    
    Args:
        loss_type: One of ['smooth_l1_with_visibility', 'mse_with_visibility', 'smooth_l1', 'mse']
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function
    """
    if loss_type == 'smooth_l1_with_visibility':
        delta = kwargs.get('delta', 1.0)
        def loss_fn(y_true, y_pred, reduction='mean'):
            return keypoint_loss_with_visibility(y_true, y_pred, reduction=reduction, delta=delta)
        return loss_fn
    elif loss_type == 'mse_with_visibility':
        def loss_fn(y_true, y_pred, reduction='mean'):
            return keypoint_mse_with_visibility(y_true, y_pred, reduction=reduction)
        return loss_fn
    elif loss_type == 'smooth_l1':
        # Standard smooth L1 without visibility handling (not recommended)
        def loss_fn(y_true, y_pred, reduction='mean'):
            loss = smooth_l1_loss(y_true, y_pred)
            if reduction == 'none':
                # Return per-sample loss by averaging over spatial dimensions
                return tf.reduce_mean(loss, axis=list(range(1, len(loss.shape))))
            else:
                return tf.reduce_mean(loss)
        return loss_fn
    elif loss_type == 'mse' or loss_type == 'mean_squared_error':
        # Standard MSE (current behavior)
        return tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")