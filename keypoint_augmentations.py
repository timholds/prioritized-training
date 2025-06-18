import tensorflow as tf
from tensorflow import keras
import numpy as np


class KeypointAwareRandomFlip(keras.layers.Layer):
    """
    Random horizontal flip augmentation that properly handles keypoint transformations.
    
    This layer:
    1. Flips the image horizontally with 50% probability
    2. Transforms keypoint x-coordinates: x' = 1.0 - x
    3. Swaps left/right keypoint pairs (e.g., left_eye <-> right_eye)
    4. Preserves visibility flags
    """
    
    # COCO keypoint indices (0-based)
    COCO_KEYPOINT_NAMES = [
        'nose',           # 0
        'left_eye',       # 1
        'right_eye',      # 2
        'left_ear',       # 3
        'right_ear',      # 4
        'left_shoulder',  # 5
        'right_shoulder', # 6
        'left_elbow',     # 7
        'right_elbow',    # 8
        'left_wrist',     # 9
        'right_wrist',    # 10
        'left_hip',       # 11
        'right_hip',      # 12
        'left_knee',      # 13
        'right_knee',     # 14
        'left_ankle',     # 15
        'right_ankle'     # 16
    ]
    
    # Swap pairs for horizontal flip (0-based indices)
    SWAP_PAIRS = [
        (1, 2),   # left_eye <-> right_eye
        (3, 4),   # left_ear <-> right_ear
        (5, 6),   # left_shoulder <-> right_shoulder
        (7, 8),   # left_elbow <-> right_elbow
        (9, 10),  # left_wrist <-> right_wrist
        (11, 12), # left_hip <-> right_hip
        (13, 14), # left_knee <-> right_knee
        (15, 16)  # left_ankle <-> right_ankle
    ]
    
    def __init__(self, include_visibility=True, **kwargs):
        """
        Args:
            include_visibility: If True, expects labels with shape (..., 51).
                              If False, expects labels with shape (..., 34).
        """
        super().__init__(**kwargs)
        self.include_visibility = include_visibility
        
    def call(self, inputs, training=None):
        """
        Apply augmentation.
        
        Args:
            inputs: Tuple of (images, labels)
                   images: Tensor of shape (batch, H, W, C)
                   labels: Tensor of shape (batch, 34) or (batch, 51)
                   
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        if not training:
            return inputs
            
        images, labels = inputs
        batch_size = tf.shape(images)[0]
        
        # Generate random flip decisions for each sample
        flip_mask = tf.random.uniform([batch_size], 0, 1) < 0.5
        flip_mask = tf.cast(flip_mask, tf.float32)
        
        # Flip images
        flipped_images = tf.image.flip_left_right(images)
        
        # Select between original and flipped based on mask
        flip_mask_img = tf.reshape(flip_mask, [batch_size, 1, 1, 1])
        augmented_images = images * (1 - flip_mask_img) + flipped_images * flip_mask_img
        
        # Process keypoints
        if self.include_visibility:
            coords = labels[:, :34]
            visibility = labels[:, 34:]
        else:
            coords = labels
            visibility = None
            
        # Reshape coordinates for easier manipulation
        coords = tf.reshape(coords, [batch_size, 17, 2])
        
        # Transform x-coordinates for flipped samples
        x_coords = coords[:, :, 0]
        y_coords = coords[:, :, 1]
        
        # x' = 1.0 - x for flipped samples
        flipped_x = 1.0 - x_coords
        flip_mask_coord = tf.reshape(flip_mask, [batch_size, 1])
        x_coords = x_coords * (1 - flip_mask_coord) + flipped_x * flip_mask_coord
        
        # Reconstruct coordinates
        coords = tf.stack([x_coords, y_coords], axis=2)
        
        # Swap left/right keypoint pairs for flipped samples
        swapped_coords = tf.identity(coords)
        for left_idx, right_idx in self.SWAP_PAIRS:
            left_kp = coords[:, left_idx, :]
            right_kp = coords[:, right_idx, :]
            
            # Swap based on flip mask
            flip_mask_swap = tf.reshape(flip_mask, [batch_size, 1])
            swapped_coords = tf.concat([
                swapped_coords[:, :left_idx, :],
                tf.expand_dims(
                    left_kp * (1 - flip_mask_swap) + right_kp * flip_mask_swap,
                    axis=1
                ),
                swapped_coords[:, left_idx+1:right_idx, :],
                tf.expand_dims(
                    right_kp * (1 - flip_mask_swap) + left_kp * flip_mask_swap,
                    axis=1
                ),
                swapped_coords[:, right_idx+1:, :]
            ], axis=1)
        
        # Flatten coordinates back
        augmented_coords = tf.reshape(swapped_coords, [batch_size, 34])
        
        # Reconstruct labels
        if self.include_visibility:
            augmented_labels = tf.concat([augmented_coords, visibility], axis=1)
        else:
            augmented_labels = augmented_coords
            
        return augmented_images, augmented_labels
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'include_visibility': self.include_visibility
        })
        return config


def create_keypoint_augmentation_pipeline(include_visibility=True):
    """
    Create a complete augmentation pipeline for keypoint detection.
    
    Currently only includes horizontal flip, but can be extended with:
    - Random crop (with keypoint coordinate adjustment)
    - Random rotation (with keypoint rotation)
    - Color jitter (no keypoint adjustment needed)
    
    Args:
        include_visibility: Whether labels include visibility flags
        
    Returns:
        List of augmentation layers
    """
    return [
        KeypointAwareRandomFlip(include_visibility=include_visibility)
    ]