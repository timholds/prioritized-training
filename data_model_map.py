from models import MLPModel, ConvModel, ResNet18, KeypointResNet18
from tensorflow import keras
from keypoint_metrics import OKSMetric, PCKMetric

data_model_map = {}
data_model_map['qmnist'] = {
    'model': ConvModel,  # Store class, not instance
    'loss': 'categorical_crossentropy',
    'target_accuracy': 0.99,
    'epochs': 2,
    'holdout_epochs': 1,
    'input_shape': (28, 28, 1),
    'metrics': ['accuracy'],
    'n_classes': 10,
    'augmentation': None,
    'batch_size': 32,
    'learning_rate': 0.001,
}

data_model_map['cifar10'] = {
    'model': ResNet18,  # Store class, not instance
    'loss': 'categorical_crossentropy',
    'target_accuracy': 0.85,
    'epochs': 10,
    'holdout_epochs': 5,
    'input_shape': (32, 32, 3),
    'metrics': ['accuracy'],
    'n_classes': 10,
    'augmentation': [keras.layers.RandomCrop, keras.layers.RandomFlip(mode='horizontal')],
    'batch_size': 32,
    'learning_rate': 0.001,
}

data_model_map['cifar100'] = {
    'model': ResNet18,  # Store class, not instance
    'loss': 'categorical_crossentropy',
    'target_accuracy': 0.45,
    'epochs': 10,
    'holdout_epochs': 5,
    'input_shape': (32, 32, 3),
    'metrics': ['accuracy'],
    'n_classes': 100,
    'augmentation': [keras.layers.RandomCrop, keras.layers.RandomFlip(mode='horizontal')],
    'batch_size': 32,
    'learning_rate': 0.001,
}

data_model_map['cinic10'] = {
    'model': ResNet18,  # Store class, not instance
    'loss': 'categorical_crossentropy',
    'target_accuracy': 0.74,
    'epochs': 10,
    'holdout_epochs': 5,
    'input_shape': (32, 32, 3),
    'metrics': ['accuracy'],
    'n_classes': 10,
    'augmentation': [keras.layers.RandomCrop, keras.layers.RandomFlip(mode='horizontal')],
    'batch_size': 64,
    'learning_rate': 0.001,
}

data_model_map['cocoreg'] = {
    'model': ResNet18,  # Store class, not instance
    'holdout_model': ResNet18,  # Use same model for holdout to ensure consistency
    'loss': 'mean_squared_error',
    'target_metric_value': 0.07,  # Target low MSE for regression
    'epochs': 10,
    'holdout_epochs': 2,  # Reduced from 5 for faster iteration
    'input_shape': (112, 112, 3),
    'metrics': [keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()],  # Regression metrics
    'n_outputs': 4,  # 4 bbox coordinates [x, y, width, height]
    'output_activation': 'linear',  # Linear activation for regression
    'augmentation': [
        keras.layers.RandomFlip(mode='horizontal'),  # 50% probability (default)
        keras.layers.RandomCrop(height=224, width=224),  # Random crop maintaining aspect ratio
    ],
    'augmentation_holdout': None,  # No augmentation for holdout loss calculation
    'batch_size': 16,  # Reduced from 32 for memory efficiency
    'learning_rate': 0.001,
}

data_model_map['cocokp'] = {
    'model': KeypointResNet18,  # Store class, not instance
    'loss': 'smooth_l1_with_visibility',  # Use custom Smooth L1 loss with visibility masking
    'loss_kwargs': {  # Additional arguments for the loss function
        'delta': 0.1,  # Reduced delta for normalized coordinates [0,1]
        'visibility_threshold': 0,  # Include all labeled keypoints (v > 0)
        'use_oks_weighting': False  # Can enable for OKS-based per-keypoint weights
    },
    'target_metric_value': 0.20,  # Target OKS of 0.20 (20% similarity) - realistic for ResNet18 direct regression
    'epochs': 20,
    'holdout_epochs': 30,
    'input_shape': (112, 112, 3),
    'metrics': [OKSMetric(), PCKMetric(threshold=0.05)],  # OKS is primary metric, MSE doesn't work with visibility labels
    'n_outputs': 34,  # 17 keypoints * 2 coordinates (no visibility prediction yet)
    'include_visibility': True,  # Load visibility flags with keypoint data
    # IMPORTANT: Augmentation is still disabled for keypoint regression
    # To enable augmentation, use the KeypointAwareRandomFlip layer from visualize_keypoint_augmentation.py
    # This properly handles coordinate transformation and left/right keypoint swapping
    'augmentation': [],
    'batch_size': 32,  
    'learning_rate': 0.001,  # Reduced learning rate for regression with normalized coordinates
    'optimizer': 'sgd',  # Standard ResNet optimizer
    'momentum': 0.9,  # Standard ResNet momentum
    'weight_decay': 5e-4,  # Standard ResNet weight decay
    'lr_schedule_type': 'step',  # Use step decay schedule
    'lr_decay_epochs': [10, 20],  # Decay at epochs 10 and 20 (adjusted for 30 total epochs)
    'lr_decay_factor': 0.1,  # Divide by 10 at each decay
}

data_model_map['mpi'] = {}
data_model_map['affectnet'] = {}
data_model_map['raf'] = {}

