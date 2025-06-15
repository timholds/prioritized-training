from models import MLPModel, ConvModel, ResNet18, KeypointResNet18
from tensorflow import keras

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
    'holdout_epochs': 5,
    'input_shape': (224, 224, 3),
    'metrics': [keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()],  # Regression metrics
    'n_outputs': 4,  # 4 bbox coordinates [x, y, width, height]
    'output_activation': 'linear',  # Linear activation for regression
    'augmentation': [
        keras.layers.RandomFlip(mode='horizontal'),  # 50% probability (default)
        keras.layers.RandomCrop(height=224, width=224),  # Random crop maintaining aspect ratio
    ],
    'augmentation_holdout': None,  # No augmentation for holdout loss calculation
    'batch_size': 32,
    'learning_rate': 0.001,
}

data_model_map['cocokp'] = {
    'model': KeypointResNet18,  # Store class, not instance
    'loss': 'mean_squared_error',
    'target_metric_value': 0.01,  # Target low MSE for keypoint regression
    'epochs': 10,
    'holdout_epochs': 5,
    'input_shape': (224, 224, 3),
    'metrics': [keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()],  # Regression metrics
    'n_outputs': 34,  # 17 keypoints * 2 coordinates
    'augmentation': [keras.layers.RandomFlip(mode='horizontal')],
    'batch_size': 32,
    'learning_rate': 0.001,
}

data_model_map['mpi'] = {}
data_model_map['affectnet'] = {}
data_model_map['raf'] = {}

