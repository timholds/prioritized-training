from models import MLPModel, ConvModel, ResNet18
from tensorflow import keras

data_model_map = {}
data_model_map['qmnist'] = {
    'model': ConvModel,  # Store class, not instance
    'loss': 'categorical_crossentropy',
    'target_accuracy': 0.985,
    'epochs': 2,
    'holdout_epochs': 1,
    'input_shape': (28, 28, 1),
    'metrics': ['accuracy'],
    'n_classes': 10,
    'augmentation': None,
    'batch_size': 32,
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

}

data_model_map['mpi'] = {}
data_model_map['affectnet'] = {}
data_model_map['coco'] = {}
data_model_map['raf'] = {}

