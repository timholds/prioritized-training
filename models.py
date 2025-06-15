import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import Model

# Removed circular import - data_model_map imports this module

class MLPModel(keras.Model):
    ''' A simple MLP model for QMNIST classification tasks '''
    def __init__(self, num_classes=10, hidden_units=512, input_shape=None):
        super(MLPModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.input_shape_ = input_shape
        if input_shape:
            self.build((None,) + tuple(input_shape))

    def create_model(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape_ if self.input_shape_ else (28, 28, 1)
        model = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(self.hidden_units, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(self.hidden_units, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        return model

    def build(self, input_shape):
        # Build a dummy model to initialize weights
        self.create_model(input_shape[1:])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'hidden_units': self.hidden_units,
            'input_shape': self.input_shape_
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConvModel(keras.Model):
    def __init__(self, num_classes=10, input_shape=None):
        super(ConvModel, self).__init__()
        self.num_classes = num_classes
        self.input_shape_ = input_shape
        if input_shape:
            self.build((None,) + tuple(input_shape))

    def create_model(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape_ if self.input_shape_ else (28, 28, 1)
        
        model = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.SpatialDropout2D(0.2),  # Add this
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.SpatialDropout2D(0.2),  # Add this
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        return model

    def build(self, input_shape):
        # Build a dummy model to initialize weights
        self.create_model(input_shape[1:])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.input_shape_
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# from tensorflow.keras import layers, Model

class BasicBlock(layers.Layer):
    def __init__(self, filters, stride=1, downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.downsample = downsample
        if downsample:
            self.downsample_conv = layers.Conv2D(filters, 1, strides=stride, use_bias=False)
            self.downsample_bn = layers.BatchNormalization()
        
        self.add = layers.Add()

    def call(self, inputs, training=None):
        identity = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity, training=training)
        
        x = self.add([x, identity])
        x = self.relu(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.conv1.filters,
            'stride': self.conv1.strides[0],
            'downsample': self.downsample
        })
        return config

class ResNet18Model(tf.keras.Model):
    def __init__(self, num_classes=None, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.input_shape_ = input_shape
        
        # Input preprocessing (adapted for 32x32 images - no downsampling)
        self.conv1 = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
        # Residual blocks (no downsampling for 32x32 inputs)
        self.layer1 = self._make_layer(64, 2, in_filters=64)
        self.layer2 = self._make_layer(128, 2, stride=1, in_filters=64)
        self.layer3 = self._make_layer(256, 2, stride=1, in_filters=128)
        self.layer4 = self._make_layer(512, 2, stride=1, in_filters=256)
        
        # Output
        self.gap = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes) if num_classes else None
        
        # Build the model
        if input_shape:
            self.build((None,) + tuple(input_shape))

    def _make_layer(self, filters, blocks, stride=1, in_filters=None):
        layer = tf.keras.Sequential()
        # First block needs downsampling if input/output filters differ or stride != 1
        downsample_needed = (stride != 1) or (in_filters != filters)
        layer.add(BasicBlock(filters, stride, downsample=downsample_needed))
        
        # Subsequent blocks
        for _ in range(1, blocks):
            layer.add(BasicBlock(filters))
        
        return layer

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        
        x = self.gap(x)
        if self.classifier:
            x = self.classifier(x)
        return x

    def build(self, input_shape):
        # Initialize the model by calling it once
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.input_shape_
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ResNet18(tf.keras.Model):
    def __init__(self, num_outputs=None, input_shape=None, output_activation=None, **kwargs):
        """
        Flexible ResNet18 for classification or regression.
        Args:
            num_outputs: Number of outputs (classes for classification, regression targets for regression)
            input_shape: Input shape tuple
            output_activation: Activation for output layer (e.g., 'softmax' for classification, 'linear' for regression)
        """
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.input_shape_ = input_shape
        self.output_activation = output_activation
        # Input preprocessing - adapt for image size
        if input_shape and input_shape[0] > 64:  # For larger images like 224x224
            self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)
            self.bn1 = layers.BatchNormalization()
            self.relu = layers.ReLU()
            self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
            # Residual blocks with proper downsampling
            self.layer1 = self._make_layer(64, 2, in_filters=64)
            self.layer2 = self._make_layer(128, 2, stride=2, in_filters=64)
            self.layer3 = self._make_layer(256, 2, stride=2, in_filters=128) 
            self.layer4 = self._make_layer(512, 2, stride=2, in_filters=256)
            self.use_maxpool = True
        else:  # For smaller images like 32x32
            self.conv1 = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)
            self.bn1 = layers.BatchNormalization()
            self.relu = layers.ReLU()
            # Residual blocks (no downsampling for 32x32 inputs)
            self.layer1 = self._make_layer(64, 2, in_filters=64)
            self.layer2 = self._make_layer(128, 2, stride=1, in_filters=64)
            self.layer3 = self._make_layer(256, 2, stride=1, in_filters=128)
            self.layer4 = self._make_layer(512, 2, stride=1, in_filters=256)
            self.use_maxpool = False
        # Output
        if self.num_outputs:
            act = self.output_activation if self.output_activation is not None else 'softmax'
            self.classifier = layers.Dense(self.num_outputs, activation=act)
        else:
            self.classifier = None
        self.gap = layers.GlobalAveragePooling2D()
        # Build the model
        if input_shape:
            self.build((None,) + tuple(input_shape))

    def _make_layer(self, filters, blocks, stride=1, in_filters=None):
        layer = tf.keras.Sequential()
        # First block needs downsampling if input/output filters differ or stride != 1
        downsample_needed = (stride != 1) or (in_filters != filters)
        layer.add(BasicBlock(filters, stride, downsample=downsample_needed))
        
        # Subsequent blocks
        for _ in range(1, blocks):
            layer.add(BasicBlock(filters))
        
        return layer

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        if self.use_maxpool:
            x = self.maxpool(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.gap(x)
        if self.classifier:
            x = self.classifier(x)
        return x

    def build(self, input_shape):
        # Initialize the model by calling it once
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_outputs': self.num_outputs,
            'input_shape': self.input_shape_,
            'output_activation': self.output_activation
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Example usage:
# conv_model = ConvModel(num_classes=10, input_shape=(28, 28, 1))
# mlp_model = MLPModel(num_classes=10, hidden_units=512, input_shape=(28, 28, 1))
# resnet_model = ResNet18(num_classes=10, input_shape=(32, 32, 3))


# Example dataset, replace with actual argument parsing if needed
# args.dataset = 'qmnist'  
# model_uncompiled = data_model_map[args.dataset]['model'] # type mlp_model
# loss = data_model_map[args.dataset]['loss']
# metrics = data_model_map[args.dataset]['metrics']
# model = compile_model(model_uncompiled, loss=loss)


class KeypointResNet18(tf.keras.Model):
    """
    ResNet18 with 2-layer MLP head for keypoint regression.
    Specialized for COCO keypoint detection (17 keypoints = 34 coordinates).
    """
    def __init__(self, num_outputs=34, input_shape=(224, 224, 3), **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.input_shape_ = input_shape
        
        # ResNet18 backbone (adapted for larger 224x224 images with proper downsampling)
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        # Residual blocks with proper downsampling for 224x224 -> 7x7 feature maps
        self.layer1 = self._make_layer(64, 2, stride=1, in_filters=64)
        self.layer2 = self._make_layer(128, 2, stride=2, in_filters=64)
        self.layer3 = self._make_layer(256, 2, stride=2, in_filters=128)
        self.layer4 = self._make_layer(512, 2, stride=2, in_filters=256)
        
        # Global average pooling to get 512-dim feature vector
        self.gap = layers.GlobalAveragePooling2D()
        
        # 2-layer MLP head for keypoint regression
        self.mlp_layer1 = layers.Dense(256, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.mlp_layer2 = layers.Dense(self.num_outputs, activation='linear')
        
        # Build the model
        if input_shape:
            self.build((None,) + tuple(input_shape))

    def _make_layer(self, filters, blocks, stride=1, in_filters=None):
        layer = tf.keras.Sequential()
        # First block needs downsampling if input/output filters differ or stride != 1
        downsample_needed = (stride != 1) or (in_filters != filters)
        layer.add(BasicBlock(filters, stride, downsample=downsample_needed))
        
        # Subsequent blocks
        for _ in range(1, blocks):
            layer.add(BasicBlock(filters))
        
        return layer

    def call(self, inputs, training=None):
        # ResNet18 backbone
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        
        # Global average pooling
        x = self.gap(x)
        
        # 2-layer MLP head
        x = self.mlp_layer1(x)
        x = self.dropout1(x, training=training)
        x = self.mlp_layer2(x)
        
        return x

    def build(self, input_shape):
        # Initialize the model by calling it once
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs)


def compile_model(model, 
            loss='categorical_crossentropy', 
            learning_rate=0.0001, 
            metrics=['accuracy']):

    ''' Compile the model with a standard optimizer and loss function '''
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    return model






