# TODO add option to take in MNIST / CIFAR datasets or pass your own in 
# TODO have option to pass a holdout model pb in, or otherwise to train the holdout model
import tensorflow as tf
from tensorflow import keras
from utils import download_qmnist, prep_data
from models import ConvModelMNIST

# Create tf data objects for MNIST 
images, labels = download_qmnist()
(x_train, y_train), (x_val, y_val), (x_holdout, y_holdout) = prep_data(images, labels)
batch_size = 32
train_dataset   = tf.data.Dataset.from_tensor_slices((x_train, y_train))    .batch(batch_size)
val_dataset     = tf.data.Dataset.from_tensor_slices((x_val, y_val))        .batch(batch_size)
holdout_dataset = tf.data.Dataset.from_tensor_slices((x_holdout, y_holdout)).batch(batch_size)
    

# Create the model
model = ConvModelMNIST()
model.compile(
        optimizer= 'adam',
        loss     = 'categorical_crossentropy',
        metrics  = ['accuracy']
    )

history = model.fit(train_dataset, 
                    epochs=1, 
                    validation_data=val_dataset, 
                    verbose=1)
