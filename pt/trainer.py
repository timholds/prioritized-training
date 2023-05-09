# TODO add option to take in MNIST / CIFAR datasets or pass your own in 
# TODO have option to pass a holdout model pb in, or otherwise to train the holdout model
import tensorflow as tf
from tensorflow import keras
from utils import download_qmnist, prep_data
from models import ConvModelMNIST

# Create tf data objects for MNIST 
images, labels = download_qmnist()
(x_train, y_train), (x_val, y_val), (x_holdout, y_holdout) = prep_data(images, labels)
# train_dataset   = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# val_dataset     = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# holdout_dataset = tf.data.Dataset.from_tensor_slices((x_holdout, y_holdout))

# Create the model
model = ConvModelMNIST()
model.compile(
        optimizer= 'adam',
        loss     = 'categorical_crossentropy',
        metrics  = ['accuracy']
    )

history = model.fit(x_train, y_train, 
                    epochs=1, 
                    validation_data=(x_val, y_val), 
                    verbose=1)
# history = model.fit(train_dataset, 
#                     epochs=1, 
#                     validation_data=val_dataset, 
#                     verbose=1)
