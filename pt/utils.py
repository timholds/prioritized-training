import psutil
import time
import subprocess
import json
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import wget
import gzip
import zipfile
import pickle
 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def download_qmnist():
    # Extract the zipfile
    zip_file = zipfile.ZipFile('archive.zip', 'r')
    extract_dir = 'QMNIST/'
    zip_file.extractall(extract_dir)
    zip_file.close()

    # Load the qmnist dataset
    qmnist = unpickle('QMNIST/MNIST-120k')
    x_qmnist = qmnist['data']
    y_qmnist = qmnist['labels']

    # Load MNIST data
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

    x_mnist = np.concatenate((x_train_mnist, x_test_mnist))
    y_mnist = np.concatenate((y_train_mnist, y_test_mnist)).reshape(-1, 1)

    # Combine MNIST and QMNIST
    x = np.concatenate((x_qmnist, x_mnist))
    y = np.concatenate((y_qmnist, y_mnist))

    print("MNIST image dataset shape:" , x_qmnist.shape)
    print("QMNIST image dataset shape:", x_mnist.shape)
    print("Final image dataset shape:" , x.shape)

    return x, y



def prep_data(images, labels, val_per=.1, holdout_per=.3):#, test_per, holdout_per):
    assert int(val_per + holdout_per) <= 1, 'val_per + holdout_per must be <= 1 in order to have data left for training'
    num_classes = len(set(labels.flatten()))
    num_images  = images.shape[0]

    images      = images.astype('float32') / 255
    labels      = keras.utils.to_categorical(labels, num_classes)

    # Shuffle the data
    idxs       = np.arange(num_images)
    np.random.shuffle(idxs)
    images     = images[idxs]
    labels     = labels[idxs]
    
    num_val     = int(num_images * val_per)
    num_holdout = int(num_images * holdout_per)
    num_train   = num_images - num_holdout - num_val 
    
    x_train   = images[          :num_train]
    x_val     = images[num_train :num_train + num_val]
    x_holdout = images[num_train + num_val:]

    y_train   = labels[          :num_train]
    y_val     = labels[num_train :num_train + num_val]
    y_holdout = labels[num_train + num_val:]
   
    # Make sure images have shape (28, 28, 1)
    x_train    = np.expand_dims(x_train  , -1)
    x_val      = np.expand_dims(x_val    , -1)
    x_holdout  = np.expand_dims(x_holdout, -1)

    print('x_train.shape {}'.format(x_train.shape))
    print('y_train.shape {}'.format(y_train.shape))
    print('x_val.shape {}'.format(x_val.shape))
    print('y_test.shape {}'.format(y_val.shape))
    print('x_holdout.shape {}'.format(x_holdout.shape))
    print('y_holdout.shape {}'.format(y_holdout.shape))

    return (x_train, y_train), (x_val, y_val), (x_holdout, y_holdout)

