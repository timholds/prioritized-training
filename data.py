import os
import numpy as np
import tensorflow as tf
import zipfile
import pickle
from tensorflow import keras


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def generate_train_test_split(images, labels, val_perc=0.1, holdout_perc=0.3):
    """
    Prepare dataset by splitting into train/val/holdout sets, normalizing, and converting to one-hot.
    
    Args:
        images: Input image data
        labels: Input labels
        val_perc: Fraction of data to use for validation
        holdout_perc: Fraction of data to use for holdout set
        
    Returns:
        (x_train, y_train), (x_val, y_val), (x_holdout, y_holdout)
    """
    assert val_perc + holdout_perc <= 1, 'val_perc + holdout_perc must be <= 1 in order to have data left for training'
    num_classes = len(set(labels.flatten()))
    num_images  = images.shape[0]

    images      = images.astype('float32') / 255
    labels      = keras.utils.to_categorical(labels, num_classes)

    # Shuffle the data
    idxs       = np.arange(num_images)
    np.random.shuffle(idxs)
    images     = images[idxs]
    labels     = labels[idxs]
    
    num_val     = int(num_images * val_perc)
    num_holdout = int(num_images * holdout_perc)
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

    # Print shapes for verification
    print('x_train.shape {}'.format(x_train.shape))
    print('y_train.shape {}'.format(y_train.shape))
    print('x_val.shape {}'.format(x_val.shape))
    print('y_val.shape {}'.format(y_val.shape))
    print('x_holdout.shape {}'.format(x_holdout.shape))
    print('y_holdout.shape {}'.format(y_holdout.shape))

    return (x_train, y_train), (x_val, y_val), (x_holdout, y_holdout)


def get_qmnist():
    """
    Load QMNIST dataset from the local directory or download if needed
    returns:
        (N, 28, 28, 1), (N, 1)
    """
    # Check if QMNIST directory exists
    extract_dir = os.path.join('datasets', 'qmnist')
    file_path = os.path.join(extract_dir, 'MNIST-120k')
    
    if not os.path.exists(file_path):
        print("QMNIST dataset not found at", file_path)
        
        # Check if we have the archive to extract
        if os.path.exists('archive.zip'):
            print("Found archive.zip, extracting...")
            zip_file = zipfile.ZipFile('archive.zip', 'r')
            os.makedirs(extract_dir, exist_ok=True)
            zip_file.extractall(extract_dir)
            zip_file.close()
        else:
            print("Using MNIST dataset as fallback")
            # Just use regular MNIST as fallback
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x = np.concatenate((x_train, x_test))
            y = np.concatenate((y_train, y_test)).reshape(-1, 1)
            return x, y
    
    # Load the qmnist dataset
    print("Loading QMNIST dataset from", file_path)
    qmnist = unpickle(file_path)
    x_qmnist = qmnist['data']
    y_qmnist = qmnist['labels']

    # Load MNIST data
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

    x_mnist = np.concatenate((x_train_mnist, x_test_mnist))
    y_mnist = np.concatenate((y_train_mnist, y_test_mnist)).reshape(-1, 1)

    # Combine MNIST and QMNIST
    images = np.concatenate((x_qmnist, x_mnist))
    labels = np.concatenate((y_qmnist, y_mnist))

    print("QMNIST image dataset shape:" , x_qmnist.shape)
    print("MNIST image dataset shape:", x_mnist.shape)
    print("Final image dataset shape:" , images.shape)

    return images, labels

def get_cifar10():
    """
    Load CIFAR-10 dataset and return images and labels (unprocessed).
    Returns:
        images, labels (N, 32, 32, 3), (N, 1)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).reshape(-1, 1)
    return x, y

def get_cifar100():
    """
    Load CIFAR-100 dataset and return images and labels (unprocessed).
    Returns:
        images, labels (N, 32, 32, 3), (N, 1)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).reshape(-1, 1)
    return x, y

def get_cinic10():
    """
    Load CINIC-10 dataset from datasets/cinic10 directory.
    Assumes directory structure:
        datasets/cinic10/train/classname/*.png
        datasets/cinic10/valid/classname/*.png
        datasets/cinic10/test/classname/*.png

    Returns:
        images: np.ndarray of shape (N, 32, 32, 3)
        labels: np.ndarray of shape (N, 1)
    """
    import glob
    from PIL import Image

    base_dir = os.path.join('datasets', 'cinic10')
    splits = ['train', 'valid', 'test']
    class_names = sorted(os.listdir(os.path.join(base_dir, 'train')))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    images = []
    labels = []

    for split in splits:
        for cls in class_names:
            img_dir = os.path.join(base_dir, split, cls)
            img_files = glob.glob(os.path.join(img_dir, '*.png'))
            for img_file in img_files:
                img = Image.open(img_file).convert('RGB')
                img = img.resize((32, 32))
                images.append(np.array(img))
                labels.append(class_to_idx[cls])

    images = np.stack(images)
    labels = np.array(labels).reshape(-1, 1)
    return images, labels

