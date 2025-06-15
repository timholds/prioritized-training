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

def generate_train_test_split(images, labels, val_perc=0.1, holdout_perc=0.3, is_regression=False, use_paths=False):
    """
    Prepare dataset by splitting into train/val/holdout sets, normalizing, and converting to one-hot.
    
    Args:
        images: Input image data or image paths
        labels: Input labels
        val_perc: Fraction of data to use for validation
        holdout_perc: Fraction of data to use for holdout set
        is_regression: Whether this is a regression task (skip one-hot encoding)
        use_paths: If True, images are file paths and won't be normalized
        
    Returns:
        (x_train, y_train), (x_val, y_val), (x_holdout, y_holdout)
    """
    assert val_perc + holdout_perc <= 1, 'val_perc + holdout_perc must be <= 1 in order to have data left for training'
    num_images  = images.shape[0]

    if not use_paths:
        images = images.astype('float32') / 255
    
    if not is_regression:
        num_classes = len(set(labels.flatten()))
        labels      = keras.utils.to_categorical(labels, num_classes)
    else:
        # For regression, ensure labels are float32
        labels = labels.astype('float32')

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
   
    if not use_paths:
        # For MNIST-like datasets, make sure images have shape (28, 28, 1)
        # For other datasets like COCO, keep original shape
        if images.shape[-1] == 1 or len(images.shape) == 3:  # MNIST case
            x_train    = np.expand_dims(x_train  , -1) if len(x_train.shape) == 3 else x_train
            x_val      = np.expand_dims(x_val    , -1) if len(x_val.shape) == 3 else x_val
            x_holdout  = np.expand_dims(x_holdout, -1) if len(x_holdout.shape) == 3 else x_holdout

    # Print shapes for verification
    if use_paths:
        print('x_train paths: {}'.format(len(x_train)))
        print('x_val paths: {}'.format(len(x_val)))
        print('x_holdout paths: {}'.format(len(x_holdout)))
    else:
        print('x_train.shape {}'.format(x_train.shape))
        print('x_val.shape {}'.format(x_val.shape))
        print('x_holdout.shape {}'.format(x_holdout.shape))
    
    print('y_train.shape {}'.format(y_train.shape))
    print('y_val.shape {}'.format(y_val.shape))
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

def get_cocoreg_paths(max_samples=None):
    """
    Load COCO2017 dataset paths and labels for efficient tf.data loading.
    
    Args:
        max_samples: int, maximum number of samples to load (for testing)
    
    Returns:
        image_paths: np.ndarray of shape (N,) - image file paths
        labels: np.ndarray of shape (N, 4) - regression targets (bbox coordinates)
    """
    import json
    import glob
    from PIL import Image
    
    base_dir = os.path.join('datasets', 'coco2017')
    
    # Load annotations
    train_ann_path = os.path.join(base_dir, 'annotations', 'instances_train2017.json')
    val_ann_path = os.path.join(base_dir, 'annotations', 'instances_val2017.json')
    
    with open(train_ann_path, 'r') as f:
        train_data = json.load(f)
    with open(val_ann_path, 'r') as f:
        val_data = json.load(f)
    
    # Combine train and val data
    all_images = train_data['images'] + val_data['images']
    all_annotations = train_data['annotations'] + val_data['annotations']
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in all_annotations:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    image_paths = []
    labels = []
    
    print(f"Processing {len(all_images)} COCO images...")
    
    for i, img_info in enumerate(all_images):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(all_images)} images")
            
        # Stop if we've reached max_samples
        if max_samples and len(images) >= max_samples:
            break
            
        # Determine which split this image belongs to
        img_filename = img_info['file_name']
        if img_filename.startswith('0000000'):  # train2017 format
            img_split = 'train2017'
        else:
            img_split = 'val2017'
            
        img_path = os.path.join(base_dir, img_split, img_filename)
        
        # Skip if image file doesn't exist
        if not os.path.exists(img_path):
            continue
            
        try:
            # Calculate regression target from annotations
            image_id = img_info['id']
            anns = image_annotations.get(image_id, [])
            
            # Skip images with no annotations
            if not anns:
                continue
                
            # Regression target: bounding box coordinates [x, y, width, height] of largest object
            largest_ann = max(anns, key=lambda a: a['area'])
            bbox = largest_ann['bbox']  # [x, y, width, height]
            
            # Normalize bbox coordinates to [0, 1]
            img_width, img_height = img_info['width'], img_info['height']
            normalized_bbox = [
                bbox[0] / img_width,   # x
                bbox[1] / img_height,  # y  
                bbox[2] / img_width,   # width
                bbox[3] / img_height   # height
            ]
            
            # Store path instead of loading image
            image_paths.append(img_path)
            labels.append(normalized_bbox)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    image_paths = np.array(image_paths, dtype=str)
    labels = np.array(labels)  # Shape: (N, 4) for bbox coordinates
    
    print(f"Loaded {len(image_paths)} COCO image paths")
    print(f"Label statistics - min: {labels.min():.4f}, max: {labels.max():.4f}, mean: {labels.mean():.4f}")
    
    return image_paths, labels

def get_cocokp(max_samples=None):
    """
    Load COCO2017 dataset from datasets/coco2017 directory for keypoint regression tasks.
    Creates regression targets from person keypoint annotations.
    
    Args:
        max_samples: int, maximum number of samples to load (for testing)
    
    Returns:
        images: np.ndarray of shape (N, H, W, 3) - resized images
        labels: np.ndarray of shape (N, 34) - flattened keypoint coordinates (17 keypoints * 2 coords)
    """
    import json
    from PIL import Image
    
    base_dir = os.path.join('datasets', 'coco2017')
    
    # Load keypoint annotations
    train_kp_path = os.path.join(base_dir, 'annotations', 'person_keypoints_train2017.json')
    val_kp_path = os.path.join(base_dir, 'annotations', 'person_keypoints_val2017.json')
    
    with open(train_kp_path, 'r') as f:
        train_data = json.load(f)
    with open(val_kp_path, 'r') as f:
        val_data = json.load(f)
    
    # Combine train and val data
    all_images = train_data['images'] + val_data['images']
    all_annotations = train_data['annotations'] + val_data['annotations']
    
    # Create image_id to image info mapping
    image_info = {img['id']: img for img in all_images}
    
    images = []
    labels = []
    target_size = (224, 224)
    
    print(f"Processing {len(all_annotations)} COCO keypoint annotations...")
    
    for i, ann in enumerate(all_annotations):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(all_annotations)} annotations")
        
        # Stop if we've reached max_samples
        if max_samples and len(images) >= max_samples:
            break
        
        # Skip annotations without keypoints or with crowd=1
        if ann.get('iscrowd', 0) == 1 or 'keypoints' not in ann:
            continue
            
        keypoints = ann['keypoints']
        if len(keypoints) != 51:  # 17 keypoints * 3 (x, y, visibility)
            continue
            
        # Get image info
        image_id = ann['image_id']
        if image_id not in image_info:
            continue
            
        img_info = image_info[image_id]
        img_filename = img_info['file_name']
        
        # Determine split
        if 'train2017' in train_kp_path and any(img['id'] == image_id for img in train_data['images']):
            img_split = 'train2017'
        else:
            img_split = 'val2017'
            
        img_path = os.path.join(base_dir, img_split, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img = img.resize(target_size)
            img_array = np.array(img)
            
            # Extract and normalize keypoints (x, y coordinates only, ignore visibility)
            kp_coords = []
            for j in range(0, len(keypoints), 3):
                x, y, v = keypoints[j], keypoints[j+1], keypoints[j+2]
                # Normalize coordinates to [0, 1] and scale to target size
                norm_x = (x / orig_w) if orig_w > 0 else 0.0
                norm_y = (y / orig_h) if orig_h > 0 else 0.0
                kp_coords.extend([norm_x, norm_y])
            
            # Only include if we have valid keypoints (not all zeros)
            if any(coord > 0 for coord in kp_coords):
                images.append(img_array)
                labels.append(kp_coords)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    labels = np.array(labels)  # Shape: (N, 34) for 17 keypoints * 2 coords
    
    print(f"Loaded {len(images)} COCO keypoint images")
    print(f"Label shape: {labels.shape}")
    print(f"Keypoint statistics - min: {labels.min():.4f}, max: {labels.max():.4f}, mean: {labels.mean():.4f}")
    
    return images, labels


def get_cocokp_paths(max_samples=None):
    """
    Load COCO2017 keypoint dataset paths and labels for efficient tf.data loading.
    
    Args:
        max_samples: int, maximum number of samples to load (for testing)
    
    Returns:
        image_paths: list of shape (N,) - image file paths
        labels: np.ndarray of shape (N, 34) - keypoint regression targets (17 keypoints * 2 coords)
    """
    import json
    
    base_dir = os.path.join('datasets', 'coco2017')
    
    # Load keypoint annotations
    train_kp_path = os.path.join(base_dir, 'annotations', 'person_keypoints_train2017.json')
    val_kp_path = os.path.join(base_dir, 'annotations', 'person_keypoints_val2017.json')
    
    if not os.path.exists(train_kp_path) or not os.path.exists(val_kp_path):
        raise FileNotFoundError(f"COCO keypoint annotations not found in {base_dir}/annotations/")
    
    with open(train_kp_path, 'r') as f:
        train_data = json.load(f)
    with open(val_kp_path, 'r') as f:
        val_data = json.load(f)
    
    # Combine all annotations
    all_annotations = train_data['annotations'] + val_data['annotations']
    
    # Create image info mapping (combine train and val images)
    image_info = {}
    for img in train_data['images']:
        image_info[img['id']] = img
    for img in val_data['images']:
        image_info[img['id']] = img
    
    image_paths = []
    labels = []
    
    print(f"Processing {len(all_annotations)} COCO keypoint annotations...")
    
    for i, ann in enumerate(all_annotations):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(all_annotations)} annotations")
            
        # Stop if we've reached max_samples
        if max_samples and len(image_paths) >= max_samples:
            break
            
        keypoints = ann['keypoints']
        if len(keypoints) != 51:  # 17 keypoints * 3 (x, y, visibility)
            continue
            
        # Get image info
        image_id = ann['image_id']
        if image_id not in image_info:
            continue
            
        img_info = image_info[image_id]
        img_filename = img_info['file_name']
        
        # Determine split and create full path
        if 'train2017' in train_kp_path and any(img['id'] == image_id for img in train_data['images']):
            img_split = 'train2017'
        else:
            img_split = 'val2017'
            
        img_path = os.path.join(base_dir, img_split, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        # Extract and normalize keypoints (x, y coordinates only, ignore visibility)
        orig_w, orig_h = img_info['width'], img_info['height']
        kp_coords = []
        for j in range(0, len(keypoints), 3):
            x, y, v = keypoints[j], keypoints[j+1], keypoints[j+2]
            # Normalize coordinates to [0, 1]
            norm_x = (x / orig_w) if orig_w > 0 else 0.0
            norm_y = (y / orig_h) if orig_h > 0 else 0.0
            kp_coords.extend([norm_x, norm_y])
        
        # Only include if we have valid keypoints (not all zeros)
        if any(coord > 0 for coord in kp_coords):
            image_paths.append(img_path)
            labels.append(kp_coords)
    
    labels = np.array(labels)  # Shape: (N, 34) for 17 keypoints * 2 coords
    
    print(f"Loaded {len(image_paths)} COCO keypoint image paths")
    print(f"Label shape: {labels.shape}")
    print(f"Keypoint statistics - min: {labels.min():.4f}, max: {labels.max():.4f}, mean: {labels.mean():.4f}")
    
    return image_paths, labels

