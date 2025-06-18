"""
Utility functions for memory-efficient data loading using tf.data pipelines.
"""
import tensorflow as tf
import numpy as np


def create_validation_dataset(image_paths, labels, batch_size, input_shape):
    """
    Create a tf.data dataset for validation/holdout data that loads images on-demand.
    
    Args:
        image_paths: Array of image file paths
        labels: Array of labels
        batch_size: Batch size for evaluation
        input_shape: Target image shape (H, W, C)
    
    Returns:
        tf.data.Dataset that yields (images, labels) batches
    """
    def load_and_preprocess_image(path, label):
        """Load and preprocess a single image with aspect ratio preservation"""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize with aspect ratio preservation and padding
        target_height, target_width = input_shape[:2]
        image = tf.image.resize_with_pad(
            image, 
            target_height, 
            target_width,
            method=tf.image.ResizeMethod.BILINEAR
        )
        
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Map loading function with parallel calls
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_training_dataset_for_holdout(image_paths, labels, batch_size, input_shape, epochs):
    """
    Create a tf.data dataset for training the holdout model.
    Includes shuffling and repeating for the specified number of epochs.
    
    Args:
        image_paths: Array of image file paths
        labels: Array of labels
        batch_size: Batch size for training
        input_shape: Target image shape (H, W, C)
        epochs: Number of epochs to repeat
    
    Returns:
        tf.data.Dataset that yields (images, labels) batches
    """
    def load_and_preprocess_image(path, label):
        """Load and preprocess a single image with aspect ratio preservation"""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize with aspect ratio preservation and padding
        target_height, target_width = input_shape[:2]
        image = tf.image.resize_with_pad(
            image, 
            target_height, 
            target_width,
            method=tf.image.ResizeMethod.BILINEAR
        )
        
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle with a buffer size
    dataset = dataset.shuffle(buffer_size=min(len(image_paths), 10000))
    
    # Map loading function with parallel calls
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch, repeat, and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def evaluate_on_dataset(model, dataset, steps=None):
    """
    Evaluate a model on a tf.data dataset.
    
    Args:
        model: Keras model to evaluate
        dataset: tf.data.Dataset
        steps: Number of steps to evaluate (None = full dataset)
    
    Returns:
        List of metric values
    """
    return model.evaluate(dataset, steps=steps, verbose=0)


def load_validation_images_batch(paths, batch_size, input_shape):
    """
    Generator that yields batches of validation images for memory-efficient loading.
    
    Args:
        paths: List of image paths
        batch_size: Batch size
        input_shape: Target image shape (H, W, C)
    
    Yields:
        Batches of preprocessed images
    """
    from PIL import Image
    
    num_batches = (len(paths) + batch_size - 1) // batch_size
    target_size = input_shape[:2]
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(paths))
        batch_paths = paths[start_idx:end_idx]
        
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                # Resize with aspect ratio preservation using PIL
                img_ratio = img.width / img.height
                target_ratio = target_size[1] / target_size[0]  # width/height
                
                if img_ratio > target_ratio:
                    # Image is wider, resize based on width
                    new_width = target_size[1]
                    new_height = int(target_size[1] / img_ratio)
                else:
                    # Image is taller, resize based on height
                    new_height = target_size[0]
                    new_width = int(target_size[0] * img_ratio)
                
                img = img.resize((new_width, new_height), Image.BILINEAR)
                
                # Create canvas and paste resized image in center
                canvas = Image.new('RGB', target_size, (0, 0, 0))
                paste_x = (target_size[1] - new_width) // 2
                paste_y = (target_size[0] - new_height) // 2
                canvas.paste(img, (paste_x, paste_y))
                
                batch_images.append(np.array(canvas))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                batch_images.append(np.zeros(target_size + (3,), dtype=np.uint8))
        
        yield np.array(batch_images).astype('float32') / 255.0