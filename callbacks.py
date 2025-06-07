from tensorflow import keras
import random
import numpy as np
import tensorflow as tf
from time import time

class PrioritizedDataGenerator(keras.utils.Sequence):
    """
    Data generator that implements prioritized training by sampling from candidate batches
    based on holdout model losses (irreducible losses).
    """
    def __init__(self, x_train, y_train, il_loss_dict, train_batch_size=32, 
                 cand_batch_size=320, steps_per_epoch=None):
        self.x_train = x_train
        self.y_train = y_train
        self.il_loss_dict = il_loss_dict
        self.train_batch_size = train_batch_size
        self.cand_batch_size = cand_batch_size
        self.steps_per_epoch = steps_per_epoch or (len(x_train) // cand_batch_size)
        
        # Create shuffled indices for each epoch
        self.on_epoch_end()
        
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        # Get candidate batch indices
        start_idx = idx * self.cand_batch_size
        end_idx = min(start_idx + self.cand_batch_size, len(self.indices))
        
        # Handle case where we don't have enough samples (wrap around)
        if end_idx - start_idx < self.cand_batch_size:
            cand_indices = self.indices[start_idx:end_idx]
            # Wrap around to get remaining samples
            remaining = self.cand_batch_size - len(cand_indices)
            cand_indices = np.concatenate([cand_indices, self.indices[:remaining]])
        else:
            cand_indices = self.indices[start_idx:end_idx]
        
        # Get IL losses for candidate batch
        cand_losses = np.array([self.il_loss_dict.get(i, 0.0) for i in cand_indices])
        
        # Select training samples based on highest losses (prioritized sampling)
        # Use top-k selection rather than probability sampling for more deterministic results
        top_k_indices = np.argsort(cand_losses)[-self.train_batch_size:]
        selected_indices = cand_indices[top_k_indices]
        
        # Return selected training batch
        x_batch = self.x_train[selected_indices]
        y_batch = self.y_train[selected_indices]
        
        return x_batch, y_batch
    
    def on_epoch_end(self):
        # Shuffle data for next epoch
        self.indices = np.arange(len(self.x_train))
        np.random.shuffle(self.indices)

class RandomDataGenerator(keras.utils.Sequence):
    """
    Data generator that implements random sampling from candidate batches.
    """
    def __init__(self, x_train, y_train, train_batch_size=32, 
                 cand_batch_size=320, steps_per_epoch=None):
        self.x_train = x_train
        self.y_train = y_train
        self.train_batch_size = train_batch_size
        self.cand_batch_size = cand_batch_size
        self.steps_per_epoch = steps_per_epoch or (len(x_train) // cand_batch_size)
        
        # Create shuffled indices for each epoch
        self.on_epoch_end()
        
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        # Get candidate batch indices
        start_idx = idx * self.cand_batch_size
        end_idx = min(start_idx + self.cand_batch_size, len(self.indices))
        
        # Handle case where we don't have enough samples (wrap around)
        if end_idx - start_idx < self.cand_batch_size:
            cand_indices = self.indices[start_idx:end_idx]
            # Wrap around to get remaining samples
            remaining = self.cand_batch_size - len(cand_indices)
            cand_indices = np.concatenate([cand_indices, self.indices[:remaining]])
        else:
            cand_indices = self.indices[start_idx:end_idx]
        
        # Randomly select training samples from candidate batch
        selected_indices = np.random.choice(cand_indices, size=self.train_batch_size, replace=False)
        
        # Return selected training batch
        x_batch = self.x_train[selected_indices]
        y_batch = self.y_train[selected_indices]
        
        return x_batch, y_batch
    
    def on_epoch_end(self):
        # Shuffle data for next epoch
        self.indices = np.arange(len(self.x_train))
        np.random.shuffle(self.indices)

class TrainingStatsCallback(keras.callbacks.Callback):
    """
    Callback to track training statistics for prioritized/random training.
    """
    def __init__(self, train_batch_size, cand_batch_size, training_type="Unknown"):
        super(TrainingStatsCallback, self).__init__()
        self.train_batch_size = train_batch_size
        self.cand_batch_size = cand_batch_size
        self.training_type = training_type
        self.total_samples_trained = 0
        self.total_samples_considered = 0
        
    def on_train_begin(self, logs=None):
        self.start_time = time()
        
    def on_batch_end(self, batch, logs=None):
        self.total_samples_trained += self.train_batch_size
        self.total_samples_considered += self.cand_batch_size
        
    def on_epoch_end(self, epoch, logs=None):
        print(f'\n{self.total_samples_trained} total images trained')
        print(f'{self.total_samples_considered} total images considered')
        print(f'Trained {self.training_type} Model for {time() - self.start_time:.3f}s so far')
        
    def on_train_end(self, logs=None):
        print(f'Took {self.training_type} Model {time() - self.start_time:.3f}s in total to train')


def compute_il_losses(holdout_model, x_train, y_train, batch_size=32):
    """
    Compute importance learning (IL) losses for training data using holdout model.
    Uses the same loss function that the holdout model was compiled with.
    
    Args:
        holdout_model: Trained holdout model (already compiled)
        x_train: Training images
        y_train: Training labels
        batch_size: Batch size for prediction
        
    Returns:
        il_loss_dict: Dictionary mapping sample indices to IL losses
    """
    print("Computing IL losses using holdout model...")
    
    # Get predictions from holdout model
    predictions = holdout_model.predict(x_train, batch_size=batch_size, verbose=0)
    
    # Use the same loss function that the holdout model was compiled with
    loss_fn = holdout_model.compiled_loss._losses[0]  # Get the actual loss function object
    
    # Compute losses for each sample using the compiled loss function
    y_true_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_pred_tensor = tf.convert_to_tensor(predictions, dtype=tf.float32)
    
    # Compute per-sample losses
    il_losses = loss_fn(y_true_tensor, y_pred_tensor).numpy()
    
    # Create dictionary mapping sample index to IL loss
    il_loss_dict = {i: loss for i, loss in enumerate(il_losses)}
    
    loss_name = loss_fn.__class__.__name__
    print(f"IL losses computed for {len(il_loss_dict)} samples (loss: {loss_name})")
    print(f"  Min loss: {il_losses.min():.4f}")
    print(f"  Max loss: {il_losses.max():.4f}")  
    print(f"  Mean loss: {il_losses.mean():.4f}")
    
    return il_loss_dict


def create_tf_data_prioritized_dataset(image_paths, labels, il_loss_dict, 
                                     train_batch_size=32, cand_batch_size=320, 
                                     steps_per_epoch=None, input_shape=(224, 224, 3)):
    """
    Create an optimized tf.data pipeline for prioritized training.
    
    Args:
        image_paths: Array of image file paths
        labels: Array of labels
        il_loss_dict: Dictionary mapping indices to IL losses
        train_batch_size: Final batch size for training
        cand_batch_size: Size of candidate batch for selection
        steps_per_epoch: Number of steps per epoch
        input_shape: Target image shape (H, W, C)
    
    Returns:
        tf.data.Dataset that yields (images, labels) batches
    """
    
    # Convert IL losses to array for faster indexing
    il_losses = np.array([il_loss_dict.get(i, 0.0) for i in range(len(image_paths))])
    
    def load_and_preprocess_image(path, label):
        """Optimized image loading function"""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, input_shape[:2])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def sample_prioritized_batch():
        """Sample a prioritized batch using IL losses"""
        def py_sample_function():
            # Sample candidate indices
            cand_indices = np.random.choice(len(image_paths), size=cand_batch_size, replace=False)
            
            # Get IL losses for candidates
            cand_losses = il_losses[cand_indices]
            
            # Select top-k based on highest losses
            top_k_indices = np.argsort(cand_losses)[-train_batch_size:]
            selected_indices = cand_indices[top_k_indices]
            
            return selected_indices.astype(np.int32)
        
        # Use tf.py_function for custom sampling logic
        selected_indices = tf.py_function(
            py_sample_function, 
            [], 
            tf.int32
        )
        selected_indices.set_shape([train_batch_size])
        
        # Gather selected paths and labels
        selected_paths = tf.gather(image_paths, selected_indices)
        selected_labels = tf.gather(labels, selected_indices)
        
        return selected_paths, selected_labels
    
    # Create dataset generator
    dataset = tf.data.Dataset.from_generator(
        lambda: iter([sample_prioritized_batch() for _ in range(steps_per_epoch or 1000)]),
        output_signature=(
            tf.TensorSpec(shape=(train_batch_size,), dtype=tf.string),
            tf.TensorSpec(shape=(train_batch_size, labels.shape[1]), dtype=tf.float32)
        )
    )
    
    # Apply image loading with parallel processing
    dataset = dataset.map(
        lambda paths, labels: (
            tf.map_fn(
                lambda path: load_and_preprocess_image(path, tf.constant(0.0))[0], 
                paths, 
                parallel_iterations=8,
                dtype=tf.float32
            ),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Add prefetching for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_tf_data_random_dataset(image_paths, labels, train_batch_size=32, 
                                cand_batch_size=320, steps_per_epoch=None, 
                                input_shape=(224, 224, 3)):
    """
    Create an optimized tf.data pipeline for random sampling.
    
    Args:
        image_paths: Array of image file paths
        labels: Array of labels  
        train_batch_size: Final batch size for training
        cand_batch_size: Size of candidate batch for selection
        steps_per_epoch: Number of steps per epoch
        input_shape: Target image shape (H, W, C)
    
    Returns:
        tf.data.Dataset that yields (images, labels) batches
    """
    
    def load_and_preprocess_image(path, label):
        """Optimized image loading function"""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, input_shape[:2])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def sample_random_batch():
        """Sample a random batch"""
        def py_sample_function():
            # Sample candidate indices
            cand_indices = np.random.choice(len(image_paths), size=cand_batch_size, replace=False)
            
            # Randomly select from candidates
            selected_indices = np.random.choice(cand_indices, size=train_batch_size, replace=False)
            
            return selected_indices.astype(np.int32)
        
        # Use tf.py_function for custom sampling logic
        selected_indices = tf.py_function(
            py_sample_function, 
            [], 
            tf.int32
        )
        selected_indices.set_shape([train_batch_size])
        
        # Gather selected paths and labels
        selected_paths = tf.gather(image_paths, selected_indices)
        selected_labels = tf.gather(labels, selected_indices)
        
        return selected_paths, selected_labels
    
    # Create dataset generator
    dataset = tf.data.Dataset.from_generator(
        lambda: iter([sample_random_batch() for _ in range(steps_per_epoch or 1000)]),
        output_signature=(
            tf.TensorSpec(shape=(train_batch_size,), dtype=tf.string),
            tf.TensorSpec(shape=(train_batch_size, labels.shape[1]), dtype=tf.float32)
        )
    )
    
    # Apply image loading with parallel processing
    dataset = dataset.map(
        lambda paths, labels: (
            tf.map_fn(
                lambda path: load_and_preprocess_image(path, tf.constant(0.0))[0], 
                paths, 
                parallel_iterations=8,
                dtype=tf.float32
            ),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Add prefetching for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset