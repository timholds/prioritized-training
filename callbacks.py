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


def compute_il_losses_streaming(holdout_model, train_paths, y_train, config, batch_size=32):
    """
    Compute IL losses by loading images in batches instead of all at once.
    """
    print("Computing IL losses using holdout model (streaming)...")
    
    # Process images in batches
    il_losses = []
    num_batches = (len(train_paths) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        if i % 10 == 0:
            print(f"  Processing batch {i+1}/{num_batches}")
        
        # Load batch of images
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(train_paths))
        batch_paths = train_paths[start_idx:end_idx]
        batch_labels = y_train[start_idx:end_idx]
        
        # Load and preprocess batch images
        batch_images = []
        target_size = config['input_shape'][:2]
        for path in batch_paths:
            try:
                from PIL import Image
                img = Image.open(path).convert('RGB')
                img = img.resize(target_size)
                batch_images.append(np.array(img))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                batch_images.append(np.zeros(target_size + (3,), dtype=np.uint8))
        
        batch_images = np.array(batch_images).astype('float32') / 255.0
        
        # Use original compute_il_losses logic
        predictions = holdout_model.predict(batch_images, batch_size=len(batch_images), verbose=0)
        
        # Get loss function from config using Keras loss registry
        loss_name = config['loss']
        loss_fn = tf.keras.losses.get(loss_name)
        # Set reduction to NONE to get per-sample losses
        loss_fn.reduction = tf.keras.losses.Reduction.NONE
        
        y_true_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
        y_pred_tensor = tf.convert_to_tensor(predictions, dtype=tf.float32)
        batch_losses = loss_fn(y_true_tensor, y_pred_tensor).numpy()
        
        il_losses.extend(batch_losses)
    
    # Create dictionary mapping sample index to IL loss
    il_loss_dict = {i: loss for i, loss in enumerate(il_losses)}
    
    print(f"IL losses computed for {len(il_loss_dict)} samples (loss: {config['loss']})")
    print(f"  Min loss: {min(il_losses):.4f}")
    print(f"  Max loss: {max(il_losses):.4f}")
    print(f"  Mean loss: {sum(il_losses)/len(il_losses):.4f}")
    
    return il_loss_dict

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
                                     steps_per_epoch=None, input_shape=(224, 224, 3),
                                     augmentation_layers=None):
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
        augmentation_layers: List of Keras augmentation layers to apply during training
    
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
    
    def epoch_generator():
        """Generate batches for one epoch with proper shuffling"""
        # Create indices for all samples
        all_indices = np.arange(len(image_paths))
        
        # Shuffle all indices at the start of each epoch
        np.random.shuffle(all_indices)
        
        # Create candidate batches by splitting shuffled indices
        for i in range(0, len(all_indices), cand_batch_size):
            # Get candidate batch indices
            cand_indices = all_indices[i:i + cand_batch_size]
            
            # Skip if we don't have enough samples for a full candidate batch
            if len(cand_indices) < cand_batch_size:
                continue
                
            # Get IL losses for candidates
            cand_losses = il_losses[cand_indices]
            
            # Select top-k based on highest losses
            top_k_indices = np.argsort(cand_losses)[-train_batch_size:]
            selected_indices = cand_indices[top_k_indices]
            
            # Gather selected paths and labels
            selected_paths = image_paths[selected_indices]
            selected_labels = labels[selected_indices]
            
            yield selected_paths, selected_labels
    
    # Create dataset for one epoch, then repeat it
    dataset = tf.data.Dataset.from_generator(
        epoch_generator,
        output_signature=(
            tf.TensorSpec(shape=(train_batch_size,), dtype=tf.string),
            tf.TensorSpec(shape=(train_batch_size, labels.shape[1]), dtype=tf.float32)
        )
    ).repeat()  # Repeat indefinitely
    
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
    
    # Apply data augmentation if specified
    if augmentation_layers:
        def apply_augmentation(images, labels):
            # Apply each augmentation layer sequentially to the batch
            augmented_images = images
            for aug_layer in augmentation_layers:
                augmented_images = aug_layer(augmented_images, training=True)
            return augmented_images, labels
        
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add prefetching for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_tf_data_random_dataset(image_paths, labels, train_batch_size=32, 
                                cand_batch_size=320, steps_per_epoch=None, 
                                input_shape=(224, 224, 3), augmentation_layers=None):
    """
    Create an optimized tf.data pipeline for random sampling.
    
    Args:
        image_paths: Array of image file paths
        labels: Array of labels  
        train_batch_size: Final batch size for training
        cand_batch_size: Size of candidate batch for selection
        steps_per_epoch: Number of steps per epoch
        input_shape: Target image shape (H, W, C)
        augmentation_layers: List of Keras augmentation layers to apply during training
    
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
    
    def epoch_generator():
        """Generate batches for one epoch with proper shuffling"""
        # Create indices for all samples
        all_indices = np.arange(len(image_paths))
        
        # Shuffle all indices at the start of each epoch
        np.random.shuffle(all_indices)
        
        # Create candidate batches by splitting shuffled indices
        for i in range(0, len(all_indices), cand_batch_size):
            # Get candidate batch indices
            cand_indices = all_indices[i:i + cand_batch_size]
            
            # Skip if we don't have enough samples for a full candidate batch
            if len(cand_indices) < cand_batch_size:
                continue
            
            # Randomly select train_batch_size samples from the candidate batch
            selected_idx = np.random.choice(len(cand_indices), size=train_batch_size, replace=False)
            selected_indices = cand_indices[selected_idx]
            
            # Gather selected paths and labels
            selected_paths = image_paths[selected_indices]
            selected_labels = labels[selected_indices]
            
            yield selected_paths, selected_labels
    
    # Create dataset for one epoch, then repeat it
    dataset = tf.data.Dataset.from_generator(
        epoch_generator,
        output_signature=(
            tf.TensorSpec(shape=(train_batch_size,), dtype=tf.string),
            tf.TensorSpec(shape=(train_batch_size, labels.shape[1]), dtype=tf.float32)
        )
    ).repeat()  # Repeat indefinitely
    
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
    
    # Apply data augmentation if specified
    if augmentation_layers:
        def apply_augmentation(images, labels):
            # Apply each augmentation layer sequentially to the batch
            augmented_images = images
            for aug_layer in augmentation_layers:
                augmented_images = aug_layer(augmented_images, training=True)
            return augmented_images, labels
        
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add prefetching for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset