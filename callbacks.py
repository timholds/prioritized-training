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


def compute_il_losses_streaming(holdout_model, train_paths, y_train, config, batch_size=128, use_cache=True):
    """
    Compute IL losses using efficient tf.data pipeline.
    Supports caching for faster subsequent runs.
    """
    # Try to load cached IL losses first
    if use_cache:
        try:
            from coco_cache import COCOCache
            cache = COCOCache()
            cached_losses = cache.load_cached_il_losses(config)
            if cached_losses is not None:
                print(f"Using cached IL losses for {len(cached_losses)} samples")
                return cached_losses
        except ImportError:
            pass  # Fall back to regular computation
    
    print("Computing IL losses using holdout model (streaming)...")
    
    # Create tf.data dataset for efficient batch processing
    from data_utils import create_validation_dataset
    dataset = create_validation_dataset(train_paths, y_train, batch_size, config['input_shape'])
    
    # Get loss function from config  
    loss_name = config['loss']
    if isinstance(loss_name, str) and loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 'smooth_l1', 'keypoint_loss_with_visibility', 'keypoint_mse_with_visibility']:
        from keypoint_losses import create_keypoint_loss
        loss_kwargs = config.get('loss_kwargs', {})
        loss_fn = create_keypoint_loss(loss_name, **loss_kwargs)
    else:
        loss_fn = tf.keras.losses.get(loss_name)
        loss_fn.reduction = tf.keras.losses.Reduction.NONE
    
    # Process all batches efficiently
    il_losses = []
    num_batches = (len(train_paths) + batch_size - 1) // batch_size
    
    for batch_idx, (batch_images, batch_labels) in enumerate(dataset):
        if batch_idx % 50 == 0:  # Reduce print frequency
            print(f"  Processing batch {batch_idx+1}/{num_batches}")
        
        # Get predictions
        predictions = holdout_model(batch_images, training=False)
        
        # Compute losses
        if loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 'keypoint_loss_with_visibility', 'keypoint_mse_with_visibility']:
            batch_losses = loss_fn(batch_labels, predictions, reduction='none').numpy()
        else:
            batch_losses = loss_fn(batch_labels, predictions).numpy()
        
        il_losses.extend(batch_losses)
    
    # Create dictionary mapping sample index to IL loss
    il_loss_dict = {i: loss for i, loss in enumerate(il_losses)}
    
    print(f"IL losses computed for {len(il_loss_dict)} samples (loss: {config['loss']})")
    print(f"  Min loss: {min(il_losses):.4f}")
    print(f"  Max loss: {max(il_losses):.4f}")
    print(f"  Mean loss: {sum(il_losses)/len(il_losses):.4f}")
    
    # Cache the IL losses if caching is enabled
    if use_cache:
        try:
            from coco_cache import COCOCache
            cache = COCOCache()
            cache.cache_il_losses(config, il_loss_dict)
        except ImportError:
            pass  # Continue without caching
    
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


def create_tf_data_prioritized_dataset(model, image_paths, labels, il_loss_dict,
                                     train_batch_size=32, cand_batch_size=320,
                                     steps_per_epoch=None, input_shape=(224, 224, 3),
                                     augmentation_layers=None, config=None,
                                     verbose=False):
    """
    Create a tf.data pipeline for EFFICIENT prioritized training.
    
    This implementation:
    1. Computes current model losses ONLY on candidate batches (not all samples)
    2. Calculates reducible losses (current_loss - il_loss)
    3. Selects samples with highest reducible losses
    4. Uses current model weights for every selection (no stale losses)
    
    Args:
        model: The model being trained
        image_paths: Array of image file paths
        labels: Array of labels
        il_loss_dict: Dictionary mapping indices to IL losses
        train_batch_size: Final batch size for training
        cand_batch_size: Size of candidate batch for selection
        steps_per_epoch: Number of steps per epoch
        input_shape: Target image shape (H, W, C)
        augmentation_layers: List of Keras augmentation layers
        config: Configuration dictionary with loss information
        verbose: Whether to print debug information
    
    Returns:
        tf.data.Dataset that yields (images, labels) batches
    """
    
    # Convert IL losses to array for faster indexing
    il_losses = np.array([il_loss_dict.get(i, 0.0) for i in range(len(image_paths))])
    
    # Get loss function
    loss_name = config['loss']
    if isinstance(loss_name, str) and loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 
                                                    'smooth_l1', 'keypoint_loss_with_visibility', 
                                                    'keypoint_mse_with_visibility']:
        from keypoint_losses import create_keypoint_loss
        loss_kwargs = config.get('loss_kwargs', {})
        loss_fn = create_keypoint_loss(loss_name, **loss_kwargs)
    else:
        loss_fn = tf.keras.losses.get(loss_name)
        if hasattr(loss_fn, 'reduction'):
            loss_fn.reduction = tf.keras.losses.Reduction.NONE
    
    # Track batch count for logging
    batch_counter = {'count': 0}
    
    def load_and_preprocess_image(path, label):
        """Optimized image loading function with aspect ratio preservation"""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        # Use aspect ratio preserving resize to match validation preprocessing
        image = tf.image.resize_with_pad(
            image, 
            input_shape[0], 
            input_shape[1],
            method=tf.image.ResizeMethod.BILINEAR
        )
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def compute_reducible_losses_tf(images, labels, indices):
        """Compute reducible losses using TF operations for efficiency"""
        # Get predictions
        predictions = model(images, training=False)
        
        # Compute current losses - need per-sample losses
        if loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 'keypoint_loss_with_visibility', 'keypoint_mse_with_visibility']:
            current_losses = loss_fn(labels, predictions, reduction='none')
        else:
            current_losses = loss_fn(labels, predictions)
        
        # Get IL losses for these indices
        il_losses_tensor = tf.constant(il_losses, dtype=tf.float32)
        il_losses_batch = tf.gather(il_losses_tensor, indices)
        
        # Calculate reducible losses
        reducible_losses = current_losses - il_losses_batch
        
        return reducible_losses
    
    def epoch_generator():
        """Generate batches for one epoch with prioritization"""
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
            
            # For efficiency, we'll compute losses in TF graph
            # First, yield the candidate indices for processing
            yield cand_indices
    
    # Create initial dataset of candidate indices
    indices_dataset = tf.data.Dataset.from_generator(
        epoch_generator,
        output_signature=tf.TensorSpec(shape=(cand_batch_size,), dtype=tf.int64)
    ).repeat()
    
    def process_candidates(cand_indices_tensor):
        """Process candidate batch and select training samples"""
        # Convert tensor to numpy for indexing
        cand_indices = cand_indices_tensor.numpy()
        
        # Get paths and labels for candidates
        cand_paths = image_paths[cand_indices]
        cand_labels = labels[cand_indices]
        
        # Load candidate images
        cand_images = []
        for path in cand_paths:
            img, _ = load_and_preprocess_image(path, None)
            cand_images.append(img)
        cand_images = tf.stack(cand_images)
        
        # Compute reducible losses ONLY for candidates
        reducible_losses = compute_reducible_losses_tf(
            cand_images, cand_labels, cand_indices_tensor
        )
        
        # Select top-k based on reducible losses
        top_k_values, top_k_indices = tf.nn.top_k(reducible_losses, k=train_batch_size)
        selected_local_indices = top_k_indices.numpy()
        selected_global_indices = cand_indices[selected_local_indices]
        
        # Increment counter here for proper synchronization
        batch_counter['count'] += 1
        
        # Log progress with explicit flush for tf.py_function compatibility
        if verbose and batch_counter['count'] % 50 == 0:
            reducible_np = reducible_losses.numpy()
            selected_reducible = reducible_np[selected_local_indices]
            import sys
            print(f"[PT-Efficient] Batch {batch_counter['count']}: Reducible losses - "
                  f"Candidate mean: {reducible_np.mean():.4f}, "
                  f"Selected mean: {selected_reducible.mean():.4f}, "
                  f"Selected max: {selected_reducible.max():.4f}", flush=True)
            sys.stdout.flush()
        
        # Return selected indices instead of paths/labels to avoid redundant loading
        return selected_global_indices
    
    # Map candidate processing to get selected indices
    dataset = indices_dataset.map(
        lambda indices: tf.py_function(
            process_candidates,
            [indices],
            tf.int64  # Return indices instead of paths/labels
        ),
        num_parallel_calls=1  # Sequential to maintain model state consistency
    )
    
    # Set shape for indices and gather paths/labels
    dataset = dataset.map(
        lambda indices: (
            tf.ensure_shape(indices, [train_batch_size]),
            indices
        )
    )
    
    # Gather paths and labels based on selected indices
    dataset = dataset.map(
        lambda indices, _: (
            tf.gather(tf.constant(image_paths), indices),
            tf.gather(tf.constant(labels, dtype=tf.float32), indices)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    def load_batch_images_py(paths):
        """Load a batch of images using Python function (avoids tf.map_fn TensorArray issues)"""
        import cv2
        
        paths_np = paths.numpy()
        images = []
        
        for path in paths_np:
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            try:
                # Try OpenCV first
                img = cv2.imread(str(path))
                if img is None:
                    # Fallback to TF
                    img_data = tf.io.read_file(str(path))
                    img = tf.image.decode_jpeg(img_data, channels=3).numpy()
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize and normalize
                img = cv2.resize(img, input_shape[:2])
                img = img.astype(np.float32) / 255.0
                images.append(img)
                
            except Exception as e:
                # Create dummy image if loading fails
                dummy_img = np.zeros(input_shape, dtype=np.float32)
                images.append(dummy_img)
        
        return np.stack(images)
    
    # Load images using tf.py_function (more stable than tf.map_fn)
    dataset = dataset.map(
        lambda paths, labels: (
            tf.py_function(
                func=load_batch_images_py,
                inp=[paths],
                Tout=tf.float32
            ),
            labels
        ),
        num_parallel_calls=1  # Sequential to avoid race conditions
    )
    
    # Set shape for the images tensor
    dataset = dataset.map(
        lambda images, labels: (
            tf.ensure_shape(images, (train_batch_size,) + input_shape),
            labels
        )
    )
    
    # Apply data augmentation if specified
    if augmentation_layers:
        def apply_augmentation(images, labels):
            augmented_images = images
            for aug_layer in augmentation_layers:
                augmented_images = aug_layer(augmented_images, training=True)
            return augmented_images, labels
        
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add prefetching for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_tf_data_prioritized_dataset_async(model, image_paths, labels, il_loss_dict,
                                           train_batch_size=32, cand_batch_size=320,
                                           steps_per_epoch=None, input_shape=(224, 224, 3),
                                           augmentation_layers=None, config=None,
                                           verbose=False, update_freq=5, use_cpu_inference=False):
    """
    Create a tf.data pipeline for ASYNC prioritized training with ~50% speedup.
    
    This implementation:
    1. Uses a separate inference model copy to compute losses asynchronously
    2. Overlaps loss computation for batch n+1 while training batch n
    3. Updates inference model weights every `update_freq` batches
    4. Pre-computes losses using double buffering
    
    Args:
        model: The model being trained (target model)
        image_paths: Array of image file paths
        labels: Array of labels
        il_loss_dict: Dictionary mapping indices to IL losses
        train_batch_size: Final batch size for training
        cand_batch_size: Size of candidate batch for selection
        steps_per_epoch: Number of steps per epoch
        input_shape: Target image shape (H, W, C)
        augmentation_layers: List of Keras augmentation layers
        config: Configuration dictionary with loss information
        verbose: Whether to print debug information
        update_freq: How often to sync inference model weights (every N batches)
        use_cpu_inference: If True, place inference model on CPU to save GPU memory
    
    Returns:
        tf.data.Dataset that yields (images, labels) batches
    """
    import threading
    import queue
    import concurrent.futures
    
    # Convert IL losses and image paths to arrays for faster indexing
    il_losses = np.array([il_loss_dict.get(i, 0.0) for i in range(len(image_paths))])
    image_paths = np.array(image_paths)  # Convert to numpy array for advanced indexing
    
    # Get loss function
    loss_name = config['loss']
    if isinstance(loss_name, str) and loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 
                                                    'smooth_l1', 'keypoint_loss_with_visibility', 
                                                    'keypoint_mse_with_visibility']:
        from keypoint_losses import create_keypoint_loss
        loss_kwargs = config.get('loss_kwargs', {})
        loss_fn = create_keypoint_loss(loss_name, **loss_kwargs)
    else:
        loss_fn = tf.keras.losses.get(loss_name)
        if hasattr(loss_fn, 'reduction'):
            loss_fn.reduction = tf.keras.losses.Reduction.NONE
    
    # Create inference model copy by recreating from config instead of cloning
    # This avoids serialization issues with custom loss functions
    device_name = '/CPU:0' if use_cpu_inference else '/GPU:0'
    print(f"[Async-PT] Creating inference model copy on {device_name} for async loss computation...")
    with tf.device(device_name):
        # Create inference model by cloning or using config
        if config.get('model') and hasattr(config['model'], '__name__'):
            # Full config provided with model class
            model_class = config['model']
            if model_class.__name__ == 'KeypointResNet18':
                inference_model = model_class(
                    num_outputs=config['n_outputs'], 
                    input_shape=config['input_shape']
                )
            elif model_class.__name__ == 'ResNet18':
                inference_model = model_class(
                    num_outputs=config.get('n_outputs', config.get('n_classes')), 
                    input_shape=config['input_shape'],
                    output_activation=config.get('output_activation')
                )
            else:
                # Handle other models that might use create_model()
                if hasattr(model_class, 'create_model'):
                    inference_model = model_class(
                        num_classes=config.get('n_classes', config.get('n_outputs')), 
                        input_shape=config['input_shape']
                    ).create_model()
                else:
                    # Fallback: try to create directly
                    inference_model = model_class(
                        num_outputs=config.get('n_outputs', config.get('n_classes')), 
                        input_shape=config['input_shape']
                    )
        else:
            # Simple case: clone the existing model (backward compatibility)
            inference_model = tf.keras.models.clone_model(model)
            inference_model.compile(optimizer=model.optimizer, loss=model.loss)
        
        # Copy weights from the main model
        inference_model.set_weights(model.get_weights())
        # Compile inference model to ensure proper initialization
        inference_model.compile(optimizer='adam', loss=loss_fn)
    
    # Track state
    batch_counter = {'count': 0, 'last_sync': 0}
    results_queue = queue.Queue(maxsize=3)  # Buffer for 3 batches ahead
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    def load_and_preprocess_image_batch(paths):
        """Load and preprocess image batch using numpy operations (more stable)"""
        import cv2
        
        # Convert to numpy if tensor
        if isinstance(paths, tf.Tensor):
            paths = paths.numpy()
        
        # Load images using OpenCV (more stable than tf.map_fn)
        images = []
        for path in paths:
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            try:
                # Load image with cv2
                img = cv2.imread(str(path))
                if img is None:
                    # Fallback to TF for JPEG files
                    img_data = tf.io.read_file(str(path))
                    img = tf.image.decode_jpeg(img_data, channels=3).numpy()
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, input_shape[:2])
                # Normalize
                img = img.astype(np.float32) / 255.0
                images.append(img)
                
            except Exception as e:
                # Create dummy image if loading fails
                dummy_img = np.zeros(input_shape, dtype=np.float32)
                images.append(dummy_img)
                if verbose:
                    print(f"[Async-PT] Failed to load image {path}: {e}")
        
        return tf.constant(np.stack(images), dtype=tf.float32)
    
    def compute_reducible_losses_async(cand_indices, cand_paths, cand_labels):
        """Compute reducible losses asynchronously using inference model"""
        try:
            # Make copies to avoid race conditions
            cand_indices = cand_indices.copy()
            cand_paths = cand_paths.copy()
            cand_labels = cand_labels.copy()
            
            # Validate inputs
            if len(cand_indices) == 0:
                return np.array([]), np.array([])
            
            # Load candidate images in batch 
            with tf.device(device_name):
                cand_images = load_and_preprocess_image_batch(cand_paths)
                
                # Get predictions from inference model
                predictions = inference_model(cand_images, training=False)
                
                # Compute current losses - need per-sample losses for prioritized training
                if loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 'keypoint_loss_with_visibility', 'keypoint_mse_with_visibility']:
                    current_losses = loss_fn(cand_labels, predictions, reduction='none')
                else:
                    current_losses = loss_fn(cand_labels, predictions)
                
                # Ensure we have per-sample losses, not a scalar
                if tf.rank(current_losses) == 0:  # Scalar
                    # This shouldn't happen if reduction is set correctly, but handle it anyway
                    current_losses = tf.repeat(current_losses, len(cand_indices))
                
                # Get IL losses for these indices - validate indices first
                valid_indices = cand_indices[cand_indices < len(il_losses)]
                if len(valid_indices) == 0:
                    raise ValueError("No valid indices found")
                
                il_losses_batch = tf.gather(tf.constant(il_losses, dtype=tf.float32), valid_indices)
                
                # Calculate reducible losses (only for valid indices)
                reducible_losses = current_losses[:len(valid_indices)] - il_losses_batch
                
                # Select top-k based on reducible losses
                k = min(train_batch_size, len(valid_indices))
                top_k_values, top_k_indices = tf.nn.top_k(reducible_losses, k=k)
                selected_local_indices = top_k_indices.numpy()
                selected_global_indices = valid_indices[selected_local_indices]
            
            return selected_global_indices, reducible_losses.numpy()
            
        except Exception as e:
            if verbose:
                print(f"[Async-PT] Error in async computation: {e}")
            # Fallback to random selection
            safe_size = min(train_batch_size, len(cand_indices))
            if safe_size > 0:
                selected_idx = np.random.choice(len(cand_indices), size=safe_size, replace=False)
                return cand_indices[selected_idx], np.zeros(len(cand_indices))
            else:
                return np.array([]), np.array([])
    
    def sync_inference_model_weights():
        """Synchronize inference model with target model weights"""
        try:
            with tf.device(device_name):
                inference_model.set_weights(model.get_weights())
            if verbose:
                print(f"[Async-PT] Synced inference model weights at batch {batch_counter['count']}")
        except Exception as e:
            if verbose:
                print(f"[Async-PT] Warning: Failed to sync weights: {e}")
    
    def epoch_generator():
        """Generate batches for one epoch with async prioritization"""
        # Create indices for all samples
        all_indices = np.arange(len(image_paths))
        np.random.shuffle(all_indices)
        
        # Pre-submit first few batches for async processing
        future_batches = []
        
        for i in range(0, len(all_indices), cand_batch_size):
            cand_indices = all_indices[i:i + cand_batch_size]
            
            if len(cand_indices) < cand_batch_size:
                continue
            
            # Get paths and labels for candidates
            cand_paths = image_paths[cand_indices]
            cand_labels = labels[cand_indices]
            
            # If we have capacity, submit async computation
            if len(future_batches) < 2:  # Keep 2 batches in flight
                future = executor.submit(
                    compute_reducible_losses_async, 
                    cand_indices, cand_paths, cand_labels
                )
                future_batches.append((future, cand_indices))
                continue
            
            # Get result from oldest submitted batch
            future, batch_indices = future_batches.pop(0)
            try:
                selected_indices, reducible_losses = future.result(timeout=30.0)
                
                # Log progress
                batch_counter['count'] += 1
                if verbose and batch_counter['count'] % 50 == 0:
                    print(f"[Async-PT] Batch {batch_counter['count']}: "
                          f"Reducible losses - mean: {reducible_losses.mean():.4f}, "
                          f"max: {reducible_losses.max():.4f}")
                
                # Sync inference model weights periodically
                if batch_counter['count'] - batch_counter['last_sync'] >= update_freq:
                    sync_inference_model_weights()
                    batch_counter['last_sync'] = batch_counter['count']
                
                yield selected_indices
                
                # Submit new batch for async processing
                future = executor.submit(
                    compute_reducible_losses_async, 
                    cand_indices, cand_paths, cand_labels
                )
                future_batches.append((future, cand_indices))
                
            except concurrent.futures.TimeoutError:
                if verbose:
                    print(f"[Async-PT] Timeout waiting for batch result, using random selection")
                # Fallback to random selection
                selected_idx = np.random.choice(len(batch_indices), size=train_batch_size, replace=False)
                yield batch_indices[selected_idx]
        
        # Process remaining batches in queue
        while future_batches:
            future, batch_indices = future_batches.pop(0)
            try:
                selected_indices, _ = future.result(timeout=10.0)
                yield selected_indices
            except concurrent.futures.TimeoutError:
                selected_idx = np.random.choice(len(batch_indices), size=train_batch_size, replace=False)
                yield batch_indices[selected_idx]
    
    # Create dataset from generator
    indices_dataset = tf.data.Dataset.from_generator(
        epoch_generator,
        output_signature=tf.TensorSpec(shape=(train_batch_size,), dtype=tf.int64)
    ).repeat()
    
    # Gather paths and labels based on selected indices
    dataset = indices_dataset.map(
        lambda indices: (
            tf.gather(tf.constant(image_paths), indices),
            tf.gather(tf.constant(labels, dtype=tf.float32), indices)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    def load_batch_images_py(paths):
        """Load a batch of images using Python function (avoids tf.map_fn issues)"""
        import cv2
        
        paths_np = paths.numpy()
        images = []
        
        for path in paths_np:
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            try:
                # Try OpenCV first
                img = cv2.imread(str(path))
                if img is None:
                    # Fallback to TF
                    img_data = tf.io.read_file(str(path))
                    img = tf.image.decode_jpeg(img_data, channels=3).numpy()
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize and normalize
                img = cv2.resize(img, input_shape[:2])
                img = img.astype(np.float32) / 255.0
                images.append(img)
                
            except Exception as e:
                # Create dummy image if loading fails
                dummy_img = np.zeros(input_shape, dtype=np.float32)
                images.append(dummy_img)
        
        return np.stack(images)
    
    # Load images using tf.py_function (more stable than tf.map_fn)
    dataset = dataset.map(
        lambda paths, labels: (
            tf.py_function(
                func=load_batch_images_py,
                inp=[paths],
                Tout=tf.float32
            ),
            labels
        ),
        num_parallel_calls=1
    )
    
    # Set shape for the images tensor
    dataset = dataset.map(
        lambda images, labels: (
            tf.ensure_shape(images, (train_batch_size,) + input_shape),
            labels
        )
    )
    
    # Apply data augmentation if specified
    if augmentation_layers:
        def apply_augmentation(images, labels):
            augmented_images = images
            for aug_layer in augmentation_layers:
                augmented_images = aug_layer(augmented_images, training=True)
            return augmented_images, labels
        
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add prefetching for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# Backward compatibility alias - both now point to the efficient implementation
create_tf_data_prioritized_dataset_approx = create_tf_data_prioritized_dataset
create_tf_data_prioritized_dataset_candidate_only = create_tf_data_prioritized_dataset


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
        """Optimized image loading function with aspect ratio preservation"""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        # Use aspect ratio preserving resize to match validation preprocessing
        image = tf.image.resize_with_pad(
            image, 
            input_shape[0], 
            input_shape[1],
            method=tf.image.ResizeMethod.BILINEAR
        )
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
    
    def load_batch_images_py(paths):
        """Load a batch of images using Python function (avoids tf.map_fn TensorArray issues)"""
        import cv2
        
        paths_np = paths.numpy()
        images = []
        
        for path in paths_np:
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            try:
                # Try OpenCV first
                img = cv2.imread(str(path))
                if img is None:
                    # Fallback to TF
                    img_data = tf.io.read_file(str(path))
                    img = tf.image.decode_jpeg(img_data, channels=3).numpy()
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize and normalize
                img = cv2.resize(img, input_shape[:2])
                img = img.astype(np.float32) / 255.0
                images.append(img)
                
            except Exception as e:
                # Create dummy image if loading fails
                dummy_img = np.zeros(input_shape, dtype=np.float32)
                images.append(dummy_img)
        
        return np.stack(images)
    
    # Apply image loading using tf.py_function (more stable than tf.map_fn)
    dataset = dataset.map(
        lambda paths, labels: (
            tf.py_function(
                func=load_batch_images_py,
                inp=[paths],
                Tout=tf.float32
            ),
            labels
        ),
        num_parallel_calls=1  # Sequential to avoid race conditions
    )
    
    # Set shape for the images tensor
    dataset = dataset.map(
        lambda images, labels: (
            tf.ensure_shape(images, (train_batch_size,) + input_shape),
            labels
        )
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


def compute_model_losses(model, dataset_paths, labels, config, batch_size=128, verbose=True):
    """
    Compute per-sample losses for any model on a dataset.
    
    This function computes losses on-the-fly without modifying the model.
    It follows the same pattern as compute_il_losses_streaming but works
    with any model, not just the holdout model.
    
    Args:
        model: Keras model (already compiled with a loss function)
        dataset_paths: Array of file paths for the dataset
        labels: Array of labels corresponding to the dataset
        config: Configuration dictionary containing 'input_shape' and 'loss' info
        batch_size: Batch size for processing (default: 128)
        verbose: Whether to print progress (default: True)
        
    Returns:
        Dictionary mapping sample indices to their loss values
    """
    if verbose:
        print(f"Computing losses for {len(dataset_paths)} samples...")
    
    # Create tf.data dataset for efficient batch processing
    from data_utils import create_validation_dataset
    dataset = create_validation_dataset(dataset_paths, labels, batch_size, config['input_shape'])
    
    # Get loss function from config
    loss_name = config['loss']
    if isinstance(loss_name, str) and loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 
                                                     'smooth_l1', 'keypoint_loss_with_visibility', 
                                                     'keypoint_mse_with_visibility']:
        from keypoint_losses import create_keypoint_loss
        loss_kwargs = config.get('loss_kwargs', {})
        loss_fn = create_keypoint_loss(loss_name, **loss_kwargs)
    else:
        loss_fn = tf.keras.losses.get(loss_name)
        # Ensure we get per-sample losses, not reduced
        loss_fn.reduction = tf.keras.losses.Reduction.NONE
    
    # Process all batches efficiently
    all_losses = []
    num_batches = (len(dataset_paths) + batch_size - 1) // batch_size
    
    for batch_idx, (batch_images, batch_labels) in enumerate(dataset):
        if verbose and batch_idx % 50 == 0:
            print(f"  Processing batch {batch_idx+1}/{num_batches}")
        
        # Get predictions
        predictions = model(batch_images, training=False)
        
        # Compute per-sample losses
        if loss_name in ['smooth_l1_with_visibility', 'mse_with_visibility', 
                         'keypoint_loss_with_visibility', 'keypoint_mse_with_visibility']:
            # These loss functions already support per-sample computation
            batch_losses = loss_fn(batch_labels, predictions, reduction='none').numpy()
        else:
            # Standard Keras losses
            batch_losses = loss_fn(batch_labels, predictions).numpy()
        
        all_losses.extend(batch_losses)
    
    # Create dictionary mapping sample index to loss
    loss_dict = {i: float(loss) for i, loss in enumerate(all_losses)}
    
    if verbose:
        print(f"Losses computed for {len(loss_dict)} samples (loss: {loss_name})")
        print(f"  Min loss: {min(all_losses):.4f}")
        print(f"  Max loss: {max(all_losses):.4f}")
        print(f"  Mean loss: {sum(all_losses)/len(all_losses):.4f}")
        print(f"  Std loss: {np.std(all_losses):.4f}")
    
    return loss_dict


def compute_model_losses_batch(model, batch_images, batch_labels, loss_fn=None):
    """
    Compute per-sample losses for a single batch.
    
    This is a lighter-weight version for computing losses on a single batch,
    useful during training or for small datasets.
    
    Args:
        model: Keras model
        batch_images: Batch of images (numpy array or tensor)
        batch_labels: Batch of labels (numpy array or tensor)
        loss_fn: Loss function to use (if None, uses model's compiled loss)
        
    Returns:
        Array of per-sample losses
    """
    # Get predictions
    predictions = model(batch_images, training=False)
    
    # Use provided loss function or extract from model
    if loss_fn is None:
        # Try to get the loss function from the compiled model
        if hasattr(model, 'compiled_loss') and model.compiled_loss._losses:
            loss_fn = model.compiled_loss._losses[0]
        else:
            raise ValueError("No loss function provided and model doesn't have a compiled loss")
    
    # Ensure loss function returns per-sample losses
    if hasattr(loss_fn, 'reduction'):
        original_reduction = loss_fn.reduction
        loss_fn.reduction = tf.keras.losses.Reduction.NONE
    
    # Compute losses
    batch_losses = loss_fn(batch_labels, predictions)
    
    # Restore original reduction if we modified it
    if hasattr(loss_fn, 'reduction'):
        loss_fn.reduction = original_reduction
    
    return batch_losses.numpy() if hasattr(batch_losses, 'numpy') else batch_losses