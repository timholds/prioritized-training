from tensorflow import keras
import random
import numpy as np
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