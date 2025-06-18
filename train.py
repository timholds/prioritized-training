#!/usr/bin/env python3

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import wandb

# Configure GPU for optimal performance
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

from data import get_cocoreg_paths, get_cocokp_paths, generate_train_test_split
from data_model_map import data_model_map
from models import compile_model, create_resnet_lr_schedule, create_cosine_lr_schedule
from callbacks import compute_il_losses, compute_il_losses_streaming, create_tf_data_prioritized_dataset, create_tf_data_prioritized_dataset_async, create_tf_data_random_dataset, compute_model_losses, compute_model_losses_batch
from data_utils import create_validation_dataset, create_training_dataset_for_holdout, evaluate_on_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train prioritized training experiments')
    parser.add_argument('--dataset', type=str, default='qmnist', 
                       choices=['qmnist', 'cocoreg', 'cocokp'],
                       help='Dataset to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], 
                       help='Random seeds to use (can specify multiple)')
    parser.add_argument('--subsample_rate', type=float, nargs='+', default=[0.1], 
                       help='Subsample rate(s) (batch_size / big_batch_size). Can specify one or more.')
    parser.add_argument('--pt_verbose', action='store_true',
                       help='Print debug info for prioritized training')
    parser.add_argument('--async_pt', action='store_true',
                       help='Use async prioritized training for ~50% speedup')
    parser.add_argument('--async_update_freq', type=int, default=5,
                       help='How often to sync inference model weights in async mode (every N batches)')
    parser.add_argument('--async_cpu_inference', action='store_true',
                       help='Use CPU for inference model in async mode (saves GPU memory)')
    # Note: Prioritized training is now efficient by default (candidate-only loss computation)
    return parser.parse_args()

def create_lr_schedule_from_config(config, steps_per_epoch, total_epochs):
    """
    Create learning rate schedule based on config parameters.
    
    Args:
        config: Configuration dictionary
        steps_per_epoch: Steps per epoch for converting epoch-based schedules to step-based
        total_epochs: Total training epochs
    
    Returns:
        Learning rate schedule or None if using fixed learning rate
    """
    lr_schedule_type = config.get('lr_schedule_type', None)
    initial_lr = config.get('learning_rate', 0.001)
    
    if lr_schedule_type == 'step':
        # Step decay schedule
        decay_epochs = config.get('lr_decay_epochs', [10, 20])
        decay_factor = config.get('lr_decay_factor', 0.1)
        
        # Convert epoch boundaries to step boundaries
        step_boundaries = [epoch * steps_per_epoch for epoch in decay_epochs]
        values = [initial_lr * (decay_factor ** i) for i in range(len(decay_epochs) + 1)]
        
        return keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=step_boundaries,
            values=values
        )
    elif lr_schedule_type == 'cosine':
        # Cosine annealing schedule
        warmup_epochs = config.get('lr_warmup_epochs', 0)
        total_steps = total_epochs * steps_per_epoch
        
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps
        )
    else:
        # No schedule, use fixed learning rate
        return None

def get_or_train_holdout_model(dataset, config, holdout_train_dataset, holdout_eval_dataset, val_dataset, holdout_steps_per_epoch):
    """
    Get cached holdout model or train a new one if it doesn't exist.
    Holdout models are cached by dataset and holdout_epochs (reused across subsample rates).
    
    Args:
        dataset: Dataset name string
        config: Configuration dictionary
        holdout_train_dataset: tf.data.Dataset for training
        holdout_eval_dataset: tf.data.Dataset for holdout evaluation
        val_dataset: tf.data.Dataset for validation
        holdout_steps_per_epoch: Number of steps per epoch for holdout training
    """
    os.makedirs('models', exist_ok=True)
    # Include holdout model type in path to avoid conflicts
    holdout_model_class = config.get('holdout_model', config['model'])
    holdout_model_name = holdout_model_class.__name__
    holdout_model_path = f"models/{dataset}_holdout_{holdout_model_name}_{config['holdout_epochs']}epochs.h5"
    
    if os.path.exists(holdout_model_path):
        print(f"Loading cached holdout model from {holdout_model_path}")
        # Import model classes for custom object scope
        from models import ResNet18, ResNet18Model, MLPModel, ConvModel, BasicBlock, KeypointResNet18
        from keypoint_losses import keypoint_loss_with_visibility, keypoint_mse_with_visibility, smooth_l1_loss
        from keypoint_metrics import OKSMetric, PCKMetric
        
        # Import the loss creation function
        from keypoint_losses import create_keypoint_loss
        
        # Create the loss function used during training
        loss_fn = create_keypoint_loss(config['loss'], **config.get('loss_kwargs', {}))
        
        custom_objects = {
            'ResNet18': ResNet18,
            'ResNet18Model': ResNet18Model,
            'MLPModel': MLPModel,
            'ConvModel': ConvModel,
            'BasicBlock': BasicBlock,
            'KeypointResNet18': KeypointResNet18,
            'keypoint_loss_with_visibility': keypoint_loss_with_visibility,
            'keypoint_mse_with_visibility': keypoint_mse_with_visibility,
            'smooth_l1_loss': smooth_l1_loss,
            'OKSMetric': OKSMetric,
            'PCKMetric': PCKMetric,
            'loss_fn': loss_fn  # Add the actual loss function
        }
        with keras.utils.custom_object_scope(custom_objects):
            holdout_model = keras.models.load_model(holdout_model_path, compile=False)
            # Recompile with the correct loss function
            from models import compile_model
            compile_model(holdout_model, 
                         loss=config['loss'],
                         loss_kwargs=config.get('loss_kwargs', {}),
                         metrics=config['metrics'],
                         learning_rate=config['learning_rate'],
                         optimizer=config.get('optimizer', 'sgd'),
                         momentum=config.get('momentum', 0.9),
                         weight_decay=config.get('weight_decay', 5e-4))
        eval_results = holdout_model.evaluate(val_dataset, verbose=0)
        holdout_test_loss = eval_results[0]  # Always the first value (loss)
        holdout_test_acc = eval_results[1]   # Main metric (mse for regression, accuracy for classification)
        print(f"Cached holdout model test metric: {holdout_test_acc:.4f}")
    else:
        print(f"Training new holdout model...")
        
        # Create fresh holdout model - use separate holdout_model if specified
        holdout_model_class = config.get('holdout_model', config['model'])
        # Handle different parameter names for different models
        if holdout_model_class.__name__ == 'KeypointResNet18':
            holdout_model = holdout_model_class(
                num_outputs=config['n_outputs'], 
                input_shape=config['input_shape']
            )
        elif holdout_model_class.__name__ == 'ResNet18':
            holdout_model = holdout_model_class(
                num_outputs=config.get('n_outputs', config.get('n_classes')), 
                input_shape=config['input_shape'],
                output_activation=config.get('output_activation')
            )
        else:
            # Handle regression models that use num_outputs instead of num_classes
            if 'n_outputs' in config:
                holdout_model = holdout_model_class(
                    num_outputs=config['n_outputs'], 
                    input_shape=config['input_shape']
                )
            else:
                holdout_model = holdout_model_class(
                    num_classes=config['n_classes'], 
                    input_shape=config['input_shape']
                ).create_model()
        
        # Create learning rate schedule for holdout model
        holdout_lr_schedule = create_lr_schedule_from_config(config, holdout_steps_per_epoch, config['holdout_epochs'])
        
        holdout_model = compile_model(
            holdout_model, 
            loss=config['loss'], 
            metrics=config['metrics'],
            learning_rate=config.get('learning_rate', 0.001),
            loss_kwargs=config.get('loss_kwargs', None),
            optimizer=config.get('optimizer', 'sgd'),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4),
            lr_schedule=holdout_lr_schedule
        )
        print(f"Compiling holdout model with loss: {config['loss']} and metrics: {config['metrics']}")
        holdout_model.summary()
        
        # Simple W&B callback for holdout model
        wandb_callback = wandb.keras.WandbCallback(
            save_model=False,
            log_batch_frequency=10,
            log_best_prefix="holdout_best_",
            validation_data=val_dataset,
            predictions=10,
            generator=None,
            save_graph=False  # Disable graph logging to avoid compatibility issues
        )
        
        # Train holdout model using tf.data pipeline with W&B logging
        holdout_history = holdout_model.fit(
            holdout_train_dataset,
            steps_per_epoch=holdout_steps_per_epoch,
            epochs=config['holdout_epochs'],
            validation_data=val_dataset,
            callbacks=[wandb_callback],
            verbose=1
        )
        
        # Evaluate and save
        eval_results = holdout_model.evaluate(val_dataset, verbose=0)
        holdout_test_loss = eval_results[0]
        holdout_test_acc = eval_results[1]
        print(f"New holdout model test metric: {holdout_test_acc:.4f}")
        
        # Save model for reuse
        holdout_model.save(holdout_model_path)
        print(f"Saved holdout model to {holdout_model_path}")
    
    return holdout_model, holdout_test_acc, holdout_model_path

def train_model_with_tracking(model, dataset, val_dataset, 
                             config, steps_per_epoch, epochs, train_batch_size, model_type,
                             global_step_offset=0):
    """
    Train a model using a data generator and track step-wise progress for plotting.
    
    Args:
        model: Model to train
        dataset: Training tf.data.Dataset
        val_dataset: Validation tf.data.Dataset
        config: Configuration dictionary
        steps_per_epoch: Steps per epoch
        epochs: Number of epochs
        train_batch_size: Training batch size
        model_type: Model type string (for logging)
        global_step_offset: Offset for global step counter (for tracking across models)
    """
    # Track progress manually since we need step counts
    step_history = {
        'steps': [],
        'val_accuracy': [],
        'val_loss': []
    }
    
    class StepTrackingCallback(keras.callbacks.Callback):
        def __init__(self, val_dataset, steps_per_epoch, model_type, config, global_step_offset, total_epochs):
            self.val_dataset = val_dataset
            self.steps_per_epoch = steps_per_epoch
            self.current_step = 0
            self.global_step = global_step_offset
            self.model_type = model_type
            self.config = config
            self.batch_count = 0
            self.start_time = None
            self.total_epochs = total_epochs
            
        def on_train_begin(self, logs=None):
            self.start_time = tf.timestamp().numpy()
            
        def on_batch_end(self, batch, logs=None):
            self.batch_count += 1
            self.current_step += 1
            self.global_step += 1
            
            # Log batch-level metrics every 10 batches
            if self.batch_count % 10 == 0:
                elapsed_time = tf.timestamp().numpy() - self.start_time
                steps_per_second = self.global_step / elapsed_time if elapsed_time > 0 else 0
                
                batch_log = {
                    "batch_loss": logs.get('loss', 0),
                    "current_step": self.current_step,
                    "global_step": self.global_step,
                    "steps_per_second": steps_per_second,
                    "model_type": self.model_type
                }
                wandb.log(batch_log)
            
        def on_epoch_end(self, epoch, logs=None):
            # Note: current_step and global_step are already incremented in on_batch_end
            # No need to increment again here
            # Evaluate on validation set
            eval_results = self.model.evaluate(self.val_dataset, verbose=0)
            val_loss = eval_results[0]
            
            # Debug logging
            print(f"[{self.model_type}] Epoch {epoch+1}/{self.total_epochs} completed - Step: {self.current_step}, Global step: {self.global_step}")
            
            # Log metrics based on task type
            wandb_log = {
                "val_loss": val_loss,
                "step": self.current_step,
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "total_training_steps": self.global_step,
                "model_type": self.model_type
            }
            
            # For regression tasks, handle different metric types
            if 'target_metric_value' in self.config:
                # Check if this is keypoint detection (has OKS metric)
                if self.config.get('loss') == 'smooth_l1_with_visibility' and len(eval_results) > 3:
                    # For keypoint: [loss, OKS, PCK, MSE]
                    val_oks = eval_results[1]  # OKS is primary metric
                    val_pck = eval_results[2] if len(eval_results) > 2 else None
                    val_mse = eval_results[3] if len(eval_results) > 3 else None
                    
                    wandb_log["val_oks"] = val_oks
                    wandb_log["val_pck"] = val_pck
                    if val_mse is not None:
                        wandb_log["val_mse"] = val_mse
                    
                    step_history['val_accuracy'].append(val_oks)  # Store OKS as primary metric
                else:
                    # Standard regression (MSE/MAE)
                    val_mse = eval_results[1] if len(eval_results) > 1 else val_loss
                    val_mae = eval_results[2] if len(eval_results) > 2 else None
                    
                    wandb_log["val_mse"] = val_mse
                    if val_mae is not None:
                        wandb_log["val_mae"] = val_mae
                    
                    step_history['val_accuracy'].append(val_mse)  # Store MSE as primary metric
            else:
                # For classification tasks, log accuracy
                val_acc = eval_results[1] if len(eval_results) > 1 else 0.0
                wandb_log["val_accuracy"] = val_acc
                step_history['val_accuracy'].append(val_acc)
            
            step_history['steps'].append(self.current_step)
            step_history['val_loss'].append(val_loss)
            
            # Add elapsed time and progress metrics
            elapsed_time = tf.timestamp().numpy() - self.start_time
            progress_percentage = (self.current_step / (self.steps_per_epoch * self.total_epochs)) * 100
            
            wandb_log.update({
                "elapsed_time_seconds": elapsed_time,
                "progress_percentage": progress_percentage
            })
            
            # Log to wandb
            wandb.log(wandb_log)
    
    step_tracker = StepTrackingCallback(val_dataset, steps_per_epoch, model_type, config, global_step_offset, epochs)
    callbacks = [step_tracker]
    
    # Train model using tf.data dataset
    history = model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    eval_results = model.evaluate(val_dataset, verbose=0)
    final_test_loss = eval_results[0]
    
    # Extract final metric based on task type
    if config.get('loss') == 'smooth_l1_with_visibility' and len(eval_results) > 3:
        # For keypoint detection, OKS is the primary metric
        final_test_acc = eval_results[1]  # OKS
    else:
        # For other tasks, use second metric (accuracy or MSE)
        final_test_acc = eval_results[1]
    
    # Calculate steps to target metric
    target_metric = config.get('target_accuracy') or config.get('target_metric_value')
    is_regression = 'target_metric_value' in config
    is_oks = config.get('loss') == 'smooth_l1_with_visibility'
    steps_to_target = None
    reached_target = False
    
    # val_accuracy contains the primary metric (OKS for keypoints, accuracy for classification, MSE for regression)
    metric_values = step_history['val_accuracy']
    
    for i, metric_val in enumerate(metric_values):
        # For OKS/accuracy: higher is better (>= target)
        # For MSE (non-OKS regression): lower is better (<= target)
        if is_oks or not is_regression:
            target_reached = metric_val >= target_metric
        else:
            target_reached = metric_val <= target_metric
            
        if target_reached:
            steps_to_target = step_history['steps'][i]
            reached_target = True
            break
    
    return {
        'final_test_acc': final_test_acc,
        'steps_to_target': steps_to_target,
        'reached_target': reached_target,
        'history_by_step': step_history,
        'total_steps': step_tracker.current_step
    }

def main():
    args = parse_args()
    
    # Initialize wandb with meaningful experiment name
    experiment_name = f"{args.dataset}_subsample_{'-'.join(map(str, args.subsample_rate))}_seeds_{'-'.join(map(str, args.seeds))}"
    wandb.init(
        project="prioritized-training",
        name=experiment_name,
        config={
            "dataset": args.dataset,
            "seeds": args.seeds,
            "subsample_rates": args.subsample_rate,
            "experiment_type": "rs_vs_pt_comparison",
            "pt_verbose": args.pt_verbose
        }
    )
    
    # Always treat subsample_rate as a list
    subsample_rates = args.subsample_rate
    
    print(f"Starting experiment with:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Subsample rates: {subsample_rates}")
    
    # Get dataset configuration
    config = data_model_map[args.dataset]
    
    
    # Load dataset paths for tf.data pipeline
    if args.dataset == 'cocoreg':
        print(f"Loading COCO regression dataset paths for tf.data pipeline...")
        image_paths, labels = get_cocoreg_paths()
        # Convert list to numpy array for compatibility with generate_train_test_split
        image_paths = np.array(image_paths)
    elif args.dataset == 'cocokp':
        print(f"Loading COCO keypoint dataset paths for tf.data pipeline...")
        # For quick testing with subsample < 0.01, limit max samples
        max_samples = None
        # Uncomment the following lines to limit samples for quick testing:
        # if min(subsample_rates) < 0.01:
        #     max_samples = 10000  # Load only 10k samples for very small subsample rates
        #     print(f"  Limiting to {max_samples} samples for quick testing")
        # Load with visibility information if specified in config
        include_visibility = config.get('include_visibility', False)
        image_paths, labels = get_cocokp_paths(max_samples=max_samples, include_visibility=include_visibility)
        # Convert list to numpy array for compatibility with generate_train_test_split
        image_paths = np.array(image_paths)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported in this optimized version")
    
    
    # Split data (once for all seeds)
    is_regression = 'target_metric_value' in config
    (train_paths, y_train), (val_paths, y_val), (holdout_paths, y_holdout) = generate_train_test_split(image_paths, labels, is_regression=is_regression, use_paths=True)
    
    
    # Create tf.data datasets for validation and holdout data (memory-efficient)
    print("Creating tf.data pipelines for validation and holdout data...")
    val_dataset = create_validation_dataset(val_paths, y_val, batch_size=config['batch_size'], input_shape=config['input_shape'])
    holdout_dataset_for_training = create_training_dataset_for_holdout(
        holdout_paths, y_holdout, 
        batch_size=config['batch_size'], 
        input_shape=config['input_shape'],
        epochs=config['holdout_epochs']
    )
    holdout_dataset_for_eval = create_validation_dataset(holdout_paths, y_holdout, batch_size=config['batch_size'], input_shape=config['input_shape'])
    
    # For compatibility with existing code that needs x_val in memory (will optimize later)
    # Load validation data in smaller batches for the step tracking callback
    print("Note: Validation data will be loaded on-demand during evaluation")
    
    
    batch_size = config['batch_size']
    holdout_epochs = config['holdout_epochs']
    
    # Create directories
    dataset_results_dir = f'results/{args.dataset}'
    os.makedirs(dataset_results_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Get or train holdout model (shared across all seeds and subsample rates)
    print(f"\n{'='*50}")
    print("GETTING/TRAINING HOLDOUT MODEL")
    print(f"{'='*50}")
    
    # Calculate steps per epoch for holdout training
    holdout_steps_per_epoch = len(holdout_paths) // batch_size
    
    holdout_model, holdout_test_acc, holdout_model_path = get_or_train_holdout_model(
        args.dataset, config, holdout_dataset_for_training, holdout_dataset_for_eval, 
        val_dataset, holdout_steps_per_epoch
    )
    
    # Log holdout model performance
    wandb.log({"holdout_test_accuracy": holdout_test_acc})
    
    # Compute IL losses using holdout model (shared across all seeds and subsample rates)
    print(f"\nComputing IL losses...")
    
    il_loss_dict = compute_il_losses_streaming(holdout_model, train_paths, y_train, config, batch_size=batch_size)
    
    
    # Train models for each subsample rate
    for subsample_rate in subsample_rates:
        print(f"\n{'='*70}")
        print(f"TRAINING SUBSAMPLE RATE: {subsample_rate}")
        print(f"{'='*70}")
        
        # Calculate training parameters based on subsample rate
        big_batch_size = int(batch_size / subsample_rate)
        dataset_size = len(train_paths)
        steps_per_epoch = dataset_size // big_batch_size
        epochs = int(config['epochs'] / subsample_rate)
        
        print(f"\nTraining configuration:")
        print(f"  Dataset size: {dataset_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Big batch size: {big_batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Holdout epochs: {holdout_epochs}")
        print(f"  Training epochs: {epochs}")
        print(f"  Subsample rate: {subsample_rate}")
        print(f"  Using tf.data pipeline")
        if args.async_pt:
            print(f"  ASYNC MODE ENABLED - Update frequency: every {args.async_update_freq} batches")
        else:
            print(f"  Using synchronous prioritized training")
        
        # Update wandb config for this subsample rate
        wandb.config.update({
            "current_subsample_rate": subsample_rate,
            "batch_size": batch_size,
            "big_batch_size": big_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "dataset_size": dataset_size,
            "async_pt": args.async_pt,
            "async_update_freq": args.async_update_freq if args.async_pt else None,
            "total_expected_steps": steps_per_epoch * epochs * 2 * len(args.seeds)  # RS + PT for each seed
        })
        
        # Initialize results structure for this subsample rate
        results = {
            'dataset': args.dataset,
            'subsample_rate': subsample_rate,
            'target_metric': config.get('target_accuracy') or config.get('target_metric_value'),
            'target_type': 'accuracy' if 'target_accuracy' in config else 'metric_value',
            'seeds': args.seeds,
            'batch_size': batch_size,
            'big_batch_size': big_batch_size,
            'steps_per_epoch': steps_per_epoch,
            'epochs': epochs,
            'holdout_epochs': holdout_epochs,
            'holdout_test_acc': holdout_test_acc,
            'holdout_model_path': holdout_model_path,
            'rs_results': {},
            'pt_results': {}
        }
        
        # Train RS and PT models for each seed
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"TRAINING MODELS FOR SEED {seed}")
            print(f"{'='*60}")
            
            # Set random seeds
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Train Random Sampling (RS) model
            print(f"\nTraining Random Sampling model (seed {seed})...")
            
            # Handle different parameter names for different models
            model_class = config['model']
            if model_class.__name__ == 'KeypointResNet18':
                rs_model = model_class(
                    num_outputs=config['n_outputs'], 
                    input_shape=config['input_shape']
                )
            elif model_class.__name__ == 'ResNet18':
                rs_model = model_class(
                    num_outputs=config.get('n_outputs', config.get('n_outputs')), 
                    input_shape=config['input_shape'],
                    output_activation=config.get('output_activation')
                )
            else:
                rs_model = model_class(
                    num_classes=config.get('n_classes', config.get('n_outputs')), 
                    input_shape=config['input_shape']
                ).create_model()
            
            # Create learning rate schedule for RS model
            rs_lr_schedule = create_lr_schedule_from_config(config, steps_per_epoch, epochs)
            
            rs_model = compile_model(
                rs_model, 
                loss=config['loss'], 
                metrics=config['metrics'],
                learning_rate=config.get('learning_rate', 0.001),
                loss_kwargs=config.get('loss_kwargs', None),
                optimizer=config.get('optimizer', 'sgd'),
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 5e-4),
                lr_schedule=rs_lr_schedule
            )
            print(f"Compiling RS model with loss: {config['loss']} and metrics: {config['metrics']}")
            rs_model.summary()

            
            # Create tf.data random dataset with augmentation
            # rs_dataset = create_tf_data_random_dataset(
            #     train_paths, y_train,
            #     train_batch_size=batch_size,
            #     cand_batch_size=big_batch_size,
            #     steps_per_epoch=steps_per_epoch,
            #     input_shape=config['input_shape'],
            #     augmentation_layers=config.get('augmentation', None)
            # )
            
            # rs_result = train_model_with_tracking(
            #     rs_model, rs_dataset, val_dataset,
            #     config, steps_per_epoch, epochs, batch_size, "rs",
            #     global_step_offset=0  # RS model starts at step 0
            # )
            
            # results['rs_results'][seed] = rs_result
            # print(f"RS model (seed {seed}) final accuracy: {rs_result['final_test_acc']:.4f}")
            
            # # Log final RS results for this seed
            # wandb.log({
            #     f"rs_final_accuracy_seed_{seed}": rs_result['final_test_acc'],
            #     f"rs_steps_to_target_seed_{seed}": rs_result['steps_to_target'],
            #     f"rs_reached_target_seed_{seed}": rs_result['reached_target'],
            #     f"rs_total_steps_seed_{seed}": rs_result['total_steps'],
            #     "current_seed": seed,
            #     "model_type": "random_sampling",
            #     "phase": "training_complete"
            # })
            
            # # Also log comparable metrics without prefix for better visualization
            # wandb.log({
            #     "final_accuracy": rs_result['final_test_acc'],
            #     "steps_to_target": rs_result['steps_to_target'],
            #     "reached_target": rs_result['reached_target'],
            #     "total_steps": rs_result['total_steps'],
            #     "model_type": "rs",
            #     "seed": seed,
            #     "phase": "final"
            # })
            
            # Train Prioritized Training (PT) model
            print(f"\nTraining Prioritized Training model (seed {seed})...")
            
            # Reset random seeds for PT model
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Handle different parameter names for different models
            model_class = config['model']
            if model_class.__name__ == 'KeypointResNet18':
                pt_model = model_class(
                    num_outputs=config['n_outputs'], 
                    input_shape=config['input_shape']
                )
            elif model_class.__name__ == 'ResNet18':
                pt_model = model_class(
                    num_outputs=config.get('n_outputs', config.get('n_classes')), 
                    input_shape=config['input_shape'],
                    output_activation=config.get('output_activation')
                )
            else:
                pt_model = model_class(
                    num_classes=config.get('n_classes', config.get('n_outputs')), 
                    input_shape=config['input_shape']
                ).create_model()
            
            # Create learning rate schedule for PT model
            pt_lr_schedule = create_lr_schedule_from_config(config, steps_per_epoch, epochs)
            
            pt_model = compile_model(
                pt_model, 
                loss=config['loss'], 
                metrics=config['metrics'],
                learning_rate=config.get('learning_rate', 0.001),
                loss_kwargs=config.get('loss_kwargs', None),
                optimizer=config.get('optimizer', 'sgd'),
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 5e-4),
                lr_schedule=pt_lr_schedule
            )
            print(f"Compiling PT model with loss: {config['loss']} and metrics: {config['metrics']}")
            pt_model.summary()
            
            # Create tf.data prioritized dataset with augmentation
            if args.async_pt:
                print(f"Creating ASYNC prioritized training dataset for ~50% speedup...")
                pt_dataset = create_tf_data_prioritized_dataset_async(
                    pt_model, train_paths, y_train,
                    il_loss_dict=il_loss_dict,
                    train_batch_size=batch_size,
                    cand_batch_size=big_batch_size,
                    steps_per_epoch=steps_per_epoch,
                    input_shape=config['input_shape'],
                    augmentation_layers=config.get('augmentation', None),
                    config=config,
                    verbose=args.pt_verbose,
                    update_freq=args.async_update_freq,
                    use_cpu_inference=args.async_cpu_inference
                )
            else:
                print(f"Creating EFFICIENT prioritized training dataset...")
                pt_dataset = create_tf_data_prioritized_dataset(
                    pt_model, train_paths, y_train,
                    il_loss_dict=il_loss_dict,
                    train_batch_size=batch_size,
                    cand_batch_size=big_batch_size,
                    steps_per_epoch=steps_per_epoch,
                    input_shape=config['input_shape'],
                    augmentation_layers=config.get('augmentation', None),
                    config=config,
                    verbose=args.pt_verbose
                )
            
            # Before training the PT model, set global_step_offset:
            global_step_offset = rs_result['total_steps'] if 'rs_result' in locals() else 0

            pt_result = train_model_with_tracking(
                pt_model, pt_dataset, val_dataset,
                config, steps_per_epoch, epochs, batch_size, "pt",
                global_step_offset=global_step_offset
            )
            
            results['pt_results'][seed] = pt_result
            print(f"PT model (seed {seed}) final accuracy: {pt_result['final_test_acc']:.4f}")
            
            # Log final PT results for this seed
            wandb.log({
                f"pt_final_accuracy_seed_{seed}": pt_result['final_test_acc'],
                f"pt_steps_to_target_seed_{seed}": pt_result['steps_to_target'],
                f"pt_reached_target_seed_{seed}": pt_result['reached_target'],
                f"pt_total_steps_seed_{seed}": pt_result['total_steps'],
                "current_seed": seed,
                "model_type": "prioritized_training",
                "phase": "training_complete",
                "total_experiment_steps": pt_result['total_steps'] + rs_result['total_steps']
            })
            
            # Also log comparable metrics without prefix for better visualization
            wandb.log({
                "final_accuracy": pt_result['final_test_acc'],
                "steps_to_target": pt_result['steps_to_target'],
                "reached_target": pt_result['reached_target'],
                "total_steps": pt_result['total_steps'],
                "model_type": "pt",
                "seed": seed,
                "phase": "final"
            })
    
        
        # Save consolidated results for this subsample rate
        result_file = f"{dataset_results_dir}/subsample_{subsample_rate}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy_types(results)
        
        with open(result_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to {result_file}")
        
        # Print summary statistics for this subsample rate
        print(f"\n{'='*50}")
        print(f"SUBSAMPLE RATE {subsample_rate} SUMMARY")
        print(f"{'='*50}")
        
        # Calculate average accuracies across seeds
        rs_accs = [results['rs_results'][seed]['final_test_acc'] for seed in args.seeds]
        pt_accs = [results['pt_results'][seed]['final_test_acc'] for seed in args.seeds]
        
        rs_mean, rs_std = np.mean(rs_accs), np.std(rs_accs)
        pt_mean, pt_std = np.mean(pt_accs), np.std(pt_accs)
        
        print(f"Random Sampling: {rs_mean:.4f} ± {rs_std:.4f}")
        print(f"Prioritized Training: {pt_mean:.4f} ± {pt_std:.4f}")
        
        if pt_mean > rs_mean:
            improvement = ((pt_mean - rs_mean) / rs_mean) * 100
            print(f"PT improvement over RS: +{improvement:.2f}%")
        else:
            decline = ((rs_mean - pt_mean) / rs_mean) * 100
            print(f"PT decline vs RS: -{decline:.2f}%")
        
        # Calculate total steps for this subsample rate
        total_steps_per_seed = steps_per_epoch * epochs * 2  # RS + PT
        total_steps_all_seeds = total_steps_per_seed * len(args.seeds)
        
        # Log summary statistics
        wandb.log({
            f"rs_mean_accuracy_subsample_{subsample_rate}": rs_mean,
            f"rs_std_accuracy_subsample_{subsample_rate}": rs_std,
            f"pt_mean_accuracy_subsample_{subsample_rate}": pt_mean,
            f"pt_std_accuracy_subsample_{subsample_rate}": pt_std,
            f"improvement_percentage_subsample_{subsample_rate}": improvement if pt_mean > rs_mean else -decline,
            "subsample_rate": subsample_rate,
            "phase": "subsample_complete",
            "total_steps_this_subsample": total_steps_all_seeds
        })
        
        # Calculate steps to target statistics
        rs_steps = [results['rs_results'][seed]['steps_to_target'] for seed in args.seeds 
                    if results['rs_results'][seed]['reached_target']]
        pt_steps = [results['pt_results'][seed]['steps_to_target'] for seed in args.seeds 
                    if results['pt_results'][seed]['reached_target']]
        
        if rs_steps and pt_steps:
            rs_steps_mean = np.mean(rs_steps)
            pt_steps_mean = np.mean(pt_steps)
            speedup = rs_steps_mean / pt_steps_mean if pt_steps_mean > 0 else 0
            print(f"Steps to target - RS: {rs_steps_mean:.0f}, PT: {pt_steps_mean:.0f}")
            print(f"PT speedup: {speedup:.2f}x")
            
            # Log speedup statistics
            wandb.log({
                f"rs_mean_steps_to_target_subsample_{subsample_rate}": rs_steps_mean,
                f"pt_mean_steps_to_target_subsample_{subsample_rate}": pt_steps_mean,
                f"speedup_subsample_{subsample_rate}": speedup,
                "subsample_rate": subsample_rate
            })
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Holdout model accuracy: {holdout_test_acc:.4f}")
    print(f"Results saved in: {dataset_results_dir}/")
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()