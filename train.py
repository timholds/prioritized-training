#!/usr/bin/env python3

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

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

from data import get_cocoreg_paths, generate_train_test_split
from data_model_map import data_model_map
from models import compile_model
from callbacks import compute_il_losses, create_tf_data_prioritized_dataset, create_tf_data_random_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train prioritized training experiments')
    parser.add_argument('--dataset', type=str, default='qmnist', 
                       choices=['qmnist', 'cocoreg', 'cocokp'],
                       help='Dataset to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], 
                       help='Random seeds to use (can specify multiple)')
    parser.add_argument('--subsample_rate', type=float, nargs='+', default=[0.1], 
                       help='Subsample rate(s) (batch_size / big_batch_size). Can specify one or more.')
    return parser.parse_args()

def get_or_train_holdout_model(dataset, config, x_holdout, y_holdout, x_val, y_val):
    """
    Get cached holdout model or train a new one if it doesn't exist.
    Holdout models are cached by dataset and holdout_epochs (reused across subsample rates).
    """
    os.makedirs('models', exist_ok=True)
    # Include holdout model type in path to avoid conflicts
    holdout_model_class = config.get('holdout_model', config['model'])
    holdout_model_name = holdout_model_class.__name__
    holdout_model_path = f"models/{dataset}_holdout_{holdout_model_name}_{config['holdout_epochs']}epochs.h5"
    
    if os.path.exists(holdout_model_path):
        print(f"Loading cached holdout model from {holdout_model_path}")
        holdout_model = keras.models.load_model(holdout_model_path)
        eval_results = holdout_model.evaluate(x_val, y_val, verbose=0)
        holdout_test_loss = eval_results[0]  # Always the first value (loss)
        holdout_test_acc = eval_results[1]   # Main metric (mse for regression, accuracy for classification)
        print(f"Cached holdout model test metric: {holdout_test_acc:.4f}")
    else:
        print(f"Training new holdout model...")
        
        # Create fresh holdout model - use separate holdout_model if specified
        holdout_model_class = config.get('holdout_model', config['model'])
        # Handle different parameter names for different models
        if holdout_model_class.__name__ == 'ResNet18':
            holdout_model = holdout_model_class(
                num_outputs=config.get('n_outputs', config.get('n_classes')), 
                input_shape=config['input_shape']
            )
        else:
            holdout_model = holdout_model_class(
                num_classes=config.get('n_classes', config.get('n_outputs')), 
                input_shape=config['input_shape']
            ).create_model()
        
        holdout_model = compile_model(
            holdout_model, 
            loss=config['loss'], 
            metrics=config['metrics']
        )
        
        # Train holdout model on holdout data
        holdout_history = holdout_model.fit(
            x_holdout, y_holdout,
            batch_size=config['batch_size'],
            epochs=config['holdout_epochs'],
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Evaluate and save
        eval_results = holdout_model.evaluate(x_val, y_val, verbose=0)
        holdout_test_loss = eval_results[0]
        holdout_test_acc = eval_results[1]
        print(f"New holdout model test metric: {holdout_test_acc:.4f}")
        
        # Save model for reuse
        holdout_model.save(holdout_model_path)
        print(f"Saved holdout model to {holdout_model_path}")
    
    return holdout_model, holdout_test_acc, holdout_model_path

def train_model_with_tracking(model, dataset, x_val, y_val, 
                             config, steps_per_epoch, epochs, train_batch_size):
    """
    Train a model using a data generator and track step-wise progress for plotting.
    """
    # Track progress manually since we need step counts
    step_history = {
        'steps': [],
        'val_accuracy': [],
        'val_loss': []
    }
    
    class StepTrackingCallback(keras.callbacks.Callback):
        def __init__(self, x_val, y_val, steps_per_epoch):
            self.x_val = x_val
            self.y_val = y_val
            self.steps_per_epoch = steps_per_epoch
            self.current_step = 0
            
        def on_epoch_end(self, epoch, logs=None):
            self.current_step += self.steps_per_epoch
            # Evaluate on validation set
            eval_results = self.model.evaluate(self.x_val, self.y_val, verbose=0)
            val_loss = eval_results[0]
            val_acc = eval_results[1]
            step_history['steps'].append(self.current_step)
            step_history['val_accuracy'].append(val_acc)
            step_history['val_loss'].append(val_loss)
    
    step_tracker = StepTrackingCallback(x_val, y_val, steps_per_epoch)
    callbacks = [step_tracker]
    
    # Train model using tf.data dataset
    history = model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    eval_results = model.evaluate(x_val, y_val, verbose=0)
    final_test_loss = eval_results[0]
    final_test_acc = eval_results[1]
    
    # Calculate steps to target metric (accuracy for classification, metric_value for regression)
    target_metric = config.get('target_accuracy') or config.get('target_metric_value')
    is_regression = 'target_metric_value' in config
    steps_to_target = None
    reached_target = False
    
    metric_values = step_history['val_loss'] if is_regression else step_history['val_accuracy']
    
    for i, metric_val in enumerate(metric_values):
        # For regression, we want loss <= target_metric_value. For classification, accuracy >= target_accuracy
        target_reached = (metric_val <= target_metric) if is_regression else (metric_val >= target_metric)
        if target_reached:
            steps_to_target = step_history['steps'][i]
            reached_target = True
            break
    
    return {
        'final_test_acc': final_test_acc,
        'steps_to_target': steps_to_target,
        'reached_target': reached_target,
        'history_by_step': step_history
    }

def main():
    args = parse_args()
    
    # Always treat subsample_rate as a list
    subsample_rates = args.subsample_rate
    
    print(f"Starting experiment with:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Subsample rates: {subsample_rates}")
    
    # Get dataset configuration
    config = data_model_map[args.dataset]
    
    # Load COCO dataset paths for tf.data pipeline
    if args.dataset != 'cocoreg':
        raise NotImplementedError(f"Only cocoreg dataset supported in this optimized version")
    
    print(f"Loading COCO dataset paths for tf.data pipeline...")
    image_paths, labels = get_cocoreg_paths()
    
    # Split data (once for all seeds)
    is_regression = 'target_metric_value' in config
    (train_paths, y_train), (val_paths, y_val), (holdout_paths, y_holdout) = generate_train_test_split(image_paths, labels, is_regression=is_regression, use_paths=True)
    
    # Load holdout and validation images into memory for holdout model training
    print("Loading holdout and validation images for holdout model training...")
    from PIL import Image
    
    def load_images_from_paths(paths):
        images = []
        target_size = config['input_shape'][:2]  # (H, W)
        for i, path in enumerate(paths):
            if i % 100 == 0:
                print(f"  Loaded {i}/{len(paths)} images")
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(target_size)
                images.append(np.array(img))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Create dummy image
                images.append(np.zeros(target_size + (3,), dtype=np.uint8))
        return np.array(images).astype('float32') / 255.0
    
    x_val = load_images_from_paths(val_paths)
    x_holdout = load_images_from_paths(holdout_paths)
    print("Holdout and validation images loaded.")
    
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
    
    holdout_model, holdout_test_acc, holdout_model_path = get_or_train_holdout_model(
        args.dataset, config, x_holdout, y_holdout, x_val, y_val
    )
    
    # Compute IL losses using holdout model (shared across all seeds and subsample rates)
    print(f"\nComputing IL losses...")
    print("Loading training images for IL loss computation...")
    x_train = load_images_from_paths(train_paths)
    print("Training images loaded for IL loss computation.")
    
    il_loss_dict = compute_il_losses(holdout_model, x_train, y_train, batch_size=batch_size)
    
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
            if model_class.__name__ == 'ResNet18':
                rs_model = model_class(
                    num_outputs=config.get('n_outputs', config.get('n_classes')), 
                    input_shape=config['input_shape']
                )
            else:
                rs_model = model_class(
                    num_classes=config.get('n_classes', config.get('n_outputs')), 
                    input_shape=config['input_shape']
                ).create_model()
            
            rs_model = compile_model(
                rs_model, 
                loss=config['loss'], 
                metrics=config['metrics']
            )
            
            # Create tf.data random dataset
            rs_dataset = create_tf_data_random_dataset(
                train_paths, y_train,
                train_batch_size=batch_size,
                cand_batch_size=big_batch_size,
                steps_per_epoch=steps_per_epoch,
                input_shape=config['input_shape']
            )
            
            rs_result = train_model_with_tracking(
                rs_model, rs_dataset, x_val, y_val,
                config, steps_per_epoch, epochs, batch_size
            )
            
            results['rs_results'][seed] = rs_result
            print(f"RS model (seed {seed}) final accuracy: {rs_result['final_test_acc']:.4f}")
            
            # Train Prioritized Training (PT) model
            print(f"\nTraining Prioritized Training model (seed {seed})...")
            
            # Reset random seeds for PT model
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Handle different parameter names for different models
            model_class = config['model']
            if model_class.__name__ == 'ResNet18':
                pt_model = model_class(
                    num_outputs=config.get('n_outputs', config.get('n_classes')), 
                    input_shape=config['input_shape']
                )
            else:
                pt_model = model_class(
                    num_classes=config.get('n_classes', config.get('n_outputs')), 
                    input_shape=config['input_shape']
                ).create_model()
            
            pt_model = compile_model(
                pt_model, 
                loss=config['loss'], 
                metrics=config['metrics']
            )
            
            # Create tf.data prioritized dataset
            pt_dataset = create_tf_data_prioritized_dataset(
                train_paths, y_train,
                il_loss_dict=il_loss_dict,
                train_batch_size=batch_size,
                cand_batch_size=big_batch_size,
                steps_per_epoch=steps_per_epoch,
                input_shape=config['input_shape']
            )
            
            pt_result = train_model_with_tracking(
                pt_model, pt_dataset, x_val, y_val,
                config, steps_per_epoch, epochs, batch_size
            )
            
            results['pt_results'][seed] = pt_result
            print(f"PT model (seed {seed}) final accuracy: {pt_result['final_test_acc']:.4f}")
    
        
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
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Holdout model accuracy: {holdout_test_acc:.4f}")
    print(f"Results saved in: {dataset_results_dir}/")

if __name__ == '__main__':
    main()