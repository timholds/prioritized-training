#!/usr/bin/env python3

import os
import json
import glob
import argparse
import numpy as np

def aggregate_dataset_results(dataset, results_dir='results'):
    """
    Aggregate results from all subsample rates for a given dataset.
    
    Args:
        dataset (str): Dataset name (e.g., 'qmnist')
        results_dir (str): Base results directory
        
    Returns:
        dict: Aggregated results across all subsample rates
    """
    dataset_dir = os.path.join(results_dir, dataset)
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return None
    
    # Find all subsample result files
    subsample_files = glob.glob(os.path.join(dataset_dir, 'subsample_*.json'))
    
    if not subsample_files:
        print(f"No subsample result files found in {dataset_dir}")
        return None
    
    print(f"Found {len(subsample_files)} subsample result files:")
    for f in subsample_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load and aggregate results
    aggregated = {
        'dataset': dataset,
        'subsample_rates': [],
        'results_by_rate': {},
        'aggregated_speedups': {},
        'aggregated_accuracies': {}
    }
    
    for file_path in sorted(subsample_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        subsample_rate = data['subsample_rate']
        aggregated['subsample_rates'].append(subsample_rate)
        aggregated['results_by_rate'][str(subsample_rate)] = data
        
        # Calculate aggregated statistics for this subsample rate
        seeds = data['seeds']
        
        # Accuracy statistics
        rs_accs = [data['rs_results'][str(seed)]['final_test_acc'] for seed in seeds]
        pt_accs = [data['pt_results'][str(seed)]['final_test_acc'] for seed in seeds]
        
        aggregated['aggregated_accuracies'][str(subsample_rate)] = {
            'rs_mean': float(np.mean(rs_accs)),
            'rs_std': float(np.std(rs_accs)),
            'pt_mean': float(np.mean(pt_accs)),
            'pt_std': float(np.std(pt_accs))
        }
        
        # Steps to target statistics
        rs_steps = [data['rs_results'][str(seed)]['steps_to_target'] for seed in seeds 
                   if data['rs_results'][str(seed)]['steps_to_target'] is not None]
        pt_steps = [data['pt_results'][str(seed)]['steps_to_target'] for seed in seeds 
                   if data['pt_results'][str(seed)]['steps_to_target'] is not None]
        
        if rs_steps and pt_steps:
            rs_steps_mean = float(np.mean(rs_steps))
            pt_steps_mean = float(np.mean(pt_steps))
            speedup = rs_steps_mean / pt_steps_mean if pt_steps_mean > 0 else 0
            
            aggregated['aggregated_speedups'][str(subsample_rate)] = {
                'rs_steps_mean': rs_steps_mean,
                'rs_steps_std': float(np.std(rs_steps)),
                'pt_steps_mean': pt_steps_mean,
                'pt_steps_std': float(np.std(pt_steps)),
                'speedup': speedup
            }
    
    # Sort subsample rates for consistent ordering
    aggregated['subsample_rates'].sort()
    
    # Save aggregated results
    output_file = os.path.join(dataset_dir, 'aggregated.json')
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Aggregated results saved to {output_file}")
    return aggregated

def print_aggregated_summary(aggregated_data):
    """Print a summary table of aggregated results."""
    if not aggregated_data:
        return
    
    dataset = aggregated_data['dataset']
    rates = aggregated_data['subsample_rates']
    
    print(f"\n{'='*60}")
    print(f"AGGREGATED SUMMARY FOR {dataset.upper()}")
    print(f"{'='*60}")
    
    print(f"{'Rate':<8} {'RS Acc':<12} {'PT Acc':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for rate in rates:
        rate_str = str(rate)
        acc_data = aggregated_data['aggregated_accuracies'][rate_str]
        
        rs_acc_str = f"{acc_data['rs_mean']:.3f}±{acc_data['rs_std']:.3f}"
        pt_acc_str = f"{acc_data['pt_mean']:.3f}±{acc_data['pt_std']:.3f}"
        
        if rate_str in aggregated_data['aggregated_speedups']:
            speedup = aggregated_data['aggregated_speedups'][rate_str]['speedup']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{rate:<8} {rs_acc_str:<12} {pt_acc_str:<12} {speedup_str:<10}")

def main():
    parser = argparse.ArgumentParser(description='Aggregate results across subsample rates')
    parser.add_argument('--dataset', type=str, default='qmnist',
                       help='Dataset to aggregate results for')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Base results directory')
    
    args = parser.parse_args()
    
    # Aggregate results
    aggregated = aggregate_dataset_results(args.dataset, args.results_dir)
    
    if aggregated:
        print_aggregated_summary(aggregated)

if __name__ == '__main__':
    main()