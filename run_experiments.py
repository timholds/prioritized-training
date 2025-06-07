#!/usr/bin/env python3

import subprocess
import sys
import os
from plotting import plot_experiment_results

def run_experiment(dataset, seeds, subsample_rate):
    """
    Run a single experiment by calling train.py with specified parameters.
    subsample_rate: list of float(s)
    """
    print(f"\n{'='*70}")
    print(f"Running experiment: {dataset}, subsample_rate={subsample_rate}, seeds={seeds}")
    print(f"{'='*70}")
    
    # Build command to call train.py
    cmd = [
        sys.executable, 'train.py',
        '--dataset', dataset,
        '--subsample_rate'
    ] + [str(rate) for rate in subsample_rate] + [
        '--seeds'
    ] + [str(seed) for seed in seeds]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Experiment failed with return code {result.returncode}")
        return False
    else:
        print(f"SUCCESS: Experiment completed successfully")
        return True

def main():
    """
    Run experiments on QMNIST and COCO datasets with subsample rate sweep.
    """
    print("Starting multi-dataset experiments...")
    
    # Multi-dataset configuration with subsample rate sweep
    seeds = [42] #, 123, 456]  
    subsample_rates = [0.1] #, 0.5]  
    experiments = [
        # {
        #     'dataset': 'qmnist',
        #     'seeds': seeds,  
        #     'subsample_rate': subsample_rates 
        # },
        {
            'dataset': 'cocoreg',
            'seeds': seeds,
            'subsample_rate': subsample_rates
        },
        # {
        #     'dataset': 'cocokp',
        #     'seeds': seeds,
        #     'subsample_rate': subsample_rates
        # }
    ]
    
    # Track success/failure
    successful_experiments = 0
    total_experiments = len(experiments)
    
    for exp in experiments:
        success = run_experiment(
            dataset=exp['dataset'],
            seeds=exp['seeds'], 
            subsample_rate=exp['subsample_rate']
        )
        if success:
            successful_experiments += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"Successful experiments: {successful_experiments}/{total_experiments}")
    
    if successful_experiments == total_experiments:
        print("🎉 All experiments completed successfully!")
        print(f"Results saved in results/ directory")
        print(f"Models cached in models/ directory")
        
        # Generate plots for completed experiments
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print(f"{'='*70}")
        try:
            for exp in experiments:
                for rate in exp['subsample_rate']:
                    result_file = f"results/{exp['dataset']}/subsample_{rate}.json"
                    if os.path.exists(result_file):
                        print(f"Generating plot for {result_file}...")
                        plot_experiment_results(result_file)
                    else:
                        print(f"Warning: Result file {result_file} not found")
            print("📊 All plots generated successfully!")
        except Exception as e:
            print(f"⚠️ Error generating plots: {e}")
            print("You can manually generate plots later using: python plotting.py --all")
    else:
        print(f"❌ {total_experiments - successful_experiments} experiments failed")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)