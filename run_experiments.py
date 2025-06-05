#!/usr/bin/env python3

import subprocess
import sys
import os
from plotting import plot_experiment_results

def run_experiment(dataset, seeds, subsample_rate):
    """
    Run a single experiment by calling train.py with specified parameters.
    """
    print(f"\n{'='*70}")
    print(f"Running experiment: {dataset}, subsample_rate={subsample_rate}, seeds={seeds}")
    print(f"{'='*70}")
    
    # Build command to call train.py
    cmd = [
        sys.executable, 'train.py',
        '--dataset', dataset,
        '--subsample_rate', str(subsample_rate),
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
    Stage 1: Test the complete pipeline with qmnist, 10% subsample rate, 1 seed.
    Later stages will expand to multiple datasets, subsample rates, and seeds.
    """
    print("Starting Stage 1 experiments...")
    
    # Stage 1 configuration: single experiment to test the pipeline
    experiments = [
        {
            'dataset': 'qmnist',
            'seeds': [42, 123, 456],  # 3 different seeds
            'subsample_rate': 0.1
        }
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
                result_file = f"results/{exp['dataset']}_subsample{exp['subsample_rate']}.json"
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