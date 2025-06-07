#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(result_file):
    """
    Create learning curves plot: test accuracy vs steps (averaged across seeds).
    
    Args:
        result_file (str): Path to the JSON results file
    """
    if not os.path.exists(result_file):
        print(f"Results file {result_file} not found.")
        return
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    dataset = results['dataset']
    subsample_rate = results['subsample_rate']
    seeds = results['seeds']
    target_accuracy = results['target_accuracy']
    
    # Collect all learning curves for averaging
    rs_all_steps = []
    rs_all_accs = []
    pt_all_steps = []
    pt_all_accs = []
    
    for seed in seeds:
        rs_steps = results['rs_results'][str(seed)]['history_by_step']['steps']
        rs_acc = results['rs_results'][str(seed)]['history_by_step']['val_accuracy']
        pt_steps = results['pt_results'][str(seed)]['history_by_step']['steps']
        pt_acc = results['pt_results'][str(seed)]['history_by_step']['val_accuracy']
        
        rs_all_steps.append(rs_steps)
        rs_all_accs.append(rs_acc)
        pt_all_steps.append(pt_steps)
        pt_all_accs.append(pt_acc)
    
    # Calculate averages (assume all runs have same step counts)
    rs_steps_avg = rs_all_steps[0]  # Use first seed's steps
    rs_acc_avg = np.mean(rs_all_accs, axis=0)
    rs_acc_std = np.std(rs_all_accs, axis=0)
    
    pt_steps_avg = pt_all_steps[0]  # Use first seed's steps  
    pt_acc_avg = np.mean(pt_all_accs, axis=0)
    pt_acc_std = np.std(pt_all_accs, axis=0)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Convert to percentage scale
    rs_acc_avg_pct = rs_acc_avg * 100
    rs_acc_std_pct = rs_acc_std * 100
    pt_acc_avg_pct = pt_acc_avg * 100
    pt_acc_std_pct = pt_acc_std * 100
    target_accuracy_pct = target_accuracy * 100
    
    # Plot with error bands
    ax.plot(rs_steps_avg, rs_acc_avg_pct, '--', color='orange', linewidth=2, label='Uniform Sampling')
    ax.fill_between(rs_steps_avg, rs_acc_avg_pct - rs_acc_std_pct, rs_acc_avg_pct + rs_acc_std_pct, 
                    alpha=0.2, color='orange')
    
    ax.plot(pt_steps_avg, pt_acc_avg_pct, '-', color='blue', linewidth=2, label='RHO-LOSS (Ours)')
    ax.fill_between(pt_steps_avg, pt_acc_avg_pct - pt_acc_std_pct, pt_acc_avg_pct + pt_acc_std_pct, 
                    alpha=0.2, color='blue')
    
    ax.axhline(y=target_accuracy_pct, color='black', linestyle=':', alpha=0.7, 
               label=f'Target Accuracy ({target_accuracy_pct:.1f}%)')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    
    # Format y-axis as percentages (auto-scale)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))
    ax.set_title(f'Learning Curves: {dataset.upper()} (subsample rate: {subsample_rate})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.7)  # lighter gridlines

    # Ensure x-axis starts at zero and 0 is labeled
    ax.set_xlim(left=0)
    xticks = ax.get_xticks()
    if 0 not in xticks:
        ax.set_xticks(np.insert(xticks, 0, 0))

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()  # keep tight layout for label spacing
    # Save plot with extra whitespace using pad_inches
    plot_file = result_file.replace('.json', '_learning_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=.3)
    print(f"Learning curves plot saved to {plot_file}")
    plt.close()
    
    return plot_file

def plot_speedup_bar_chart(result_file):
    """
    Create speedup bar chart: steps to target accuracy comparison.
    
    Args:
        result_file (str): Path to the JSON results file
    """
    if not os.path.exists(result_file):
        print(f"Results file {result_file} not found.")
        return
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    dataset = results['dataset']
    subsample_rate = results['subsample_rate']
    seeds = results['seeds']
    target_accuracy = results['target_accuracy']
    
    # Collect steps to target data
    rs_steps_to_target = []
    pt_steps_to_target = []
    
    for seed in seeds:
        rs_steps = results['rs_results'][str(seed)]['steps_to_target']
        pt_steps = results['pt_results'][str(seed)]['steps_to_target']
        if rs_steps is not None:
            rs_steps_to_target.append(rs_steps)
        if pt_steps is not None:
            pt_steps_to_target.append(pt_steps)
    
    # Calculate means and stds - handle empty arrays
    rs_mean = np.mean(rs_steps_to_target) if rs_steps_to_target else np.nan
    rs_std = np.std(rs_steps_to_target) if len(rs_steps_to_target) > 1 else 0
    pt_mean = np.mean(pt_steps_to_target) if pt_steps_to_target else np.nan
    pt_std = np.std(pt_steps_to_target) if len(pt_steps_to_target) > 1 else 0
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    methods = ['Random\nSampling', 'Prioritized\nTraining']
    means = [rs_mean, pt_mean]
    stds = [rs_std, pt_std]
    colors = ['blue', 'red']
    
    bars = ax.barh(methods, means, xerr=stds, capsize=5, alpha=0.7, color=colors, 
                   edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        if not np.isnan(mean) and np.isfinite(mean):
            ax.text(bar.get_width() + std + 20, bar.get_y() + bar.get_height()/2,
                    f'{mean:.0f}±{std:.0f}', ha='left', va='center', fontweight='bold')
    
    # Calculate and display speedup
    if (not np.isnan(rs_mean) and not np.isnan(pt_mean) and 
        np.isfinite(rs_mean) and np.isfinite(pt_mean) and pt_mean > 0):
        speedup = rs_mean / pt_mean
        ax.text(0.95, 0.05, f'Speedup: {speedup:.2f}x', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        ax.text(0.95, 0.05, 'Speedup: N/A', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.set_xlabel('Steps to Target Accuracy', fontsize=12)
    ax.set_title(f'Training Efficiency: {dataset.upper()} (target: {target_accuracy:.3f}%)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout(pad=2.0)
    
    # Save plot
    plot_file = result_file.replace('.json', '_speedup.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.35)
    print(f"Speedup plot saved to {plot_file}")
    plt.close()
    
    return plot_file

def plot_paper_style_table(result_file):
    """
    Create a paper-style comparison table using LaTeX for professional formatting.
    
    Args:
        result_file (str): Path to the JSON results file
    """
    if not os.path.exists(result_file):
        print(f"Results file {result_file} not found.")
        return
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    dataset = results['dataset']
    target_accuracy = results['target_accuracy']
    seeds = results['seeds']
    steps_per_epoch = results['steps_per_epoch']
    
    # Calculate statistics for uniform sampling (RS) and RHO-LOSS (PT)
    rs_final_accs = [results['rs_results'][str(seed)]['final_test_acc'] for seed in seeds]
    pt_final_accs = [results['pt_results'][str(seed)]['final_test_acc'] for seed in seeds]
    
    rs_steps_to_target = [results['rs_results'][str(seed)]['steps_to_target'] 
                         for seed in seeds if results['rs_results'][str(seed)]['steps_to_target'] is not None]
    pt_steps_to_target = [results['pt_results'][str(seed)]['steps_to_target'] 
                         for seed in seeds if results['pt_results'][str(seed)]['steps_to_target'] is not None]
    
    # Convert steps to epochs
    rs_epochs_to_target = [int(steps / steps_per_epoch) for steps in rs_steps_to_target]
    pt_epochs_to_target = [int(steps / steps_per_epoch) for steps in pt_steps_to_target]
    
    # Calculate means
    rs_acc_mean = np.mean(rs_final_accs) * 100  # Convert to percentage
    pt_acc_mean = np.mean(pt_final_accs) * 100  # Convert to percentage
    rs_epochs_mean = np.mean(rs_epochs_to_target) if rs_epochs_to_target else 999
    pt_epochs_mean = np.mean(pt_epochs_to_target) if pt_epochs_to_target else 999
    
    # Create LaTeX table
    latex_content = rf"""
\documentclass{{standalone}}
\usepackage{{booktabs}}
\usepackage{{array}}
\begin{{document}}
\begin{{tabular}}{{lccc}}
\toprule
\multicolumn{{4}}{{c}}{{\textit{{Number of epochs method needs to reach target accuracy $\downarrow$ (Final accuracy in parentheses)}}}} \\
\midrule
Dataset & Target Acc & Uniform Sample & RHO-LOSS \\
\midrule
{dataset.upper()} & {target_accuracy*100:.1f}\% & {int(rs_epochs_mean)} ({rs_acc_mean:.0f}\%) & {int(pt_epochs_mean)} ({pt_acc_mean:.0f}\%) \\
\bottomrule
\end{{tabular}}
\end{{document}}
"""
    
    # Write LaTeX file
    latex_file = result_file.replace('.json', '_paper_table.tex')
    with open(latex_file, 'w') as f:
        f.write(latex_content)
    
    # Compile LaTeX to PDF then convert to PNG
    import subprocess
    import tempfile
    import shutil
    
    try:
        # Get directory and base name
        base_dir = os.path.dirname(latex_file)
        base_name = os.path.splitext(os.path.basename(latex_file))[0]
        
        # Compile LaTeX
        subprocess.run(['pdflatex', '-output-directory', base_dir, latex_file], 
                      check=True, capture_output=True)
        
        # Convert PDF to PNG with high DPI and white background
        pdf_file = os.path.join(base_dir, f'{base_name}.pdf')
        png_file = result_file.replace('.json', '_paper_table.png')
        
        subprocess.run(['convert', '-density', '300', '-quality', '100', 
                       '-background', 'white', '-alpha', 'remove',
                       pdf_file, png_file], check=True, capture_output=True)
        
        # Clean up auxiliary files
        aux_extensions = ['.aux', '.log', '.pdf', '.tex']
        for ext in aux_extensions:
            aux_file = os.path.join(base_dir, f'{base_name}{ext}')
            if os.path.exists(aux_file):
                os.remove(aux_file)
        
        print(f"LaTeX-generated paper table saved to {png_file}")
        return png_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX: {e}")
        print("Falling back to matplotlib version...")
        return _plot_matplotlib_table_fallback(result_file, dataset, target_accuracy, 
                                             rs_epochs_mean, rs_acc_mean, pt_epochs_mean, pt_acc_mean)
    except FileNotFoundError as e:
        print(f"LaTeX or ImageMagick not found: {e}")
        print("Falling back to matplotlib version...")
        return _plot_matplotlib_table_fallback(result_file, dataset, target_accuracy, 
                                             rs_epochs_mean, rs_acc_mean, pt_epochs_mean, pt_acc_mean)

def _plot_matplotlib_table_fallback(result_file, dataset, target_accuracy, rs_epochs_mean, rs_acc_mean, pt_epochs_mean, pt_acc_mean):
    """Fallback matplotlib table if LaTeX fails"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Dataset', 'Target Acc', 'Uniform Sample', 'RHO-LOSS'],
        [dataset.upper(), f'{target_accuracy*100:.4f}%', 
         f'{int(rs_epochs_mean)} ({rs_acc_mean:.0f}%)', 
         f'{int(pt_epochs_mean)} ({pt_acc_mean:.0f}%)']
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center')
    
    header_text = 'Number of epochs method needs to reach target accuracy ↓ (Final accuracy in parentheses)'
    ax.text(0.5, 0.85, header_text, transform=ax.transAxes, ha='center', va='center',
            fontsize=10, style='italic')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    
    plt.title('Training Efficiency Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout(pad=2.0)
    
    plot_file = result_file.replace('.json', '_paper_table.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.35)
    plt.close()
    
    return plot_file

def plot_subsample_rate_speedup(aggregated_file):
    """
    Create horizontal grouped bar chart showing speedup across different subsample rates.
    This is now a wrapper that calls the grouped speedup function for backward compatibility.
    
    Args:
        aggregated_file (str): Path to the aggregated JSON results file
    """
    if not os.path.exists(aggregated_file):
        print(f"Aggregated results file {aggregated_file} not found.")
        return
    
    with open(aggregated_file, 'r') as f:
        data = json.load(f)
    
    dataset = data['dataset']
    subsample_rates = data['subsample_rates']
    
    # Use the grouped function with single dataset
    return plot_multi_dataset_grouped_speedup([dataset], subsample_rates, 
                                             os.path.dirname(os.path.dirname(aggregated_file)))

def plot_experiment_results(result_file):
    """
    Generate all plots for a prioritized training experiment.
    
    Args:
        result_file (str): Path to the JSON results file
    
    Returns:
        list: Paths to the generated plot files
    """
    plot_files = []
    
    # Generate all plots
    learning_curves_file = plot_learning_curves(result_file)
    if learning_curves_file:
        plot_files.append(learning_curves_file)
    
    speedup_file = plot_speedup_bar_chart(result_file)
    if speedup_file:
        plot_files.append(speedup_file)
    
    paper_table_file = plot_paper_style_table(result_file)
    if paper_table_file:
        plot_files.append(paper_table_file)
    
    return plot_files

def plot_aggregated_results(dataset, results_dir='results', default_subsample_rate=0.1):
    """
    Plot aggregated results for a dataset across multiple subsample rates.
    
    Args:
        dataset (str): Dataset name
        results_dir (str): Base results directory
        default_subsample_rate (float): Subsample rate to use for learning curves
    """
    dataset_dir = os.path.join(results_dir, dataset)
    aggregated_file = os.path.join(dataset_dir, 'aggregated.json')
    
    if not os.path.exists(aggregated_file):
        print(f"Aggregated file {aggregated_file} not found. Run aggregate_results.py first.")
        return
    
    plot_files = []
    
    # Plot subsample rate speedup chart
    speedup_file = plot_subsample_rate_speedup(aggregated_file)
    if speedup_file:
        plot_files.append(speedup_file)
    
    # Plot learning curves using the default subsample rate
    default_result_file = os.path.join(dataset_dir, f'subsample_{default_subsample_rate}.json')
    if os.path.exists(default_result_file):
        print(f"Plotting learning curves using subsample rate {default_subsample_rate}")
        
        learning_curves_file = plot_learning_curves(default_result_file)
        if learning_curves_file:
            plot_files.append(learning_curves_file)
        
        speedup_file = plot_speedup_bar_chart(default_result_file)
        if speedup_file:
            plot_files.append(speedup_file)
        
        summary_file = plot_summary_table(default_result_file)
        if summary_file:
            plot_files.append(summary_file)
        
        paper_table_file = plot_paper_style_table(default_result_file)
        if paper_table_file:
            plot_files.append(paper_table_file)
    else:
        print(f"Default subsample rate file {default_result_file} not found.")
    
    return plot_files

def plot_multi_dataset_grouped_speedup(datasets, subsample_rates, results_dir='results'):
    """
    Create a grouped horizontal bar chart comparing speedups across multiple datasets and subsample rates.
    
    Args:
        datasets (list): List of dataset names
        subsample_rates (list): List of subsample rates to compare
        results_dir (str): Base results directory
    """
    # Collect speedup data
    speedup_data = {}
    
    for dataset in datasets:
        speedup_data[dataset] = {}
        for rate in subsample_rates:
            result_file = os.path.join(results_dir, dataset, f'subsample_{rate}.json')
            
            if not os.path.exists(result_file):
                print(f"Warning: Result file {result_file} not found")
                speedup_data[dataset][rate] = 0
                continue
                
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            seeds = results['seeds']
            
            # Collect steps to target data
            rs_steps_to_target = []
            pt_steps_to_target = []
            
            for seed in seeds:
                rs_steps = results['rs_results'][str(seed)]['steps_to_target']
                pt_steps = results['pt_results'][str(seed)]['steps_to_target']
                if rs_steps is not None:
                    rs_steps_to_target.append(rs_steps)
                if pt_steps is not None:
                    pt_steps_to_target.append(pt_steps)
            
            # Calculate speedup
            rs_mean = np.mean(rs_steps_to_target) if rs_steps_to_target else np.nan
            pt_mean = np.mean(pt_steps_to_target) if pt_steps_to_target else np.nan
            
            if (not np.isnan(rs_mean) and not np.isnan(pt_mean) and 
                np.isfinite(rs_mean) and np.isfinite(pt_mean) and pt_mean > 0):
                speedup = rs_mean / pt_mean
                speedup_data[dataset][rate] = speedup
            else:
                speedup_data[dataset][rate] = 0
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Colors for different subsample rates
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(subsample_rates)]
    
    # Position parameters
    bar_height = 0.15
    y_positions = np.arange(len(datasets))
    
    # Plot bars for each subsample rate
    for i, rate in enumerate(subsample_rates):
        speedups = [speedup_data[dataset][rate] for dataset in datasets]
        y_pos = y_positions + i * bar_height
        
        bars = ax.barh(y_pos, speedups, bar_height, 
                       color=colors[i], alpha=0.8, 
                       label=f'{int(rate*100)}%',
                       edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            if speedup > 0:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                        f'{speedup:.2f}', ha='left', va='center', fontsize=9)
    
    # Customize plot
    ax.set_yticks(y_positions + bar_height * (len(subsample_rates) - 1) / 2)
    ax.set_yticklabels([dataset.upper() for dataset in datasets])
    ax.set_xlabel('Speedup', fontsize=12)
    ax.set_title('RHO-LOSS Training Speedup by Dataset and Subsample Rate', 
                 fontsize=14, fontweight='bold')
    
    # Add legend with title
    legend = ax.legend(title='Percentage Selected', loc='lower right', 
                      bbox_to_anchor=(0.98, 0.02))
    legend.get_title().set_fontweight('bold')
    
    # Add grid and styling
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    
    # Add vertical line at speedup = 1
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(results_dir, 'multi_dataset_grouped_speedup.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Multi-dataset grouped speedup plot saved to {plot_file}")
    plt.close()
    
    return plot_file

def plot_all_experiments(results_dir='results'):
    """
    Plot results for all experiment files in the results directory and generate grouped speedup chart.
    
    Args:
        results_dir (str): Directory containing JSON result files
    """
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found.")
        return
    
    # Find all dataset directories and JSON files
    datasets = []
    all_subsample_rates = set()
    
    for item in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, item)
        if os.path.isdir(dataset_path):
            # Check if this directory contains experiment results
            json_files = [f for f in os.listdir(dataset_path) if f.startswith('subsample_') and f.endswith('.json')]
            if json_files:
                datasets.append(item)
                # Extract subsample rates from filenames
                for json_file in json_files:
                    rate_str = json_file.replace('subsample_', '').replace('.json', '')
                    try:
                        rate = float(rate_str)
                        all_subsample_rates.add(rate)
                    except ValueError:
                        continue
    
    # Skip legacy JSON files in root directory - only process dataset directories
    if not datasets:
        print(f"No experiment dataset directories found in {results_dir}")
        return
    
    # Plot individual experiment results
    all_result_files = []
    
    # Process dataset directories
    for dataset in datasets:
        dataset_path = os.path.join(results_dir, dataset)
        json_files = [f for f in os.listdir(dataset_path) if f.startswith('subsample_') and f.endswith('.json')]
        
        print(f"\nProcessing {dataset} experiments:")
        for json_file in json_files:
            print(f"  - {json_file}")
            result_path = os.path.join(dataset_path, json_file)
            all_result_files.append(result_path)
            try:
                plot_experiment_results(result_path)
            except Exception as e:
                print(f"Error plotting {json_file}: {e}")
    
    # Legacy files in root directory are ignored - we only process dataset directories
    
    # Generate grouped speedup chart
    if datasets and all_subsample_rates:
        print(f"\nGenerating grouped speedup chart for datasets: {datasets}")
        print(f"Subsample rates: {sorted(all_subsample_rates)}")
        try:
            plot_multi_dataset_grouped_speedup(datasets, sorted(all_subsample_rates), results_dir)
        except Exception as e:
            print(f"Error generating grouped speedup chart: {e}")
    
    print(f"\nAll plots saved in {results_dir}/")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot prioritized training experiment results')
    parser.add_argument('--file', type=str, help='Specific JSON results file to plot')
    parser.add_argument('--all', action='store_true', help='Plot all results in results/ directory')
    parser.add_argument('--aggregated', type=str, help='Plot aggregated results for dataset (e.g., "qmnist")')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       help='Plot grouped speedup chart for datasets (e.g., qmnist cocoreg cocokp)')
    parser.add_argument('--subsample_rates', type=float, nargs='+', default=[0.01, 0.1, 0.5],
                       help='Subsample rates to include in grouped chart (default: 0.01 0.1 0.5)')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory (default: results)')
    parser.add_argument('--default_subsample_rate', type=float, default=0.1, 
                       help='Default subsample rate for learning curves (default: 0.1)')
    
    args = parser.parse_args()
    
    if args.file:
        plot_experiment_results(args.file)
    elif args.aggregated:
        plot_aggregated_results(args.aggregated, args.results_dir, args.default_subsample_rate)
    elif args.datasets:
        plot_multi_dataset_grouped_speedup(args.datasets, args.subsample_rates, args.results_dir)
    elif args.all:
        plot_all_experiments(args.results_dir)
    else:
        print("Please specify --file, --aggregated, --datasets, or --all")
        print("Examples:")
        print("  python plotting.py --file results/qmnist/subsample_0.1.json")
        print("  python plotting.py --aggregated qmnist")
        print("  python plotting.py --datasets qmnist cocoreg cocokp")
        print("  python plotting.py --all")