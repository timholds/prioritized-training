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
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = result_file.replace('.json', '_learning_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
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
        rs_steps_to_target.append(rs_steps if rs_steps is not None else np.nan)
        pt_steps_to_target.append(pt_steps if pt_steps is not None else np.nan)
    
    # Calculate means and stds
    rs_mean = np.nanmean(rs_steps_to_target)
    rs_std = np.nanstd(rs_steps_to_target)
    pt_mean = np.nanmean(pt_steps_to_target)
    pt_std = np.nanstd(pt_steps_to_target)
    
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
        if not np.isnan(mean):
            ax.text(bar.get_width() + std + 20, bar.get_y() + bar.get_height()/2,
                    f'{mean:.0f}±{std:.0f}', ha='left', va='center', fontweight='bold')
    
    # Calculate and display speedup
    if not np.isnan(rs_mean) and not np.isnan(pt_mean) and pt_mean > 0:
        speedup = rs_mean / pt_mean
        ax.text(0.95, 0.05, f'Speedup: {speedup:.2f}x', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Steps to Target Accuracy', fontsize=12)
    ax.set_title(f'Training Efficiency: {dataset.upper()} (target: {target_accuracy:.3f}%)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = result_file.replace('.json', '_speedup.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Speedup plot saved to {plot_file}")
    plt.close()
    
    return plot_file

def plot_summary_table(result_file):
    """
    Create summary statistics table as a plot.
    
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
    
    # Calculate statistics
    rs_final_accs = [results['rs_results'][str(seed)]['final_test_acc'] for seed in seeds]
    pt_final_accs = [results['pt_results'][str(seed)]['final_test_acc'] for seed in seeds]
    
    rs_steps_to_target = [results['rs_results'][str(seed)]['steps_to_target'] 
                         for seed in seeds if results['rs_results'][str(seed)]['steps_to_target'] is not None]
    pt_steps_to_target = [results['pt_results'][str(seed)]['steps_to_target'] 
                         for seed in seeds if results['pt_results'][str(seed)]['steps_to_target'] is not None]
    
    rs_acc_mean = np.mean(rs_final_accs)
    rs_acc_std = np.std(rs_final_accs)
    pt_acc_mean = np.mean(pt_final_accs)
    pt_acc_std = np.std(pt_final_accs)
    
    rs_steps_mean = np.mean(rs_steps_to_target) if rs_steps_to_target else np.nan
    pt_steps_mean = np.mean(pt_steps_to_target) if pt_steps_to_target else np.nan
    
    # Calculate improvements
    acc_improvement = ((pt_acc_mean - rs_acc_mean) / rs_acc_mean * 100) if rs_acc_mean > 0 else 0
    speedup = rs_steps_mean / pt_steps_mean if not np.isnan(rs_steps_mean) and not np.isnan(pt_steps_mean) and pt_steps_mean > 0 else np.nan
    
    # Create table data
    table_data = [
        ['Metric', 'Random Sampling', 'Prioritized Training'],
        ['Final Accuracy', f'{rs_acc_mean:.4f} ± {rs_acc_std:.4f}', f'{pt_acc_mean:.4f} ± {pt_acc_std:.4f}'],
        ['Steps to Target', f'{rs_steps_mean:.0f}' if not np.isnan(rs_steps_mean) else 'N/A', 
         f'{pt_steps_mean:.0f}' if not np.isnan(pt_steps_mean) else 'N/A'],
        ['Accuracy Improvement', '—', f'{acc_improvement:+.2f}%'],
        ['Speedup', '—', f'{speedup:.2f}x' if not np.isnan(speedup) else 'N/A'],
        ['', '', ''],
        ['Dataset', dataset.upper(), ''],
        ['Subsample Rate', f'{subsample_rate}', ''],
        ['Target Accuracy', f'{target_accuracy:.2f}', ''],
        ['Holdout Accuracy', f'{results["holdout_test_acc"]:.4f}', ''],
        ['Holdout Epochs', f'{results["holdout_epochs"]}', ''],
        ['Seeds Used', f'{len(seeds)} ({", ".join(map(str, seeds))})', '']
    ]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement rows
    for i in [3, 4]:  # Improvement and speedup rows
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor('#e6f3ff')
    
    # Color separator row
    for j in range(len(table_data[0])):
        table[(5, j)].set_facecolor('#f0f0f0')
    
    plt.title(f'Summary Statistics: {dataset.upper()} (subsample rate: {subsample_rate})', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = result_file.replace('.json', '_summary_table.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Summary table saved to {plot_file}")
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
    rs_epochs_mean = np.mean(rs_epochs_to_target) if rs_epochs_to_target else float('inf')
    pt_epochs_mean = np.mean(pt_epochs_to_target) if pt_epochs_to_target else float('inf')
    
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
        [dataset.upper(), f'{target_accuracy*100:.1f}%', 
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
    plt.tight_layout()
    
    plot_file = result_file.replace('.json', '_paper_table.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

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
    
    summary_file = plot_summary_table(result_file)
    if summary_file:
        plot_files.append(summary_file)
    
    paper_table_file = plot_paper_style_table(result_file)
    if paper_table_file:
        plot_files.append(paper_table_file)
    
    return plot_files

def plot_all_experiments(results_dir='results'):
    """
    Plot results for all experiment files in the results directory.
    
    Args:
        results_dir (str): Directory containing JSON result files
    """
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found.")
        return
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON result files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} result files:")
    for json_file in json_files:
        print(f"  - {json_file}")
        result_path = os.path.join(results_dir, json_file)
        try:
            plot_experiment_results(result_path)
        except Exception as e:
            print(f"Error plotting {json_file}: {e}")
    
    print(f"\nAll plots saved in {results_dir}/")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot prioritized training experiment results')
    parser.add_argument('--file', type=str, help='Specific JSON results file to plot')
    parser.add_argument('--all', action='store_true', help='Plot all results in results/ directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory (default: results)')
    
    args = parser.parse_args()
    
    if args.file:
        plot_experiment_results(args.file)
    elif args.all:
        plot_all_experiments(args.results_dir)
    else:
        print("Please specify --file or --all")
        print("Examples:")
        print("  python plotting.py --file results/qmnist_subsample0.1.json")
        print("  python plotting.py --all")