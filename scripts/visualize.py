import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import torch
import numpy as np
from datetime import datetime
import os
import time
import diffuser.utils as utils



def setup_all_seeds_figures(projection_variants):
    figs_all_seeds, axes_all_seeds = zip(*[plt.subplots(1, 1, figsize=(9, 10)) for _ in range(len(projection_variants))])
    figs_all_seeds = list(figs_all_seeds)
    axes_all_seeds = list(axes_all_seeds)
    return figs_all_seeds, axes_all_seeds


def setup_figures(n_trials, plot_how_many, projection_variants):
    fig_all, ax_all = plt.subplots(min(n_trials, plot_how_many), len(projection_variants), figsize=(10 * len(projection_variants), 10 * min(n_trials, plot_how_many)))
    ax_all = ax_all.reshape(n_trials, -1)
    return fig_all, ax_all

def plot_trial_results(i, variant_idx, variant, obs_buffer, obs_indices, sampled_trajectories_all, ax, ax_all, axes_all_seeds, seed, seeds, 
                       colors, n_success, collision_free_completed, n_steps, ax_limits, constraint_types, polytopic_constraints, 
                       obstacle_constraints, args):
    """Plot the results of a single trial."""
    # Define plot states
    plot_states = ['x', 'y', 'x_des', 'y_des']

    # Create a better color palette
    state_colors = plt.cm.viridis(np.linspace(0, 0.8, len(plot_states)))
    trajectory_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot individual state variables with improved styling
    for j in range(len(plot_states)):
        ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]], 
                     color=state_colors[j], linewidth=2)
        ax[i, j].set_title(plot_states[j], fontsize=12, fontweight='bold')
        ax[i, j].grid(True, linestyle='--', alpha=0.7)
        ax[i, j].set_xlabel('Timestep', fontsize=10)
        ax[i, j].set_ylabel(f'{plot_states[j]} value', fontsize=10)
    
    # Plot trajectories in 2D space
    axes = [ax[i, 4], ax_all[i, variant_idx]]
    for curr_ax in axes:
        # Plot trajectory
        curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']], 
                    np.array(obs_buffer)[:, obs_indices['y']], 
                    color='#333333', linewidth=2.5, label='Trajectory')
        
        # Mark start point with a more visible marker
        curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']], 
                    np.array(obs_buffer)[0, obs_indices['y']], 
                    marker='o', markersize=10, color='#00cc00', 
                    markeredgecolor='black', markeredgewidth=1.5, label='Start')
        
        # Add better labels and formatting
        curr_ax.set_xlim(ax_limits[0])
        curr_ax.set_ylim(ax_limits[1])
        curr_ax.set_xlabel('X Position', fontsize=10)
        curr_ax.set_ylabel('Y Position', fontsize=10)
        curr_ax.set_title(f'Trajectory (Variant: {variant})', fontsize=12, fontweight='bold')
        curr_ax.grid(True, linestyle='--', alpha=0.3)

    # Plot trajectories across all seeds with distinct colors
    seed_colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    axes_all_seeds[variant_idx].plot(np.array(obs_buffer)[:, obs_indices['x']], 
                                   np.array(obs_buffer)[:, obs_indices['y']], 
                                   color=seed_colors[seed % len(seed_colors)], 
                                   linewidth=2.5, label=f'Seed {seed}')
    
    # Add legend to the all seeds plot (only once per variant)
    if seed == seeds[-1]:
        axes_all_seeds[variant_idx].legend(fontsize=8, loc='upper right')
    
    # Plot sampled trajectories
    axes = [ax[i, 5], ax_all[i, variant_idx]]
    for __ in range(len(sampled_trajectories_all[i])):
        for ___ in range(min(args.batch_size, 4)):
            for curr_ax in axes:
                # Plot sampled trajectories with transparency
                curr_ax.plot(sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['x']], 
                           sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['y']], 
                           color=trajectory_colors[___ % len(trajectory_colors)], 
                           alpha=0.3, linewidth=1)
                
                # Mark the start of sampled trajectories
                curr_ax.plot(sampled_trajectories_all[i][__][___, 0, obs_indices['x']], 
                           sampled_trajectories_all[i][__][___, 0, obs_indices['y']], 
                           marker='o', markersize=6, color='green', alpha=0.5)
    
    # Format the sampled trajectories plot
    ax[i, 5].set_xlim(ax_limits[0])
    ax[i, 5].set_ylim(ax_limits[1])
    ax[i, 5].set_title('Sampled Trajectories', fontsize=12, fontweight='bold')
    ax[i, 5].set_xlabel('X Position', fontsize=10)
    ax[i, 5].set_ylabel('Y Position', fontsize=10)
    ax[i, 5].grid(True, linestyle='--', alpha=0.3)

    # Plot constraints with improved visibility
    exp = 'avoiding-d3il'
    axes = [ax[i, 4], ax[i, 5], ax_all[i, variant_idx]]
    for curr_ax in axes: 
        # Plot environment constraints
        utils.plot_environment_constraints(exp, curr_ax)

        # Plot halfspace constraints with better visibility
        if 'halfspace' in constraint_types:
            utils.plot_halfspace_constraints(exp, polytopic_constraints, curr_ax, ax_limits)

        # Plot obstacle constraints with improved styling
        if 'obstacles' in constraint_types:
            for idx, constraint in enumerate(obstacle_constraints):
                obstacle = matplotlib.patches.Circle(
                    constraint['center'], 
                    constraint['radius'], 
                    color=plt.cm.Reds(0.6 + 0.3*(idx/len(obstacle_constraints))), 
                    alpha=0.3,
                    linewidth=1.5,
                    edgecolor='red'
                )
                curr_ax.add_patch(obstacle)
    
    # Add a text box with trial success information
    success_text = f"Success: {'Yes' if n_success[i] == 1 else 'No'}\n"
    success_text += f"Constraints Met: {'Yes' if collision_free_completed[i] == 1 else 'No'}\n"
    success_text += f"Steps: {int(n_steps[i])}"
    
    ax[i, 4].text(0.02, 0.02, success_text,
                 transform=ax[i, 4].transAxes,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                 fontsize=9, verticalalignment='bottom')


def create_summary_visualization(compare_result, save_path):
    """Create separate visualization plots comparing metrics across different variants.
    
    Args:
        compare_result: Dictionary containing metrics for different variants
        save_path: Path to save the visualizations
    """
    # Get variants and prepare colors
    variants = list(compare_result.keys())
    variant_colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Success Metrics Comparison
    fig_success = plt.figure(figsize=(10, 6), dpi=120)
    ax_success = fig_success.add_subplot(111)
    
    metric_names = ['Success Rate', 'Constraint\nSatisfaction', 'Success with\nConstraints']
    x = np.arange(len(metric_names))
    width = 0.8 / len(variants)
    
    for i, variant in enumerate(variants):
        metrics = compare_result[variant]
        success_rates = [
            np.mean(metrics['n_success']) * 100,
            np.mean(metrics['collision_free_completed']) * 100,
            np.mean(metrics['n_success_and_constraints']) * 100
        ]
        
        offset = (i - len(variants)/2 + 0.5) * width
        bars = ax_success.bar(x + offset, success_rates, width, 
                          label=variant, color=variant_colors[i], 
                          edgecolor='black', linewidth=1)
        
        # Add percentage labels on bars
        for bar, value in zip(bars, success_rates):
            height = bar.get_height()
            ax_success.text(bar.get_x() + bar.get_width()/2, height + 2,
                         f'{value:.1f}%', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')
    
    ax_success.set_ylabel('Percentage (%)', fontsize=12)
    ax_success.set_title('Success Metrics Comparison Across Variants', fontsize=14, fontweight='bold')
    ax_success.set_xticks(x)
    ax_success.set_xticklabels(metric_names, fontsize=11)
    ax_success.set_ylim(0, 110)  # Leave room for percentage text
    ax_success.grid(axis='y', linestyle='--', alpha=0.7)
    ax_success.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    success_path = os.path.join(save_path, 'success_metrics.png')
    fig_success.savefig(success_path, bbox_inches='tight', dpi=120)
    plt.close(fig_success)
    
    # 2. Steps to Completion Comparison
    fig_steps = plt.figure(figsize=(8, 6), dpi=120)
    ax_steps = fig_steps.add_subplot(111)
    
    for i, variant in enumerate(variants):
        metrics = compare_result[variant]
        
        # Plot steps distribution using violin plot
        vp = ax_steps.violinplot(
            [metrics['n_steps']], 
            positions=[i], 
            showmeans=False,
            showmedians=True
        )
        
        # Customize violin plot
        for pc in vp['bodies']:
            pc.set_facecolor(variant_colors[i])
            pc.set_alpha(0.5)
        
        # Add scatter points
        ax_steps.scatter(
            np.ones_like(metrics['n_steps']) * i, 
            metrics['n_steps'],
            color=variant_colors[i], s=50, alpha=0.6, edgecolor='black'
        )
        
        # Add mean as text
        mean_steps = np.mean(metrics['n_steps'])
        ax_steps.text(i, np.max(metrics['n_steps']) + 5, 
                     f'{mean_steps:.1f}', 
                     ha='center', fontsize=10, fontweight='bold')
    
    ax_steps.set_ylabel('Number of Steps', fontsize=12)
    ax_steps.set_title('Steps to Completion Comparison Across Variants', fontsize=14, fontweight='bold')
    ax_steps.set_xticks(range(len(variants)))
    ax_steps.set_xticklabels(variants, fontsize=11)
    ax_steps.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    steps_path = os.path.join(save_path, 'steps_comparison.png')
    fig_steps.savefig(steps_path, bbox_inches='tight', dpi=120)
    plt.close(fig_steps)
    
    # 3. Violations Comparison
    fig_violations = plt.figure(figsize=(8, 6), dpi=120)
    ax_violations = fig_violations.add_subplot(111)
    
    for i, variant in enumerate(variants):
        metrics = compare_result[variant]
        
        # Add violin plot
        vp = ax_violations.violinplot(
            [metrics['n_violations']], 
            positions=[i], 
            showmeans=False, 
            showmedians=True
        )
        
        # Customize violin plot
        for pc in vp['bodies']:
            pc.set_facecolor(variant_colors[i])
            pc.set_alpha(0.5)
        
        # Add scatter points
        ax_violations.scatter(
            np.ones_like(metrics['n_violations']) * i, 
            metrics['n_violations'],
            color=variant_colors[i], s=50, alpha=0.6, edgecolor='black',
            label=variant
        )
        
        # Add mean as text
        mean_violations = np.mean(metrics['n_violations'])
        ax_violations.text(i, np.max(metrics['n_violations']) + 0.5, 
                     f'Mean: {mean_violations:.2f}', 
                     ha='center', fontsize=10)
    
    ax_violations.set_ylabel('Number of Violations', fontsize=12)
    ax_violations.set_title('Constraint Violations Comparison Across Variants', fontsize=14, fontweight='bold')
    ax_violations.set_xticks(range(len(variants)))
    ax_violations.set_xticklabels(variants, fontsize=11)
    ax_violations.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    violations_path = os.path.join(save_path, 'violations_comparison.png')
    fig_violations.savefig(violations_path, bbox_inches='tight', dpi=120)
    plt.close(fig_violations)
    
    # 4. Computation Time Comparison
    fig_time = plt.figure(figsize=(8, 6), dpi=120)
    ax_time = fig_time.add_subplot(111)
    
    for i, variant in enumerate(variants):
        metrics = compare_result[variant]
        
        # Violin plot for computation time
        vp = ax_time.violinplot(
            [metrics['avg_time']], 
            positions=[i], 
            showmeans=False,
            showmedians=True
        )
        
        # Customize violin plot
        for pc in vp['bodies']:
            pc.set_facecolor(variant_colors[i])
            pc.set_alpha(0.5)
        
        # Add individual points
        ax_time.scatter(
            np.ones_like(metrics['avg_time']) * i, 
            metrics['avg_time'],
            color=variant_colors[i], s=50, alpha=0.6, edgecolor='black'
        )
        
        # Add mean as text and horizontal line
        mean_time = np.mean(metrics['avg_time'])
        ax_time.hlines(
            mean_time, i-0.3, i+0.3, 
            colors='red', linestyles='--', linewidth=1.5
        )
        ax_time.text(i, mean_time * 1.1, 
                     f'{mean_time:.3f}s', 
                     ha='center', fontsize=10, color='red')
    
    ax_time.set_ylabel('Computation Time (s)', fontsize=12)
    ax_time.set_title('Computation Time Comparison Across Variants', fontsize=14, fontweight='bold')
    ax_time.set_xticks(range(len(variants)))
    ax_time.set_xticklabels(variants, fontsize=11)
    ax_time.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    time_path = os.path.join(save_path, 'computation_time_comparison.png')
    fig_time.savefig(time_path, bbox_inches='tight', dpi=120)
    plt.close(fig_time)
    
    # 5. Summary Statistics Table
    fig_table = plt.figure(figsize=(12, 4), dpi=120)
    ax_table = fig_table.add_subplot(111)
    
    # Hide axes
    ax_table.axis('off')
    ax_table.axis('tight')
    
    # Create the table data
    stat_cols = ['Success\nRate (%)', 'Constraint\nSatisfaction (%)', 
                'Success with\nConstraints (%)', 'Avg\nSteps', 
                'Violations\n(avg #)', 'Violation\nMagnitude', 'Comp.\nTime (s)']
    cell_text = []
    
    for variant in variants:
        metrics = compare_result[variant]
        row = [
            f"{np.mean(metrics['n_success'])*100:.1f}",
            f"{np.mean(metrics['collision_free_completed'])*100:.1f}",
            f"{np.mean(metrics['n_success_and_constraints'])*100:.1f}",
            f"{np.mean(metrics['n_steps']):.1f}",
            f"{np.mean(metrics['n_violations']):.2f}",
            f"{np.mean(metrics['total_violations']):.2e}",
            f"{np.mean(metrics['avg_time']):.3f}"
        ]
        cell_text.append(row)
    
    # Create table
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=variants,
        colLabels=stat_cols,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Title for the table
    fig_table.suptitle('Summary Statistics Across Variants', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    table_path = os.path.join(save_path, 'summary_statistics.png')
    fig_table.savefig(table_path, bbox_inches='tight', dpi=120)
    plt.close(fig_table)
    
    print(f"Comparison visualizations saved to {save_path}")
