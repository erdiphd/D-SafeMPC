"""
Evaluation script with integrated trajectory generation

This script combines trajectory generation and evaluation in a single run.
The script can either:
1. Generate new random obstacle trajectories using configurable seeds
2. Load existing trajectory files

Configuration (in config/projection_eval.yaml):
- generate_trajectories: True/False to generate new or use existing trajectories
- trajectory_seed: Seed for reproducible trajectory generation
- trajectory_file: Path to save/load trajectory file

Usage:
    python scripts/eval.py

The script will first generate/load trajectories, then run the evaluation.
"""

import time
import yaml
import os
import torch
from copy import copy
import minari
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import sys
path_str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_str)
import diffuser.utils as utils
from diffuser.sampling import Policy, Projector
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from scripts.visualize import (setup_all_seeds_figures, setup_figures, plot_trial_results, create_summary_visualization)
import os
import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ==================== TRAJECTORY GENERATION COMPONENTS ====================

@dataclass
class ObstacleTrajectoryParams:
    """Parameters for circular trajectory generation"""
    r: float            # radius
    omega: float        # angular velocity
    phi: float          # phase offset
    x_c: float          # center x
    y_c: float          # center y
    noise_std: float = 0.02  # noise standard deviation


def generate_random_trajectory_params(num_obstacles: int = 6, 
                                    seed: int = 42,
                                    x_bounds: Tuple[float, float] = (0.2, 0.8),
                                    y_bounds: Tuple[float, float] = (-0.4, 0.4),
                                    r_range: Tuple[float, float] = (0.05, 0.15),
                                    omega_range: Tuple[float, float] = (1.0, 3.0),
                                    noise_range: Tuple[float, float] = (0.0, 0.05),
                                    given_positions: dict = None) -> List[ObstacleTrajectoryParams]:
    """Generate random trajectory parameters within specified constraints"""
    np.random.seed(seed)
    
    print(f"Generating {num_obstacles} random trajectory parameters with seed {seed}")
    print(f"  - Using given positions: {given_positions is not None}")
    
    params_list = []
    for i in range(num_obstacles):
        r = np.random.uniform(r_range[0], r_range[1])
        
        if given_positions is not None:
            # Use given positions from YAML
            key = list(given_positions.keys())[i] if i < len(given_positions) else f'obstacle_{i+1}'
            if key in given_positions:
                x_c, y_c = given_positions[key]
                print(f"  Obstacle {i+1}: Using given center position ({x_c:.3f}, {y_c:.3f})")
                # Adjust radius if the given position would cause the circle to go out of bounds
                max_r_x = min(x_c - x_bounds[0], x_bounds[1] - x_c)
                max_r_y = min(y_c - y_bounds[0], y_bounds[1] - y_c)
                max_r = min(max_r_x, max_r_y)
                if r > max_r:
                    r = max_r * 0.9  # Use 90% of max possible radius for safety
                    print(f"    Adjusted radius to {r:.3f} to stay within bounds")
            else:
                # Fallback to random if not enough given positions
                x_min_center = x_bounds[0] + r
                x_max_center = x_bounds[1] - r
                x_c = np.random.uniform(x_min_center, x_max_center)
                y_min_center = y_bounds[0] + r
                y_max_center = y_bounds[1] - r
                y_c = np.random.uniform(y_min_center, y_max_center)
                print(f"  Obstacle {i+1}: Generated random center ({x_c:.3f}, {y_c:.3f}) - not enough given positions")
        else:
            # Generate random center position within bounds
            x_min_center = x_bounds[0] + r
            x_max_center = x_bounds[1] - r
            x_c = np.random.uniform(x_min_center, x_max_center)
            y_min_center = y_bounds[0] + r
            y_max_center = y_bounds[1] - r
            y_c = np.random.uniform(y_min_center, y_max_center)
            print(f"  Obstacle {i+1}: Generated random center ({x_c:.3f}, {y_c:.3f})")
        
        omega = np.random.uniform(omega_range[0], omega_range[1])
        phi = np.random.uniform(0, 2 * np.pi)
        noise_std = np.random.uniform(noise_range[0], noise_range[1])
        
        params = ObstacleTrajectoryParams(r=r, omega=omega, phi=phi, x_c=x_c, y_c=y_c, noise_std=noise_std)
        params_list.append(params)
        
        print(f"  Obstacle {i+1}: r={r:.3f}, ω={omega:.2f}, φ={phi:.2f}, center=({x_c:.3f}, {y_c:.3f}), noise={noise_std:.3f}")
    
    return params_list


def generate_random_initial_positions(num_obstacles: int = 6,
                                    seed: int = 43,
                                    x_bounds: Tuple[float, float] = (0.2, 0.8),
                                    y_bounds: Tuple[float, float] = (-0.4, 0.4)) -> List[np.ndarray]:
    """Generate random initial positions within workspace bounds"""
    np.random.seed(seed)
    
    print(f"Generating {num_obstacles} random initial positions with seed {seed}")
    
    initial_positions = []
    for i in range(num_obstacles):
        x_init = np.random.uniform(x_bounds[0], x_bounds[1])
        y_init = np.random.uniform(y_bounds[0], y_bounds[1])
        pos = np.array([x_init, y_init])
        initial_positions.append(pos)
        print(f"  Obstacle {i+1} initial position: ({x_init:.3f}, {y_init:.3f})")
    
    return initial_positions


def generate_given_initial_positions(num_obstacles: int = 6,
                                    given_positions: dict = None,
                                    x_bounds: Tuple[float, float] = (0.2, 0.8),
                                    y_bounds: Tuple[float, float] = (-0.4, 0.4),
                                    seed: int = 43) -> List[np.ndarray]:
    """
    Generate initial positions from given positions dictionary
    
    Args:
        num_obstacles: Number of obstacles
        given_positions: Dictionary with predefined positions {'obstacle_1': [x, y], ...}
        x_bounds: X-axis bounds (min_x, max_x) for fallback random generation
        y_bounds: Y-axis bounds (min_y, max_y) for fallback random generation
        seed: Random seed for fallback positions
    
    Returns:
        List of initial positions as numpy arrays
    """
    
    print(f"Generating {num_obstacles} initial positions from given positions")
    print(f"  - Using given positions: {given_positions is not None}")
    
    if given_positions is None:
        print("  ⚠️  No given positions provided, falling back to random generation")
        return generate_random_initial_positions(num_obstacles, seed, x_bounds, y_bounds)
    
    initial_positions = []
    np.random.seed(seed)  # For fallback random generation if needed
    
    for i in range(num_obstacles):
        # Try to get position from given_positions dictionary
        obstacle_keys = list(given_positions.keys())
        
        if i < len(obstacle_keys):
            # Use the i-th key from the dictionary
            key = obstacle_keys[i]
            if key in given_positions:
                x_init, y_init = given_positions[key]
                print(f"  Obstacle {i+1}: Using given initial position ({x_init:.3f}, {y_init:.3f}) from key '{key}'")
            else:
                # This shouldn't happen, but fallback just in case
                x_init = np.random.uniform(x_bounds[0], x_bounds[1])
                y_init = np.random.uniform(y_bounds[0], y_bounds[1])
                print(f"  Obstacle {i+1}: Key '{key}' not found, using random position ({x_init:.3f}, {y_init:.3f})")
        else:
            # Not enough given positions, generate random position
            x_init = np.random.uniform(x_bounds[0], x_bounds[1])
            y_init = np.random.uniform(y_bounds[0], y_bounds[1])
            print(f"  Obstacle {i+1}: Not enough given positions, using random position ({x_init:.3f}, {y_init:.3f})")
        
        # Ensure position is within bounds (safety check)
        x_init = np.clip(x_init, x_bounds[0], x_bounds[1])
        y_init = np.clip(y_init, y_bounds[0], y_bounds[1])
        
        pos = np.array([x_init, y_init])
        initial_positions.append(pos)
    
    print(f"✅ Generated {len(initial_positions)} initial positions from given data")
    return initial_positions


def generate_static_trajectory_params(num_obstacles: int = 6, 
                                    seed: int = 42,
                                    x_bounds: Tuple[float, float] = (0.2, 0.8),
                                    y_bounds: Tuple[float, float] = (-0.4, 0.4),
                                    noise_range: Tuple[float, float] = (0.0, 0.02),
                                    given_positions: dict = None) -> List[ObstacleTrajectoryParams]:
    """
    Generate static trajectory parameters where obstacles don't move
    
    Args:
        num_obstacles: Number of obstacles to generate parameters for
        seed: Random seed for reproducibility
        x_bounds: (min_x, max_x) workspace bounds for x-axis
        y_bounds: (min_y, max_y) workspace bounds for y-axis
        noise_range: (min_noise, max_noise) range for noise standard deviation
        given_positions: Dictionary with predefined positions {'obstacle_1': [x, y], ...}
    
    Returns:
        List of ObstacleTrajectoryParams with omega=0 (static obstacles)
    """
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    print(f"Generating {num_obstacles} STATIC trajectory parameters with seed {seed}")
    print(f"Constraints:")
    print(f"  - X bounds: {x_bounds}")
    print(f"  - Y bounds: {y_bounds}")
    print(f"  - Angular velocity: 0.0 (STATIC)")
    print(f"  - Noise range: {noise_range}")
    print(f"  - Using given positions: {given_positions is not None}")
    
    params_list = []
    
    for i in range(num_obstacles):
        # For static obstacles, we set:
        # - omega = 0 (no movement)
        # - r = 0 (no circular motion, just stay at center)
        # - x_c, y_c = the static position
        
        if given_positions is not None:
            # Use given positions from YAML
            key = list(given_positions.keys())[i] if i < len(given_positions) else f'obstacle_{i+1}'
            if key in given_positions:
                x_c, y_c = given_positions[key]
                print(f"  Obstacle {i+1}: Using given position ({x_c:.3f}, {y_c:.3f})")
            else:
                # Fallback to random if not enough given positions
                x_c = np.random.uniform(x_bounds[0], x_bounds[1])
                y_c = np.random.uniform(y_bounds[0], y_bounds[1])
                print(f"  Obstacle {i+1}: Generated random position ({x_c:.3f}, {y_c:.3f}) - not enough given positions")
        else:
            # Generate random static position within bounds
            x_c = np.random.uniform(x_bounds[0], x_bounds[1])
            y_c = np.random.uniform(y_bounds[0], y_bounds[1])
            print(f"  Obstacle {i+1}: Generated random position ({x_c:.3f}, {y_c:.3f})")
        
        # Static parameters
        r = 0.0          # No radius for static obstacles
        omega = 0.0      # No angular velocity (STATIC)
        phi = 0.0        # Phase doesn't matter for static obstacles
        
        # Generate random noise (can still have some noise even if static)
        noise_std = np.random.uniform(noise_range[0], noise_range[1])
        
        params = ObstacleTrajectoryParams(
            r=r,
            omega=omega,
            phi=phi,
            x_c=x_c,
            y_c=y_c,
            noise_std=noise_std
        )
        
        params_list.append(params)
        
        print(f"  Obstacle {i+1}: STATIC at position ({x_c:.3f}, {y_c:.3f}), noise={noise_std:.3f}")
    
    return params_list


def generate_mixed_trajectory_params(num_obstacles: int = 6,
                                   num_static: int = 3,
                                   seed: int = 42,
                                   x_bounds: Tuple[float, float] = (0.2, 0.8),
                                   y_bounds: Tuple[float, float] = (-0.4, 0.4),
                                   r_range: Tuple[float, float] = (0.05, 0.15),
                                   omega_range: Tuple[float, float] = (1.0, 3.0),
                                   noise_range: Tuple[float, float] = (0.0, 0.05),
                                   given_positions: dict = None) -> List[ObstacleTrajectoryParams]:
    """
    Generate mixed trajectory parameters with both moving and static obstacles
    
    Args:
        num_obstacles: Total number of obstacles
        num_static: Number of static obstacles (rest will be moving)
        seed: Random seed for reproducibility
        x_bounds: (min_x, max_x) workspace bounds for x-axis
        y_bounds: (min_y, max_y) workspace bounds for y-axis
        r_range: (min_radius, max_radius) range for trajectory radius (moving obstacles)
        omega_range: (min_omega, max_omega) range for angular velocity (moving obstacles)
        noise_range: (min_noise, max_noise) range for noise standard deviation
        given_positions: Dictionary with predefined positions for static obstacles
    
    Returns:
        List of ObstacleTrajectoryParams with mixed static and moving obstacles
    """
    
    if num_static > num_obstacles:
        raise ValueError(f"num_static ({num_static}) cannot be greater than num_obstacles ({num_obstacles})")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    num_moving = num_obstacles - num_static
    
    print(f"Generating {num_obstacles} MIXED trajectory parameters with seed {seed}")
    print(f"  - {num_static} STATIC obstacles")
    print(f"  - {num_moving} MOVING obstacles")
    print(f"Constraints:")
    print(f"  - X bounds: {x_bounds}")
    print(f"  - Y bounds: {y_bounds}")
    print(f"  - Radius range (moving): {r_range}")
    print(f"  - Angular velocity range (moving): {omega_range}")
    print(f"  - Noise range: {noise_range}")
    print(f"  - Using given positions: {given_positions is not None}")
    
    params_list = []
    
    # Generate static obstacles first
    for i in range(num_static):
        if given_positions is not None:
            # Use given positions from YAML
            key = list(given_positions.keys())[i] if i < len(given_positions) else f'obstacle_{i+1}'
            if key in given_positions:
                x_c, y_c = given_positions[key]
            else:
                x_c = np.random.uniform(x_bounds[0], x_bounds[1])
                y_c = np.random.uniform(y_bounds[0], y_bounds[1])
        else:
            # Generate random static position
            x_c = np.random.uniform(x_bounds[0], x_bounds[1])
            y_c = np.random.uniform(y_bounds[0], y_bounds[1])
        
        # Static parameters
        r = 0.0
        omega = 0.0
        phi = 0.0
        noise_std = np.random.uniform(noise_range[0], noise_range[1])
        
        params = ObstacleTrajectoryParams(
            r=r, omega=omega, phi=phi, x_c=x_c, y_c=y_c, noise_std=noise_std
        )
        params_list.append(params)
        
        print(f"  Obstacle {i+1}: STATIC at position ({x_c:.3f}, {y_c:.3f}), noise={noise_std:.3f}")
    
    # Generate moving obstacles
    for i in range(num_static, num_obstacles):
        # Generate random radius
        r = np.random.uniform(r_range[0], r_range[1])
        
        # Generate center coordinates ensuring the circle stays within bounds
        x_min_center = x_bounds[0] + r
        x_max_center = x_bounds[1] - r
        x_c = np.random.uniform(x_min_center, x_max_center)
        
        y_min_center = y_bounds[0] + r
        y_max_center = y_bounds[1] - r
        y_c = np.random.uniform(y_min_center, y_max_center)
        
        # Generate random angular velocity and phase
        omega = np.random.uniform(omega_range[0], omega_range[1])
        phi = np.random.uniform(0, 2 * np.pi)
        noise_std = np.random.uniform(noise_range[0], noise_range[1])
        
        params = ObstacleTrajectoryParams(
            r=r, omega=omega, phi=phi, x_c=x_c, y_c=y_c, noise_std=noise_std
        )
        params_list.append(params)
        
        print(f"  Obstacle {i+1}: MOVING r={r:.3f}, ω={omega:.2f}, φ={phi:.2f}, "
              f"center=({x_c:.3f}, {y_c:.3f}), noise={noise_std:.3f}")
    
    return params_list


def create_static_trajectories(seed: int = None, 
                             num_obstacles: int = 6,
                             x_bounds: Tuple[float, float] = (0.2, 0.8),
                             y_bounds: Tuple[float, float] = (-0.4, 0.4),
                             given_positions: dict = None) -> Tuple[List[ObstacleTrajectoryParams], List[np.ndarray]]:
    """
    Convenience function to create static trajectories
    
    Args:
        seed: Random seed (if None, a random seed will be used)
        num_obstacles: Number of obstacles
        x_bounds: X-axis bounds (min_x, max_x)
        y_bounds: Y-axis bounds (min_y, max_y)
        given_positions: Dictionary with predefined positions
    
    Returns:
        Tuple of (trajectory_params, initial_positions)
    """
    
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    print(f"\n🛑 Creating STATIC trajectories with seed: {seed}")
    print("=" * 50)
    
    # Generate static parameters
    static_trajectory_params = generate_static_trajectory_params(
        num_obstacles=num_obstacles,
        seed=seed,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        noise_range=(0.0, 0.02),  # Low noise for static obstacles
        given_positions=given_positions
    )
    
    # For static obstacles, initial positions are the same as the static positions
    static_initial_positions = []
    for params in static_trajectory_params:
        static_initial_positions.append(np.array([params.x_c, params.y_c]))
    
    return static_trajectory_params, static_initial_positions


class PreGeneratedTrajectoryController:
    """Controller that pre-generates entire trajectories for maximum efficiency"""
    
    def __init__(self, dt: float = 0.01, k_p: float = 5.0, k_d: float = 0.5):
        self.dt = dt
        self.k_p = k_p
        self.k_d = k_d
        
    def generate_reference_trajectory(self, params: ObstacleTrajectoryParams, time_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-generate the entire reference trajectory (feedforward part)"""
        angles = params.omega * time_array + params.phi
        x_ref = params.x_c + params.r * np.cos(angles)
        y_ref = params.y_c + params.r * np.sin(angles)
        positions = np.column_stack([x_ref, y_ref])
        dx_ref = -params.r * params.omega * np.sin(angles)
        dy_ref = params.r * params.omega * np.cos(angles)
        velocities = np.column_stack([dx_ref, dy_ref])
        return positions, velocities
    
    def generate_random_noise_trajectory(self, noise_std: float, num_steps: int, seed: Optional[int] = None) -> np.ndarray:
        """Pre-generate entire random noise trajectory"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(0, noise_std, size=(num_steps, 2))
    
    def simulate_trajectory(self, params: ObstacleTrajectoryParams, time_array: np.ndarray, 
                          initial_position: np.ndarray, noise_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate entire trajectory with pre-generated references and feedback control"""
        num_steps = len(time_array)
        ref_positions, ref_velocities = self.generate_reference_trajectory(params, time_array)
        noise_trajectory = self.generate_random_noise_trajectory(params.noise_std, num_steps, noise_seed)
        
        actual_positions = np.zeros((num_steps, 2))
        actual_velocities = np.zeros((num_steps, 2))
        control_inputs = np.zeros((num_steps, 2))
        
        actual_positions[0] = initial_position.copy()
        actual_velocities[0] = np.zeros(2)
        
        for i in range(num_steps - 1):
            current_pos = actual_positions[i]
            current_vel = actual_velocities[i]
            ref_pos = ref_positions[i]
            ref_vel = ref_velocities[i]
            pos_error = ref_pos - current_pos
            vel_error = ref_vel - current_vel
            control = ref_vel + self.k_p * pos_error + self.k_d * vel_error + noise_trajectory[i]
            control_inputs[i] = control
            actual_velocities[i + 1] = control
            actual_positions[i + 1] = current_pos + control * self.dt
        
        control_inputs[-1] = control_inputs[-2]  # Copy last control
        return actual_positions, control_inputs


def save_trajectories_as_array(pos_hist, ctrl_hist, save_path=None):
    """Convert position and control histories to structured numpy array"""
    obstacle_names = sorted(pos_hist.keys())
    num_obstacles = len(obstacle_names)
    num_timesteps = pos_hist[obstacle_names[0]].shape[0]
    
    print(f"Converting trajectories to structured array:")
    print(f"  - Number of obstacles: {num_obstacles}")
    print(f"  - Number of timesteps: {num_timesteps}")
    
    batch_size = 1
    num_features = 4  # x_pos, y_pos, u_x, u_y
    trajectory_array = np.zeros((batch_size, num_obstacles, num_features, num_timesteps))
    
    for i, obstacle_name in enumerate(obstacle_names):
        positions = pos_hist[obstacle_name]
        controls = ctrl_hist[obstacle_name]
        trajectory_array[0, i, 0, :] = positions[:, 0]  # x_position
        trajectory_array[0, i, 1, :] = positions[:, 1]  # y_position
        trajectory_array[0, i, 2, :] = controls[:, 0]   # u_x
        trajectory_array[0, i, 3, :] = controls[:, 1]   # u_y
        print(f"  - {obstacle_name}: ✓")
    
    print(f"Final array shape: {trajectory_array.shape}")
    
    if save_path:
        np.save(save_path, trajectory_array)
        print(f"Array saved to: {save_path}")
    
    return trajectory_array


def plot_optimized_results(time_array, all_positions, all_controls, method_name="Generated Trajectories", save_path=None):
    """Plot results from trajectory generation"""
    
    plt.figure(figsize=(15, 10))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    
    # Plot trajectories
    plt.subplot(2, 3, 1)
    for i, (name, positions) in enumerate(all_positions.items()):
        plt.plot(positions[:, 0], positions[:, 1], 
                color=colors[i], label=name, linewidth=2)
        plt.plot(positions[0, 0], positions[0, 1], 
                'o', color=colors[i], markersize=8, markeredgecolor='black')
    
    # Load and plot existing dataset trajectories for comparison if available
    data_directory = path_str + '/d3il/environments/dataset/data/avoiding/data/'
    if os.path.exists(data_directory):
        state_files = os.listdir(data_directory)
        files_to_plot = state_files

        for file in files_to_plot:
            with open(os.path.join(data_directory, file), 'rb') as f:
                env_state = pickle.load(f)
                robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
                robot_c_pos = env_state['robot']['c_pos'][:, :2]
                
                # Plot desired trajectory
                plt.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], 'b-', alpha=0.1, label='Desired trajectory' if file == files_to_plot[0] else "")
                
                # Plot actual trajectory
                plt.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], 'r-', alpha=0.1, label='Actual trajectory' if file == files_to_plot[0] else "")
                
                # Plot start points
                plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^', label='Start points' if file == files_to_plot[0] else "")
                
                # Plot end points
                plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x', label='End points' if file == files_to_plot[0] else "")

    plt.xlim(0.2, 0.8)
    plt.ylim(-0.4, 0.4)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'{method_name} - Dynamic Obstacle Trajectories')
    # plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot X positions over time
    plt.subplot(2, 3, 2)
    for i, (name, positions) in enumerate(all_positions.items()):
        plt.plot(time_array, positions[:, 0], color=colors[i], label=name, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position')
    plt.title('X Positions vs Time')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    
    # Plot Y positions over time
    plt.subplot(2, 3, 3)
    for i, (name, positions) in enumerate(all_positions.items()):
        plt.plot(time_array, positions[:, 1], color=colors[i], label=name, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    plt.title('Y Positions vs Time')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    
    # Plot controls for first obstacle
    plt.subplot(2, 3, 4)
    first_controls = all_controls['obstacle_1']
    plt.plot(time_array, first_controls[:, 0], 'r-', label='u_x', linewidth=2)
    plt.plot(time_array, first_controls[:, 1], 'b-', label='u_y', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input')
    plt.title('Control Inputs (Obstacle 1)')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    
    # Plot speeds
    plt.subplot(2, 3, 5)
    for i, (name, controls) in enumerate(all_controls.items()):
        speeds = np.linalg.norm(controls, axis=1)
        plt.plot(time_array, speeds, color=colors[i], label=name, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.title('Obstacle Speeds')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    
    # Performance summary
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f"Method: {method_name}", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f"Time steps: {len(time_array)}", fontsize=10)
    plt.text(0.1, 0.6, f"Obstacles: {len(all_positions)}", fontsize=10)
    plt.text(0.1, 0.5, "✅ Random trajectories", fontsize=10, color='green')
    plt.text(0.1, 0.4, "✅ Vectorized computations", fontsize=10, color='green')
    plt.text(0.1, 0.3, "✅ Configurable seeds", fontsize=10, color='green')
    
    # Add trajectory info
    max_speed = max([np.max(np.linalg.norm(controls, axis=1)) for controls in all_controls.values()])
    min_speed = min([np.min(np.linalg.norm(controls, axis=1)) for controls in all_controls.values()])
    plt.text(0.1, 0.2, f"Speed range: {min_speed:.3f}-{max_speed:.3f}", fontsize=9)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Generation Summary')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Trajectory plot saved to: {save_path}")
    else:
        plt.savefig('generated_trajectories.png')
        print(f"📊 Trajectory plot saved to: generated_trajectories.png")
    
    plt.close()  # Close to save memory


def generate_and_save_trajectories(trajectory_seed: int = 42, save_path: str = "generated_trajectories.npy", 
                                  plot_trajectories: bool = True, trajectory_type: str = 'dynamic',
                                  trajectory_pos: str = 'random', given_positions: dict = None,
                                  num_static: int = 3, dynamic_initial_pos: str = 'start_in_random_position',
                                  dynamic_end_pos: str = 'end_in_random_position'):
    """
    Generate trajectories and save them to file with support for static/mixed/dynamic trajectories
    
    Args:
        trajectory_seed: Random seed for reproducible generation
        save_path: Path to save trajectory file
        plot_trajectories: Whether to generate trajectory plots
        trajectory_type: 'static', 'dynamic', or 'mixed'
        trajectory_pos: Legacy position mode ('given' or 'random')
        given_positions: Dictionary with predefined positions
        num_static: Number of static obstacles (for mixed trajectories)
        dynamic_initial_pos: 'start_in_given_position' or 'start_in_random_position'
        dynamic_end_pos: 'end_in_given_position' or 'end_in_random_position'
    """
    print(f"\n🚀 Generating {trajectory_type} trajectories with seed {trajectory_seed}...")
    if trajectory_type == 'dynamic':
        print(f"📍 Initial position mode: {dynamic_initial_pos}")
        print(f"🎯 End position mode: {dynamic_end_pos}")
    else:
        print(f"📍 Position mode: {trajectory_pos}")
    print("=" * 60)
    
    # Choose trajectory generation method based on configuration
    if trajectory_type == 'static':
        print("🛑 Generating STATIC trajectories...")
        if trajectory_pos == 'given' and given_positions is not None:
            print(f"📍 Using given positions from YAML: {list(given_positions.keys())}")
            trajectory_params = generate_static_trajectory_params(
                num_obstacles=6,
                seed=trajectory_seed,
                x_bounds=(0.2, 0.8),
                y_bounds=(-0.4, 0.4),
                noise_range=(0.0, 0.02),
                given_positions=given_positions
            )
            # For static obstacles with given positions, initial positions should match the given positions
            print("📍 Setting initial positions to match given static positions")
            initial_positions = []
            for params in trajectory_params:
                initial_positions.append(np.array([params.x_c, params.y_c]))
        else:
            print("📍 Using random positions for static obstacles")
            trajectory_params = generate_static_trajectory_params(
                num_obstacles=6,
                seed=trajectory_seed,
                x_bounds=(0.2, 0.8),
                y_bounds=(-0.4, 0.4),
                noise_range=(0.0, 0.02),
                given_positions=None
            )
        
        # For static obstacles, initial positions match static positions
        initial_positions = []
        for params in trajectory_params:
            initial_positions.append(np.array([params.x_c, params.y_c]))
            
    elif trajectory_type == 'mixed':
        print(f"🔄 Generating MIXED trajectories ({num_static} static, {6-num_static} moving)...")
        if trajectory_pos == 'given' and given_positions is not None:
            print(f"📍 Using given positions for static obstacles from YAML")
            trajectory_params = generate_mixed_trajectory_params(
                num_obstacles=6,
                num_static=num_static,
                seed=trajectory_seed,
                x_bounds=(0.2, 0.8),
                y_bounds=(-0.4, 0.4),
                r_range=(0.05, 0.15),
                omega_range=(1.0, 3.0),
                noise_range=(0.0, 0.05),
                given_positions=given_positions
            )
        else:
            print("📍 Using random positions for static obstacles")
            trajectory_params = generate_mixed_trajectory_params(
                num_obstacles=6,
                num_static=num_static,
                seed=trajectory_seed,
                x_bounds=(0.2, 0.8),
                y_bounds=(-0.4, 0.4),
                r_range=(0.05, 0.15),
                omega_range=(1.0, 3.0),
                noise_range=(0.0, 0.05),
                given_positions=None
            )
        
        # Generate initial positions (different seed for extra randomness)
        initial_positions = generate_random_initial_positions(
            num_obstacles=6,
            seed=trajectory_seed + 1,
            x_bounds=(0.2, 0.8),
            y_bounds=(-0.4, 0.4)
        )
    else:  # dynamic (original behavior)
        print("🌀 Generating DYNAMIC trajectories...")
        print(f"📍 Initial position mode: {dynamic_initial_pos}")
        print(f"🎯 End position mode: {dynamic_end_pos}")
        
        # Generate trajectory parameters based on end position configuration
        if dynamic_end_pos == 'end_in_given_position' and given_positions is not None:
            print("🎯 Using given center positions for circular trajectories")
            trajectory_params = generate_random_trajectory_params(
                num_obstacles=6, 
                seed=trajectory_seed,
                x_bounds=(0.2, 0.8), 
                y_bounds=(-0.4, 0.4),
                r_range=(0.05, 0.15),
                omega_range=(1.0, 3.0),
                noise_range=(0.0, 0.00),
                given_positions=given_positions
            )
        else:
            print("🎯 Using random center positions for circular trajectories")
            trajectory_params = generate_random_trajectory_params(
                num_obstacles=6, 
                seed=trajectory_seed,
                x_bounds=(0.2, 0.8), 
                y_bounds=(-0.4, 0.4),
                r_range=(0.05, 0.15),
                omega_range=(1.0, 3.0),
                noise_range=(0.0, 0.05),
                given_positions=None
            )
        
        # Generate initial positions based on initial position configuration
        if dynamic_initial_pos == 'start_in_given_position' and given_positions is not None:
            print("📍 Using given initial positions")
            initial_positions = generate_given_initial_positions(
                num_obstacles=6,
                given_positions=given_positions,
                x_bounds=(0.2, 0.8),
                y_bounds=(-0.4, 0.4),
                seed=trajectory_seed + 1
            )
        else:
            print("📍 Using random initial positions")
            initial_positions = generate_random_initial_positions(
                num_obstacles=6,
                seed=trajectory_seed + 1,
                x_bounds=(0.2, 0.8),
                y_bounds=(-0.4, 0.4)
            )
    
    # Simulation parameters
    dt = 0.01
    T = 2.05
    time_array = np.arange(0, T, dt)
    
    print(f"Pre-computing {len(time_array)} steps for 6 obstacles...")
    
    # Create controller and generate trajectories
    controller = PreGeneratedTrajectoryController(dt=dt)
    results = {}
    
    for i, (params, init_pos) in enumerate(zip(trajectory_params, initial_positions)):
        obstacle_name = f"obstacle_{i+1}"
        actual_pos, controls = controller.simulate_trajectory(
            params, time_array, init_pos, noise_seed=i + trajectory_seed
        )
        results[obstacle_name] = {
            'positions': actual_pos,
            'controls': controls
        }
    
    # Convert to structured array and save
    pos_hist = {name: data['positions'] for name, data in results.items()}
    ctrl_hist = {name: data['controls'] for name, data in results.items()}
    
    trajectory_array = save_trajectories_as_array(pos_hist, ctrl_hist, save_path)
    
    # Plot trajectories if requested
    if plot_trajectories:
        plot_name = f"Generated {trajectory_type.capitalize()} Trajectories (seed={trajectory_seed})"
        plot_save_path = save_path.replace('.npy', '_plot.png')
        plot_optimized_results(time_array, pos_hist, ctrl_hist, plot_name, plot_save_path)
    
    print(f"✅ {trajectory_type.capitalize()} trajectory generation complete!")
    print(f"📊 Trajectory type: {trajectory_type}")
    
    if trajectory_type == 'dynamic':
        print(f"📍 Initial positions: {dynamic_initial_pos}")
        print(f"🎯 End positions: {dynamic_end_pos}")
        print("🌀 All obstacles are DYNAMIC")
    elif trajectory_type == 'static':
        print(f"📍 Position mode: {trajectory_pos}")
        print("🛑 All obstacles are STATIC (omega=0)")
    elif trajectory_type == 'mixed':
        print(f"📍 Position mode: {trajectory_pos}")
        print(f"🔄 {num_static} STATIC + {6-num_static} MOVING obstacles")
    
    return trajectory_array

# ==================== END TRAJECTORY GENERATION COMPONENTS ====================


def plot_each_timestep_enhanced(obs, itr, n_trial, variant, current_timestep=0, 
                               obstacle_trajectories=None, past_horizon=10, future_horizon=8):
    """
    Enhanced plotting function for publication-quality visualizations.
    Shows dynamic obstacles with past trajectory history and future horizon predictions.
    
    Args:
        obs: Robot observations
        itr: Iteration number
        n_trial: Trial number
        variant: Variant name
        current_timestep: Current timestep in the simulation
        obstacle_trajectories: Pre-generated obstacle trajectories [batch, obstacles, features, timesteps]
        past_horizon: Number of past steps to show with fading
        future_horizon: Number of future steps to show as prediction
    """
    obs = np.array(obs)
    
    # Create figure with publication-quality settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set axis limits and labels
    ax_limits = config['ax_limits'][exp]
    ax.set_xlim(ax_limits[0])
    ax.set_ylim(ax_limits[1])
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Dynamic Obstacle Avoidance with Trajectory Prediction', fontweight='bold', pad=20)
    
    # Robot current position
    ax.scatter(obs[:, 2], obs[:, 3], color='navy', s=50, marker='o', 
              label='Robot Position', zorder=10, edgecolors='black', linewidth=1)
    
    # Enhanced obstacle visualization
    if obstacle_trajectories is not None:
        plot_dynamic_obstacles_enhanced(ax, obstacle_trajectories, current_timestep, 
                                      past_horizon, future_horizon)
    else:
        # Fallback to static obstacle positions
        obstacle_positions = env.get_obstacle_position()
        for i, (obs_name, obs_pos) in enumerate(obstacle_positions.items()):
            ax.add_patch(plt.Circle(obs_pos[:2], 0.025, color='red', alpha=0.8, 
                                   label='Static Obstacles' if i == 0 else "", zorder=8))

    # Static obstacle constraints (CBF regions) - ESSENTIAL avoidance zones
    print(f"Plotting {len(obstacle_constraints)} obstacle constraints")  # Debug
    for i, constraint in enumerate(obstacle_constraints):
        pos_i = constraint['center']
        r_i = constraint['radius']
        print(f"  Constraint {i}: center={pos_i}, radius={r_i}")  # Debug
        # Make constraint zones more visible for publication
        circle = plt.Circle(pos_i, r_i, color='cornflowerblue', alpha=0.35, 
                   label='Avoidance Zones' if i == 0 else "", zorder=3)
        ax.add_patch(circle)

        # Add constraint boundary - matching existing codebase style
        boundary_circle = plt.Circle(pos_i, r_i, fill=False, color='royalblue', 
                           linestyle='--', alpha=0.9, linewidth=2, zorder=4)
        ax.add_patch(boundary_circle)

    # Goal line (finish line)
    ax.plot([0.2, 0.8], [0.35, 0.35], color='green', linewidth=6, 
           label='Goal Line', alpha=0.8, zorder=5)
    
    # Add goal markers
    ax.scatter([0.2, 0.8], [0.35, 0.35], color='green', s=100, marker='*', 
              zorder=6, edgecolors='darkgreen', linewidth=1)

    # Workspace boundaries and polytopic constraints (commented out to avoid dark appearance)
    # Large triangular constraint areas make the plot too dark for publication
    utils.plot_halfspace_constraints('avoiding-d3il', polytopic_constraints, ax, ax_limits)

    # Expert trajectories from dataset
    plot_expert_trajectories_enhanced(ax)
    
    # Enhanced legend
    # ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
    #          framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    
    ax.legend().set_visible(False)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add timestamp annotation
    # ax.text(0.02, 0.98, f'Time Step: {current_timestep}', transform=ax.transAxes, 
    #        fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
    #        facecolor='wheat', alpha=0.8))
    
    # Save with high quality
    os.makedirs(f'{path_str}/logs/{seed}/{variant}_{n_trial}', exist_ok=True)
    fig.savefig(f'{path_str}/logs/{seed}/{variant}_{n_trial}/obstacle_avoidance{itr}.png', 
               dpi=300, bbox_inches='tight', format='png', facecolor='white')
    plt.close(fig)


def plot_dynamic_obstacles_enhanced(ax, obstacle_trajectories, current_timestep, 
                                  past_horizon=10, future_horizon=8):
    """
    Plot dynamic obstacles with past history and future horizon.
    
    Args:
        ax: Matplotlib axis
        obstacle_trajectories: Array [batch, obstacles, features, timesteps]
        current_timestep: Current timestep
        past_horizon: Steps to show in the past
        future_horizon: Steps to show in the future
    """
    batch_size, num_obstacles, num_features, total_timesteps = obstacle_trajectories.shape
    
    # Bright and colorful palette for obstacles
    # obstacle_colors = [
    #     '#FF4500',  # Bright Orange Red
    #     '#1E90FF',  # Bright Blue  
    #     '#32CD32',  # Lime Green
    #     '#FF1493',  # Deep Pink
    #     '#FFD700',  # Gold
    #     '#8A2BE2',  # Blue Violet
    #     '#FF69B4',  # Hot Pink
    #     '#00CED1',  # Dark Turquoise
    # ]

    obstacle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    
    for obs_idx in range(num_obstacles):
        color = obstacle_colors[obs_idx % len(obstacle_colors)]
        
        # Extract positions for this obstacle
        x_positions = obstacle_trajectories[0, obs_idx, 0, :]  # x positions
        y_positions = obstacle_trajectories[0, obs_idx, 1, :]  # y positions
        
        # Ensure current_timestep is within bounds
        current_timestep =env.step_counter
        
        # Current position
        current_x = x_positions[current_timestep]
        current_y = y_positions[current_timestep]
        
        # Plot current obstacle position (larger, prominent)
        ax.add_patch(plt.Circle((current_x, current_y), 0.025, color=color, alpha=0.9, 
                               zorder=9, edgecolor='black', linewidth=1.5,
                               label=f'Dynamic Obstacles' if obs_idx == 0 else ""))
        
        # Plot past trajectory with fading effect
        past_start = max(0, current_timestep - past_horizon)
        if past_start < current_timestep:
            past_x = x_positions[past_start:current_timestep]
            past_y = y_positions[past_start:current_timestep]
            
            # Create fading effect for past trajectory
            for i in range(len(past_x) - 1):
                alpha = 0.1 + 0.4 * (i / max(1, len(past_x) - 1))  # Fade from 0.1 to 0.5
                ax.plot([past_x[i], past_x[i + 1]], [past_y[i], past_y[i + 1]], 
                       color=color, alpha=alpha, linewidth=2, zorder=3)
                
                # Small markers for past positions
                if i % 2 == 0:  # Every other point to avoid clutter
                    ax.scatter(past_x[i], past_y[i], color=color, alpha=alpha, 
                             s=10, zorder=4)
        
        # Plot future trajectory (prediction horizon)
        future_end = min(total_timesteps, current_timestep + future_horizon + 1)
        if current_timestep < future_end - 1:
            future_x = x_positions[current_timestep:future_end]
            future_y = y_positions[current_timestep:future_end]
            
            # Plot future trajectory with dashed line
            ax.plot(future_x, future_y, color=color, linestyle='--', alpha=0.7, 
                   linewidth=2.5, zorder=7, 
                   label='Future Horizon' if obs_idx == 0 else "")
            
            # Plot future predicted positions
            for i in range(1, len(future_x)):
                alpha = 0.7 - 0.1 * (i / max(1, len(future_x) - 1))  # Fade future predictions
                ax.scatter(future_x[i], future_y[i], color=color, alpha=alpha, 
                         s=25, marker='s', zorder=6, edgecolors='black', linewidth=0.5)
            
            # Highlight the end of prediction horizon
            if len(future_x) > 1:
                ax.scatter(future_x[-1], future_y[-1], color=color, alpha=0.6, 
                         s=40, marker='D', zorder=6, edgecolors='black', linewidth=1)
        
        # Add velocity vectors at current position (optional)
        if num_features >= 4:  # If we have control/velocity information
            vel_x = obstacle_trajectories[0, obs_idx, 2, current_timestep] * 0.1  # Scale for visibility
            vel_y = obstacle_trajectories[0, obs_idx, 3, current_timestep] * 0.1
            ax.arrow(current_x, current_y, vel_x, vel_y, head_width=0.01, 
                    head_length=0.01, fc=color, ec=color, alpha=0.6, zorder=8)


def plot_expert_trajectories_enhanced(ax):
    """
    Plot expert trajectories with enhanced styling for publication.
    """
    data_directory = path_str + '/d3il/environments/dataset/data/avoiding/data/'
    
    if not os.path.exists(data_directory):
        return
        
    state_files = os.listdir(data_directory)
    files_to_plot = state_files[:90]  # Limit to prevent overcrowding
    
    for i, file in enumerate(files_to_plot):
        try:
            with open(os.path.join(data_directory, file), 'rb') as f:
                env_state = pickle.load(f)
                
                robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
                robot_c_pos = env_state['robot']['c_pos'][:, :2]
                
                # Plot desired trajectory (thinner, more transparent)
                ax.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], color='lightcoral', 
                       alpha=0.2, linewidth=1, zorder=1,
                       label='Expert Desired' if i == 0 else "")
                
                # Plot actual trajectory (slightly thicker)
                # ax.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], color='lightcoral', 
                #        alpha=0.5, linewidth=1.5, zorder=2,
                #        label='Expert Actual' if i == 0 else "")
                
                # Plot start and end points for first trajectory only (to avoid clutter)
                if i == 0:
                    ax.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], 
                             c='green', s=80, marker='^', zorder=5, 
                             edgecolors='darkgreen', linewidth=1, label='Start Point')
                    ax.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], 
                             c='black', s=80, marker='X', zorder=5, 
                             edgecolors='white', linewidth=1, label='End Point')
        except Exception as e:
            print(f"Error loading trajectory file {file}: {e}")
            continue


# Update the original function to use the enhanced version
def plot_each_timestep(obs, itr, n_trial, variant):
    """
    Original function updated to use enhanced visualization.
    This function now loads and uses the pre-generated obstacle trajectories.
    """
    # Try to load obstacle trajectories
    trajectory_file = config.get('trajectory_file', 'generated_trajectories.npy')
    obstacle_trajectories = None
    
    try:
        if os.path.exists(trajectory_file):
            obstacle_trajectories = np.load(trajectory_file)
            print(f"Loaded obstacle trajectories from {trajectory_file}")
        else:
            print(f"Trajectory file {trajectory_file} not found, using static obstacles")
    except Exception as e:
        print(f"Error loading trajectory file: {e}")
    
    # Estimate current timestep (you may want to pass this as a parameter)
    # For now, we'll use a simple estimation based on iteration
    current_timestep = min(itr * 5, 190) if obstacle_trajectories is not None else 0
    
    # Call the enhanced plotting function
    plot_each_timestep_enhanced(obs, itr, n_trial, variant, current_timestep, 
                               obstacle_trajectories, past_horizon=12, future_horizon=8)




# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

# General
exps = config['exps']
seeds = config['seeds']
projection_variants = config['projection_variants']
halfspace_variants = config['avoiding_halfspace_variants'] if 'avoiding' in exps[0] else ['top-left']
n_trials = config['n_trials']
plot_how_many = config['plot_how_many']

# Constraint projection
constraint_types = config['constraint_types']

for exp in exps:
    for halfspace_variant in halfspace_variants:
        robot_name = exp.split('-')[0]
        if halfspace_variant == 'top-left-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][0]]
            obstacle_constraints = [config['obstacle_constraints'][exp][3]]
        elif halfspace_variant == 'top-right-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][1]]
            obstacle_constraints = [config['obstacle_constraints'][exp][4]]
        elif halfspace_variant == 'both-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][2], config['halfspace_constraints'][exp][3]]
            obstacle_constraints = [config['obstacle_constraints'][exp][5]]

        bounds = config['bounds'][exp]
        ax_limits = config['ax_limits'][exp]
        enlarge_constraints = config['enlarge_constraints'][robot_name]
        dt = config['dt'][robot_name]

        class Parser(utils.Parser):
            dataset: str = exp
            config: str = 'config.' + exp


        figs_all_seeds, axes_all_seeds = setup_all_seeds_figures(projection_variants)
        
        # ==================== TRAJECTORY GENERATION/LOADING ====================
        trajectory_file = config.get('trajectory_file', 'generated_trajectories.npy')
        generate_trajectories = config.get('generate_trajectories', True)
        trajectory_seed = config.get('trajectory_seed', 42)
        plot_trajectories = config.get('plot_generated_trajectories', True)
        
        # Read new trajectory configuration options
        trajectory_type = config.get('trajectory_type', 'dynamic')  # 'static', 'dynamic', or 'mixed'
        trajectory_pos = config.get('trajectory_pos', 'random')     # 'given' or 'random' (legacy)
        num_static = config.get('num_static', 3)                    # For mixed trajectories
        
        # Read dynamic obstacle position configuration
        dynamic_initial_pos = config.get('dynamic_obstacle_initial_position', 'start_in_random_position')
        dynamic_end_pos = config.get('dynamic_obstacle_end_position', 'end_in_random_position')
        
        # Get given obstacle positions from YAML (if available)
        given_obstacle_positions = None
        if trajectory_pos == 'given':
            given_positions_config = config.get('given_obstacle_positions', {})
            experiment_key = exp  # Use the current experiment name as key
            if experiment_key in given_positions_config:
                given_obstacle_positions = given_positions_config[experiment_key]
                print(f"🗺️  Loaded given obstacle positions for {experiment_key}:")
                for obs_name, pos in given_obstacle_positions.items():
                    print(f"   {obs_name}: {pos}")
            else:
                print(f"⚠️  No given positions found for {experiment_key}, falling back to random positions")
                trajectory_pos = 'random'  # Fallback to random if no positions found
        
        if generate_trajectories:
            print(f"\n🚀 Generating new trajectories with seed {trajectory_seed}...")
            print(f"   📊 Type: {trajectory_type}")
            print(f"   📍 Position mode: {trajectory_pos} (legacy)")
            print(f"   📍 Dynamic initial: {dynamic_initial_pos}")
            print(f"   🎯 Dynamic end: {dynamic_end_pos}")
            external_obstacle_pos = generate_and_save_trajectories(
                trajectory_seed=trajectory_seed, 
                save_path=trajectory_file,
                plot_trajectories=plot_trajectories,
                trajectory_type=trajectory_type,
                trajectory_pos=trajectory_pos,
                given_positions=given_obstacle_positions,
                num_static=num_static,
                dynamic_initial_pos=dynamic_initial_pos,
                dynamic_end_pos=dynamic_end_pos
            )
        else:
            print(f"\n📂 Loading existing trajectories from {trajectory_file}...")
            if os.path.exists(trajectory_file):
                external_obstacle_pos = np.load(trajectory_file)
                print(f"✅ Loaded trajectory file with shape: {external_obstacle_pos.shape}")
            else:
                print(f"❌ Trajectory file {trajectory_file} not found! Generating new trajectories...")
                external_obstacle_pos = generate_and_save_trajectories(
                    trajectory_seed=trajectory_seed, 
                    save_path=trajectory_file,
                    plot_trajectories=plot_trajectories,
                    trajectory_type=trajectory_type,
                    trajectory_pos=trajectory_pos,
                    given_positions=given_obstacle_positions,
                    num_static=num_static,
                    dynamic_initial_pos=dynamic_initial_pos,
                    dynamic_end_pos=dynamic_end_pos
                )
        all_results = {}
        # ==================== END TRAJECTORY GENERATION/LOADING ====================
        for seed in seeds:
            args = Parser().parse_args(experiment='plan', seed=seed)

            # Get model
            diffusion_experiment = utils.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), epoch=args.diffusion_epoch, device=args.device)
            diffusion = diffusion_experiment.diffusion
            dataset = diffusion_experiment.dataset

            if 'pointmaze' in exp or 'antmaze' in exp:
                minari_dataset = minari.load_dataset(exp, download=True)
                env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment
            elif 'avoiding' in exp:
                env = ObstacleAvoidanceEnv()
                env.start()

            if robot_name == 'pointmaze': env.env.env.env.point_env.frame_skip = 2
            if robot_name == 'antmaze': env.env.env.env.ant_env.frame_skip = 5

            obs_indices = config['observation_indices'][robot_name]
            act_indices = config['action_indices'][robot_name]

            # Create projector
            if diffusion.__class__.__name__ == 'GaussianDiffusion':
                trajectory_dim = diffusion.transition_dim - diffusion.goal_dim
                action_dim = diffusion.action_dim
                diffuser_variant = 'states_actions'
                obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
                act_obs_indices = {**act_indices, **obs_indices_updated}
            else:
                trajectory_dim = diffusion.observation_dim - diffusion.goal_dim
                action_dim = 0
                diffuser_variant = 'states'
                act_obs_indices = obs_indices

            # -------------------- Load constraints ------------------
            constraint_list = []
            constraint_list_tightened = []
            # Halfspace constraints
            constraint_list_polytopic_not_tightened = []
            if 'halfspace' in constraint_types:
                for constraint in polytopic_constraints:
                    constraint_list.append(('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))
                    constraint_list_tightened.append(('ineq', utils.formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices)))
                    constraint_list_polytopic_not_tightened.append(('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))

            # Bounds
            if 'bounds' in constraint_types:
                lower_bound, upper_bound = utils.formulate_bounds_constraints(constraint_types, bounds, trajectory_dim, act_obs_indices)
                constraint_list.extend([['lb', lower_bound], ['ub', upper_bound]])
                constraint_list_tightened.extend([['lb', lower_bound], ['ub', upper_bound]])

            # Obstacle constraints
            if 'obstacles' in constraint_types:
                for constr in obstacle_constraints:
                    constraint_list.append([constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]], constr['center'], constr['radius']])
                    constraint_list_tightened.append([constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]], constr['center'], constr['radius'] + enlarge_constraints])

            # Dynamics constraints
            constraint_list_without_prior = copy(constraint_list)
            constraint_list_without_prior_tightened = copy(constraint_list_tightened)
            dynamics_constraints = []
            if 'dynamics' in constraint_types: dynamics_constraints = utils.formulate_dynamics_constraints(exp, act_obs_indices, action_dim)

            for constraint in dynamics_constraints:
                constraint_list.append(constraint)
                constraint_list_tightened.append(constraint)

            # -------------------- Run experiments ------------------
            env_seeds = config['env_seeds'][exp] if 'pointmaze-umaze' in exp else np.arange(100)
            fig_all, ax_all = setup_figures(n_trials, plot_how_many, projection_variants)       

            compare_result = {}
            
            for variant_idx, variant in enumerate(projection_variants):
                print(f'------------------------Running {exp} - {halfspace_variant} - {variant} ({seed})----------------------------')

                gradient = True if 'gradient' in variant else False

                if 'model_free' in variant and 'tightened' in variant:
                    constraints = constraint_list_without_prior_tightened
                elif 'model_free' in variant and not 'tightened' in variant:
                    constraints = constraint_list_without_prior
                elif not 'model_free' in variant and 'tightened' in variant:
                    constraints = constraint_list_tightened
                elif not 'model_free' in variant and ('diffmpc' or 'diffmpc_project') in variant:
                    constraints = constraint_list_tightened                    
                else:
                    constraints = constraint_list

                delta_t = dt
                if 'dt0p25' in variant:
                    delta_t = 0.25 * dt
                elif 'dt0p5' in variant:
                    delta_t = 0.5 * dt
                elif 'dt2p0' in variant:
                    delta_t = 2.0 * dt
                elif 'dt4p0' in variant:
                    delta_t = 4.0 * dt

                # Create projector
                projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=diffusion.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer, 
                                        gradient=gradient, gradient_weights=[1, 0.5, 2], variant=diffuser_variant, dt=delta_t, cost_dims=None, device=args.device, solver='scipy', env=env, method=variant, obstacle_constraints=obstacle_constraints)
                projector = None if variant == 'diffuser' else projector

                trajectory_selection = 'random'
                if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
                if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
                if 'diffmpc' in variant or 'diffmpc_project' in variant: trajectory_selection = 'temporal_consistency'

                # Create policy
                policy = Policy(model=diffusion, normalizer=dataset.normalizer, preprocess_fns=args.preprocess_fns, 
                                test_ret=args.test_ret, projector=projector, trajectory_selection=trajectory_selection)    

                # Run policy
                fig, ax = plt.subplots(min(n_trials, plot_how_many), 6, figsize=(30, 5 * min(n_trials, plot_how_many)))
                ax = ax.reshape(-1, 6)
                fig.suptitle(f'{exp} - {variant}')

                save_samples_every = args.horizon // 2

                # Store a few sampled trajectories
                sampled_trajectories_all = []        
                n_success = np.zeros(n_trials)
                n_success_and_constraints = np.zeros(n_trials)
                n_steps = np.zeros(n_trials)
                n_violations = np.zeros(n_trials)
                total_violations = np.zeros(n_trials)
                avg_time = np.zeros(n_trials)
                collision_free_completed = np.ones(n_trials)
                pos_tracking_errors = np.zeros((n_trials, args.max_episode_length - 1))
                for i in range(n_trials):
                    torch.manual_seed(i)
                    env_seed = env_seeds[i] if ('pointmaze-umaze' in exp) else i
                    
                    # Reset environment
                    if 'avoiding' in exp:
                        obs = env.reset()
                        action = env.robot_state()[:2]
                        fixed_z = env.robot_state()[2:]
                    else:
                        obs, _ = env.reset(seed=env_seed)
                    
                    if 'pointmaze' in exp:
                        obs = np.concatenate((obs['observation'], obs['desired_goal']))
                    elif 'antmaze' in exp:
                        obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                    elif 'avoiding' in exp:
                        obs = np.concatenate((action[:2], obs))           
                        
                    obs_buffer = []
                    action_buffer = []

                    sampled_trajectories = []
                    disable_projection = False
                    for _ in range(args.max_episode_length):
                        # Check if a safety constraint is violated
                        violated_this_timestep = 0
                        if 'halfspace' in constraint_types:
                            for constraint in constraint_list_polytopic_not_tightened:
                                if constraint[0] == 'ineq':
                                    c, d = constraint[1]
                                    obs_to_check = obs[:-diffusion.goal_dim] if diffusion.goal_dim > 0 else obs
                                    if obs_to_check @ c[action_dim:] >= d:
                                        violated_this_timestep = 1
                                        total_violations[i] += obs_to_check @ c[action_dim:] - d
                                        collision_free_completed[i] = 0

                        if 'obstacles' in constraint_types:
                            for constraint in obstacle_constraints:
                                if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center']) < constraint['radius']:
                                    violated_this_timestep = 1
                                    total_violations[i] += constraint['radius'] - np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center'])
                                    collision_free_completed[i] = 0
                        
                        if _ > 0 and 'bounds' in constraint_types:
                            act_obs = np.concatenate((action, obs)) if action_dim > 0 else obs
                            total_violations[i] += np.sum(np.maximum(0, act_obs - upper_bound)) + np.sum(np.maximum(0, lower_bound - act_obs))

                        n_violations[i] += violated_this_timestep
                        
                        # Calculate action
                        start = time.time()
                        action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection, env=env, config_obstacle_constraints=obstacle_constraints, polytopic_constraints=polytopic_constraints,variant=variant, save_dir=f'{path_str}/logs/{seed}/{variant}_{i}')
                        avg_time[i] += time.time() - start

                        # Step environment
                        if 'avoiding' in exp:
                            next_pos_des = action + obs[:2] 
                            obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
                            # env.step(np.concatenate((0 * next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
                            obs = next_pos_des
                            success = info[1]
                            for n_obs in range(external_obstacle_pos.shape[1]):
                                external_pos = external_obstacle_pos[0, n_obs, :2, _]
                                name_of_obstacle = list(env.get_obstacle_position().keys())[n_obs]
                                env.set_obstacle_position(name_of_obstacle,np.concatenate([external_pos,[0]]))                            
                            
                            # Check for collisions with obstacles
                            obstacle_positions = env.get_obstacle_position()
                            for obstacle_id, obstacle_pos in obstacle_positions.items():
                                # Calculate distance between robot and obstacle center
                                obstacle_radius = 0.025  # Using the same radius as in plot_each_timestep
                                distance = np.linalg.norm(obs - obstacle_pos[:2])
                                if distance <= obstacle_radius:
                                    # Collision detected
                                    terminated = True
                                    success = False
                                    break
                                    
                            if obs[1] > 0.35 and not terminated:
                                success = True
                                terminated = True
                        else:
                            obs, rew, terminated, truncated, info = env.step(action)
                            success = info['success']

                        if 'pointmaze' in exp:
                            obs = np.concatenate((obs['observation'], obs['desired_goal']))
                        elif 'antmaze' in exp:
                            obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                        elif 'avoiding' in exp:
                            obs = np.concatenate((next_pos_des[:2], obs))

                        # Get tracking error
                        if _ >= 1:
                            pos_tracking_errors[i, _-1] = np.linalg.norm(obs[obs_indices['x']:obs_indices['y']+1] - desired_next_pos)
                        desired_next_pos = samples.observations[0, 1, [obs_indices['x'], obs_indices['y']]]

                        if _ % save_samples_every == 0:
                            sampled_trajectories.append(samples.observations[:, :, :])

                        obs_buffer.append(obs)
                        action_buffer.append(action)
                        if success: n_success[i] = 1
                        if (terminated or _ == args.max_episode_length - 1) and (not success): collision_free_completed[i] = 0
                        plot_each_timestep(obs_buffer, _, i, variant)
                        # if success or terminated or truncated or _ == n_timesteps - 1:
                        if success or terminated or _ == args.max_episode_length - 1:
                            n_steps[i] = _
                            avg_time[i] /= _
                            if success and collision_free_completed[i]: n_success_and_constraints[i] = 1
                            break

                    sampled_trajectories_all.append(sampled_trajectories)
                    if i >= plot_how_many:     # Plot only the first n trials
                        continue
                    # plot_states = ['x', 'y', 'vx', 'vy'] if 'maze' in exp else ['x', 'y']
                    
                    plot_trial_results(i, variant_idx, variant, obs_buffer, obs_indices, 
                                         sampled_trajectories_all, ax, ax_all, axes_all_seeds, 
                                         seed, seeds, colors=None, n_success=n_success, 
                                         collision_free_completed=collision_free_completed, 
                                         n_steps=n_steps, ax_limits=ax_limits, 
                                         constraint_types=constraint_types, 
                                         polytopic_constraints=polytopic_constraints, 
                                         obstacle_constraints=obstacle_constraints, args=args)
                    
                    # plot_states = ['x', 'y', 'x_des', 'y_des']

                    # for j in range(len(plot_states)):
                    #     ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                    #     ax[i, j].set_title(plot_states[j])
                    
                    # axes = [ax[i, 4], ax_all[i, variant_idx]]
                    # for curr_ax in axes:
                    #     curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], 'k')
                    #     curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']], np.array(obs_buffer)[0, obs_indices['y']], 'go', label='Start')            # Start
                    #     # if 'maze' in exp: curr_ax.plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
                    #     curr_ax.set_xlim(ax_limits[0])
                    #     curr_ax.set_ylim(ax_limits[1])

                    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                    # axes_all_seeds[variant_idx].plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], colors[seed % len(colors)], linewidth=2)
                    
                    # axes = [ax[i, 5], ax_all[i, variant_idx]]
                    # for __ in range(len(sampled_trajectories_all[i])):          # Iterate over timesteps of sampled trajectories
                    #     for ___ in range(min(args.batch_size, 4)):              # Iterate over batch
                    #         for curr_ax in axes:
                    #             curr_ax.plot(sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['x']], sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['y']], 'b')
                    #             curr_ax.plot(sampled_trajectories_all[i][__][___, 0, obs_indices['x']], sampled_trajectories_all[i][__][___, 0, obs_indices['y']], 'go', label='Start')    # Current state
                    # # if 'maze' in exp: ax[i, 5].plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
                    # ax[i, 5].set_xlim(ax_limits[0])
                    # ax[i, 5].set_ylim(ax_limits[1])

                    # # Plot constraints
                    # axes = [ax[i, 4], ax[i, 5], ax_all[i, variant_idx]]
                    # for curr_ax in axes: 
                    #     utils.plot_environment_constraints(exp, curr_ax)

                    #     if 'halfspace' in constraint_types: utils.plot_halfspace_constraints(exp, polytopic_constraints, curr_ax, ax_limits)

                    #     if 'obstacles' in constraint_types:
                    #         for constraint in obstacle_constraints:
                    #             curr_ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))


                all_results[variant + '_' + str(seed)] = {
                    'n_success': n_success,
                    'n_success_and_constraints': n_success_and_constraints,
                    'n_steps': n_steps,
                    'n_violations': n_violations,
                    'total_violations': total_violations,
                    'avg_time': avg_time,
                    'collision_free_completed': collision_free_completed,
                }
                print(f'Success rate: {np.mean(n_success)}')
                print(f'Constraints satisfied: {np.mean(collision_free_completed)}')
                print(f'Success rate (goal and constraints): {np.mean(n_success_and_constraints)}')
                print(f'Avg number of steps: {(np.mean(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f} +- {(np.std(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f}')
                print(f'Avg number of constraint violations: {np.mean(n_violations):.2f} +- {np.std(n_violations):.2f}')
                print(f'Avg total violation: {np.mean(total_violations):.3f} +- {np.std(total_violations):.3f}')
                print(f'Average computation time per step: {np.mean(avg_time):.3f}')
                if variant == 'diffuser': print(f'Tracking error: {np.max(pos_tracking_errors):.3f}')

                compare_result[variant] = {
                    'n_success': n_success,
                    'n_success_and_constraints': n_success_and_constraints,
                    'n_steps': n_steps,
                    'n_violations': n_violations,
                    'total_violations': total_violations,
                    'avg_time': avg_time,
                    'collision_free_completed': collision_free_completed
                }
                

                
                save_path = f'{args.savepath}/results/halfspace_{halfspace_variant}' if 'avoiding' in exp else f'{args.savepath}/results'
                
                
                os.makedirs(save_path, exist_ok=True)
                if config['write_to_file']:
                    np.savez(f'{save_path}/{variant}.npz', 
                            n_success=n_success, 
                            n_success_and_constraints=n_success_and_constraints,
                            n_steps=n_steps, 
                            n_violations=n_violations, 
                            total_violations=total_violations, 
                            avg_time=avg_time, 
                            collision_free_completed=collision_free_completed, 
                            args=args)

                fig.savefig(f'{save_path}/{variant}.png')   
                plt.close(fig)

                ax_all[0, variant_idx].set_title(variant)
                env.close()

            
            create_summary_visualization(compare_result, save_path)
            
            fig_all.savefig(f'{save_path}/all.png')
            plt.show()
        
        # Save all_results dictionary to logs folder
        import pickle
        import json
        from datetime import datetime
        
        # Create logs directory if it doesn't exist
        logs_dir = f'{path_str}/logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate filename with timestamp and experiment info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = exp.replace('-', '_')
        halfspace_name = halfspace_variant.replace('-', '_')
        
        # Save as pickle (preserves numpy arrays exactly)
        pickle_file = f'{logs_dir}/all_results_{exp_name}_{halfspace_name}_{timestamp}.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"✅ All results saved as pickle to: {pickle_file}")
        
        # Save as JSON (human-readable, but converts numpy arrays to lists)
        json_file = f'{logs_dir}/all_results_{exp_name}_{halfspace_name}_{timestamp}.json'
        json_results = {}
        for key, value in all_results.items():
            json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in value.items()}
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"✅ All results saved as JSON to: {json_file}")
        
        # Save as numpy archive (good for analysis)
        npz_file = f'{logs_dir}/all_results_{exp_name}_{halfspace_name}_{timestamp}.npz'
        np.savez(npz_file, **all_results)
        print(f"✅ All results saved as NPZ to: {npz_file}")
        
        # Print summary of what was saved
        print(f"\n📊 Summary of saved results:")
        print(f"   - Experiment: {exp}")
        print(f"   - Halfspace variant: {halfspace_variant}")
        print(f"   - Number of configurations: {len(all_results)}")
        print(f"   - Configurations: {list(all_results.keys())}")
        print(f"   - Data keys per configuration: {list(next(iter(all_results.values())).keys())}")
        print(f"   - Timing data included: ✅ avg_time (computation time per step for each method)")
        print(f"   - Additional metrics: success rates, constraint violations, collision data")



        variant_idx = 0
        path = f'{os.path.dirname(args.savepath)}/all_seeds/{halfspace_variant}'
        os.makedirs(path, exist_ok=True)
        for fig, ax in zip(figs_all_seeds, axes_all_seeds):
            ax.set_xlim(ax_limits[0])
            ax.set_ylim(ax_limits[1])
            ax.set_facecolor([1, 1, 0.9])
            utils.plot_environment_constraints(exp, ax)
            if 'halfspace' in constraint_types: utils.plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits, enlarge_constraints=enlarge_constraints)
            if 'obstacles' in constraint_types:
                for constraint in obstacle_constraints:
                    ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))
                    ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'] + enlarge_constraints, color='b', alpha=0.1, linestyle='--'))
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.png', bbox_inches='tight')
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.pdf', bbox_inches='tight', format='pdf')
            variant_idx += 1
        