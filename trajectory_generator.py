import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os,pickle


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
                                    noise_range: Tuple[float, float] = (0.0, 0.05)) -> List[ObstacleTrajectoryParams]:
    """
    Generate random trajectory parameters within specified constraints
    
    Args:
        num_obstacles: Number of obstacles to generate parameters for
        seed: Random seed for reproducibility
        x_bounds: (min_x, max_x) workspace bounds for x-axis
        y_bounds: (min_y, max_y) workspace bounds for y-axis
        r_range: (min_radius, max_radius) range for trajectory radius
        omega_range: (min_omega, max_omega) range for angular velocity
        noise_range: (min_noise, max_noise) range for noise standard deviation
    
    Returns:
        List of ObstacleTrajectoryParams with random values within constraints
    """
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    print(f"Generating {num_obstacles} random trajectory parameters with seed {seed}")
    print(f"Constraints:")
    print(f"  - X bounds: {x_bounds}")
    print(f"  - Y bounds: {y_bounds}")
    print(f"  - Radius range: {r_range}")
    print(f"  - Angular velocity range: {omega_range}")
    print(f"  - Noise range: {noise_range}")
    
    params_list = []
    
    for i in range(num_obstacles):
        # Generate random radius
        r = np.random.uniform(r_range[0], r_range[1])
        
        # Generate center coordinates ensuring the circle stays within bounds
        # x_c must be in [x_min + r, x_max - r] to keep circle within x_bounds
        x_min_center = x_bounds[0] + r
        x_max_center = x_bounds[1] - r
        x_c = np.random.uniform(x_min_center, x_max_center)
        
        # y_c must be in [y_min + r, y_max - r] to keep circle within y_bounds
        y_min_center = y_bounds[0] + r
        y_max_center = y_bounds[1] - r
        y_c = np.random.uniform(y_min_center, y_max_center)
        
        # Generate random angular velocity
        omega = np.random.uniform(omega_range[0], omega_range[1])
        
        # Generate random phase offset (0 to 2*pi)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Generate random noise
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
        
        print(f"  Obstacle {i+1}: r={r:.3f}, ω={omega:.2f}, φ={phi:.2f}, "
              f"center=({x_c:.3f}, {y_c:.3f}), noise={noise_std:.3f}")
    
    return params_list


def generate_random_initial_positions(num_obstacles: int = 6,
                                    seed: int = 43,
                                    x_bounds: Tuple[float, float] = (0.2, 0.8),
                                    y_bounds: Tuple[float, float] = (-0.4, 0.4)) -> List[np.ndarray]:
    """
    Generate random initial positions within workspace bounds
    
    Args:
        num_obstacles: Number of initial positions to generate
        seed: Random seed for reproducibility (different from trajectory params)
        x_bounds: (min_x, max_x) workspace bounds for x-axis
        y_bounds: (min_y, max_y) workspace bounds for y-axis
    
    Returns:
        List of numpy arrays representing initial positions
    """
    
    # Use different seed for initial positions to add more randomness
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


def generate_static_trajectory_params(num_obstacles: int = 6, 
                                    seed: int = 42,
                                    x_bounds: Tuple[float, float] = (0.2, 0.8),
                                    y_bounds: Tuple[float, float] = (-0.4, 0.4),
                                    noise_range: Tuple[float, float] = (0.0, 0.02)) -> List[ObstacleTrajectoryParams]:
    """
    Generate static trajectory parameters where obstacles don't move
    
    Args:
        num_obstacles: Number of obstacles to generate parameters for
        seed: Random seed for reproducibility
        x_bounds: (min_x, max_x) workspace bounds for x-axis
        y_bounds: (min_y, max_y) workspace bounds for y-axis
        noise_range: (min_noise, max_noise) range for noise standard deviation
    
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
    
    params_list = []
    
    for i in range(num_obstacles):
        # For static obstacles, we set:
        # - omega = 0 (no movement)
        # - r = 0 (no circular motion, just stay at center)
        # - x_c, y_c = the static position
        


        
        # Generate random static position within bounds
        # x_c = np.random.uniform(x_bounds[0], x_bounds[1])
        # y_c = np.random.uniform(y_bounds[0], y_bounds[1])

        key = list(given_obstacle_pos.keys())[i]
        x_c = given_obstacle_pos[key][0]
        y_c = given_obstacle_pos[key][1]
        
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
                                   noise_range: Tuple[float, float] = (0.0, 0.05)) -> List[ObstacleTrajectoryParams]:
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
    
    params_list = []
    
    # Generate static obstacles first
    for i in range(num_static):
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



#### GENERATE RANDOM STATIC TRAJECTORY


given_obstacle_pos = {
    'l1_obs': np.array([0.5, 0.15,  0. ]),
    'l2_top_obs': np.array([0.4, -0.05 , 0.   ]),
    'l2_bottom_obs': np.array([0.5, -0.05 , 0. ]),
    'l3_top_obs': np.array([0.35, 0.08, 0.  ]),
    'l3_mid_obs': np.array([0.5 , 0.08, 0. ]),
    'l3_bottom_obs': np.array([0.35, 0.15, 0. ])
}


trajectory_params = generate_static_trajectory_params(
    num_obstacles=6, 
    seed=42,  # Main seed for trajectory parameters
    x_bounds=(0.2, 0.8), 
    y_bounds=(-0.4, 0.4),
    noise_range=(0.0, 0.02)    # Low noise range for static obstacles
)
# Generate random initial positions
initial_positions = []
for params in trajectory_params:
    initial_positions.append(np.array([params.x_c, params.y_c]))



#### GENERATE RANDOM DYNAMIC TRAJECTORY
# trajectory_params = generate_random_trajectory_params(
#     num_obstacles=6, 
#     seed=42,  # Main seed for trajectory parameters
#     x_bounds=(0.2, 0.8), 
#     y_bounds=(-0.4, 0.4),
#     r_range=(0.05, 0.15),      # Reasonable radius range
#     omega_range=(1.0, 3.0),    # Angular velocity range
#     noise_range=(0.0, 0.05)    # Low noise range
# )

# # Generate random initial positions
# initial_positions = generate_random_initial_positions(
#     num_obstacles=6,
#     seed=43,  # Different seed for initial positions
#     x_bounds=(0.2, 0.8),
#     y_bounds=(-0.4, 0.4)
# )


print(f"Set initial positions to match static reference positions")


def create_new_random_trajectories(seed: int = None, 
                                 num_obstacles: int = 6,
                                 x_bounds: Tuple[float, float] = (0.2, 0.8),
                                 y_bounds: Tuple[float, float] = (-0.4, 0.4)) -> Tuple[List[ObstacleTrajectoryParams], List[np.ndarray]]:
    """
    Convenience function to create new random trajectories with a different seed
    
    Args:
        seed: Random seed (if None, a random seed will be used)
        num_obstacles: Number of obstacles
        x_bounds: X-axis bounds (min_x, max_x)
        y_bounds: Y-axis bounds (min_y, max_y)
    
    Returns:
        Tuple of (trajectory_params, initial_positions)
    """
    
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    print(f"\n🎲 Creating new random trajectories with seed: {seed}")
    print("=" * 50)
    
    # Generate new random parameters
    new_trajectory_params = generate_random_trajectory_params(
        num_obstacles=num_obstacles,
        seed=seed,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        r_range=(0.05, 0.15),
        omega_range=(1.0, 3.0),
        noise_range=(0.0, 0.05)
    )
    
    # Generate new initial positions with different seed
    new_initial_positions = generate_random_initial_positions(
        num_obstacles=num_obstacles,
        seed=seed + 1,  # Different seed for initial positions
        x_bounds=x_bounds,
        y_bounds=y_bounds
    )
    
    return new_trajectory_params, new_initial_positions


def create_static_trajectories(seed: int = None, 
                             num_obstacles: int = 6,
                             x_bounds: Tuple[float, float] = (0.2, 0.8),
                             y_bounds: Tuple[float, float] = (-0.4, 0.4)) -> Tuple[List[ObstacleTrajectoryParams], List[np.ndarray]]:
    """
    Convenience function to create static trajectories
    
    Args:
        seed: Random seed (if None, a random seed will be used)
        num_obstacles: Number of obstacles
        x_bounds: X-axis bounds (min_x, max_x)
        y_bounds: Y-axis bounds (min_y, max_y)
    
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
        noise_range=(0.0, 0.0)  # Low noise for static obstacles
    )
    
    # For static obstacles, initial positions are the same as the static positions
    static_initial_positions = []
    for params in static_trajectory_params:
        static_initial_positions.append(np.array([params.x_c, params.y_c]))
    
    return static_trajectory_params, static_initial_positions


def create_mixed_trajectories(seed: int = None,
                            num_obstacles: int = 6,
                            num_static: int = 3,
                            x_bounds: Tuple[float, float] = (0.2, 0.8),
                            y_bounds: Tuple[float, float] = (-0.4, 0.4)) -> Tuple[List[ObstacleTrajectoryParams], List[np.ndarray]]:
    """
    Convenience function to create mixed static and moving trajectories
    
    Args:
        seed: Random seed (if None, a random seed will be used)
        num_obstacles: Total number of obstacles
        num_static: Number of static obstacles
        x_bounds: X-axis bounds (min_x, max_x)
        y_bounds: Y-axis bounds (min_y, max_y)
    
    Returns:
        Tuple of (trajectory_params, initial_positions)
    """
    
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    print(f"\n🔄 Creating MIXED trajectories with seed: {seed}")
    print("=" * 50)
    
    # Generate mixed parameters
    mixed_trajectory_params = generate_mixed_trajectory_params(
        num_obstacles=num_obstacles,
        num_static=num_static,
        seed=seed,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        r_range=(0.05, 0.15),
        omega_range=(1.0, 3.0),
        noise_range=(0.0, 0.05)
    )
    
    # Generate initial positions (different seed for extra randomness)
    mixed_initial_positions = generate_random_initial_positions(
        num_obstacles=num_obstacles,
        seed=seed + 1,
        x_bounds=x_bounds,
        y_bounds=y_bounds
    )
    
    return mixed_trajectory_params, mixed_initial_positions


class PreGeneratedTrajectoryController:
    """Controller that pre-generates entire trajectories for maximum efficiency"""
    
    def __init__(self, dt: float = 0.01, k_p: float = 5.0, k_d: float = 0.5):
        self.dt = dt
        self.k_p = k_p
        self.k_d = k_d
        
    def generate_reference_trajectory(self, 
                                    params: ObstacleTrajectoryParams, 
                                    time_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-generate the entire reference trajectory (feedforward part)
        Returns: (positions, velocities) as arrays of shape (N, 2)
        """
        # Vectorized computation for all time steps at once
        angles = params.omega * time_array + params.phi
        
        # Position trajectory
        x_ref = params.x_c + params.r * np.cos(angles)
        y_ref = params.y_c + params.r * np.sin(angles)
        positions = np.column_stack([x_ref, y_ref])
        
        # Velocity trajectory (derivative of position)
        dx_ref = -params.r * params.omega * np.sin(angles)
        dy_ref = params.r * params.omega * np.cos(angles)
        velocities = np.column_stack([dx_ref, dy_ref])
        
        return positions, velocities
    
    def generate_random_noise_trajectory(self, 
                                       noise_std: float, 
                                       num_steps: int, 
                                       seed: Optional[int] = None) -> np.ndarray:
        """
        Pre-generate entire random noise trajectory
        Returns: noise array of shape (N, 2)
        """
        if seed is not None:
            np.random.seed(seed)
        
        return np.random.normal(0, noise_std, size=(num_steps, 2))
    
    def simulate_trajectory(self, 
                          params: ObstacleTrajectoryParams,
                          time_array: np.ndarray,
                          initial_position: np.ndarray,
                          noise_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate entire trajectory with pre-generated references and feedback control
        Returns: (actual_positions, control_inputs)
        """
        num_steps = len(time_array)
        
        # Pre-generate reference trajectory (feedforward)
        ref_positions, ref_velocities = self.generate_reference_trajectory(params, time_array)
        
        # Pre-generate random noise
        noise_trajectory = self.generate_random_noise_trajectory(
            params.noise_std, num_steps, noise_seed
        )
        
        # Initialize arrays for simulation
        actual_positions = np.zeros((num_steps, 2))
        actual_velocities = np.zeros((num_steps, 2))
        control_inputs = np.zeros((num_steps, 2))
        
        # Set initial conditions
        actual_positions[0] = initial_position.copy()
        actual_velocities[0] = np.zeros(2)
        
        # Simulate step by step (only feedback part needs to be computed iteratively)
        for i in range(num_steps - 1):
            # Current state
            current_pos = actual_positions[i]
            current_vel = actual_velocities[i]
            
            # Reference at current time
            ref_pos = ref_positions[i]
            ref_vel = ref_velocities[i]
            
            # Feedback control (this part still needs current state)
            pos_error = ref_pos - current_pos
            vel_error = ref_vel - current_vel
            
            # Total control: feedforward + feedback + noise
            control = ref_vel + self.k_p * pos_error + self.k_d * vel_error + noise_trajectory[i]
            control_inputs[i] = control
            
            # Update state using Euler integration
            actual_velocities[i + 1] = control
            actual_positions[i + 1] = current_pos + control * self.dt
        
        # Handle last time step
        control_inputs[-1] = control_inputs[-2]  # Copy last control
        
        return actual_positions, control_inputs


def create_optimized_simulation():
    """Create and run the optimized simulation with pre-generated trajectories"""
    
    # Simulation parameters
    dt = 0.01
    T = 2.0
    time_array = np.arange(0, T, dt)
    num_steps = len(time_array)
    
    print(f"Pre-generating trajectories for {num_steps} time steps...")
    
    # Create controller
    controller = PreGeneratedTrajectoryController(dt=dt)
    
   
    # Pre-generate ALL trajectories at once
    all_positions = {}
    all_controls = {}
    all_references = {}
    
    for i, (params, init_pos) in enumerate(zip(trajectory_params, initial_positions)):
        obstacle_name = f"obstacle_{i+1}"
        print(f"Generating trajectory for {obstacle_name}...")
        
        # Generate reference trajectory (pure feedforward)
        ref_pos, ref_vel = controller.generate_reference_trajectory(params, time_array)
        all_references[obstacle_name] = {'positions': ref_pos, 'velocities': ref_vel}
        
        # Simulate actual trajectory with feedback and noise
        actual_pos, controls = controller.simulate_trajectory(
            params, time_array, init_pos, noise_seed=i  # Different seed for each obstacle
        )
        
        all_positions[obstacle_name] = actual_pos
        all_controls[obstacle_name] = controls
    
    print("✅ All trajectories pre-generated successfully!")
    
    return time_array, all_positions, all_controls, all_references


def create_fully_precomputed_simulation():
    """Alternative: Fully pre-compute everything (even more optimized)"""
    
    # Simulation parameters
    dt = 0.01
    T = 2.0
    time_array = np.arange(0, T, dt)
    num_steps = len(time_array)
    k_p = 5.0
    
    print(f"Fully pre-computing {num_steps} steps for 5 obstacles...")
    
    # Define all parameters
    
    results = {}
    
    for i, (params, init_pos) in enumerate(zip(trajectory_params, initial_positions)):
        obstacle_name = f"obstacle_{i+1}"
        
        # Pre-compute reference trajectory using vectorization
        angles = params.omega * time_array + params.phi
        x_ref = params.x_c + params.r * np.cos(angles)
        y_ref = params.y_c + params.r * np.sin(angles)
        dx_ref = -params.r * params.omega * np.sin(angles)
        dy_ref = params.r * params.omega * np.cos(angles)
        
        # Pre-generate random noise
        np.random.seed(i + 42)  # Reproducible noise
        noise = np.random.normal(0, params.noise_std, size=(num_steps, 2))
        
        # Initialize simulation arrays
        positions = np.zeros((num_steps, 2))
        velocities = np.zeros((num_steps, 2))
        controls = np.zeros((num_steps, 2))
        
        # Set initial state
        positions[0] = init_pos
        
        # Vectorized simulation (where possible)
        for step in range(num_steps - 1):
            # Current state
            current_pos = positions[step]
            current_vel = velocities[step]
            
            # Reference at this time step
            ref_pos = np.array([x_ref[step], y_ref[step]])
            ref_vel = np.array([dx_ref[step], dy_ref[step]])
            
            # Control computation
            pos_error = ref_pos - current_pos
            control = ref_vel + k_p * pos_error + noise[step]
            controls[step] = control
            
            # State update
            velocities[step + 1] = control
            positions[step + 1] = current_pos + control * dt
        
        results[obstacle_name] = {
            'positions': positions,
            'controls': controls,
            'reference_positions': np.column_stack([x_ref, y_ref]),
            'reference_velocities': np.column_stack([dx_ref, dy_ref])
        }
    
    print("✅ Fully pre-computed simulation complete!")
    return time_array, results


def plot_optimized_results(time_array, all_positions, all_controls, method_name="Optimized"):
    """Plot results from optimized simulation"""
    
    plt.figure(figsize=(15, 10))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    
    # Plot trajectories
    plt.subplot(2, 3, 1)
    for i, (name, positions) in enumerate(all_positions.items()):
        plt.plot(positions[:, 0], positions[:, 1], 
                color=colors[i], label=name, linewidth=2)
        plt.plot(positions[0, 0], positions[0, 1], 
                'o', color=colors[i], markersize=8, markeredgecolor='black')
    

    data_directory = '/home/erdi/Storage/publications/dpcc/d3il/environments/dataset/data/avoiding/data/'

    state_files = os.listdir(data_directory)
    files_to_plot = state_files

    for file in files_to_plot:
        with open(os.path.join(data_directory, file), 'rb') as f:
            env_state = pickle.load(f)
            
            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            robot_c_pos = env_state['robot']['c_pos'][:, :2]
            
            # Create masks for points outside the 0.2-0.3 y-range
            # valid_y_mask = ~((0.2 <= robot_des_pos[:, 1]) & (robot_des_pos[:, 1] <= 0.5))
            
            # # Filter the trajectories
            # robot_des_pos = robot_des_pos[valid_y_mask]
            # robot_c_pos = robot_c_pos[valid_y_mask]
            
            # Plot desired trajectory
            plt.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], 'b-', alpha=0.1, label='Desired trajectory' if file == files_to_plot[0] else "")
            
            # Plot actual trajectory
            plt.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], 'r-', alpha=0.1, label='Actual trajectory' if file == files_to_plot[0] else "")
            
            # Plot start points
            plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^', label='Start points' if file == files_to_plot[0] else "")
            
            # Plot end points
            plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x', label='End points' if file == files_to_plot[0] else "")
    


    plt.xlim(0.3, 0.8)
    plt.ylim(-0.3, 0.4)
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
    
    # Performance comparison
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f"Method: {method_name}", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f"Time steps: {len(time_array)}", fontsize=10)
    plt.text(0.1, 0.6, f"Obstacles: {len(all_positions)}", fontsize=10)
    plt.text(0.1, 0.5, "✅ Pre-generated trajectories", fontsize=10, color='green')
    plt.text(0.1, 0.4, "✅ Vectorized computations", fontsize=10, color='green')
    plt.text(0.1, 0.3, "✅ Efficient memory usage", fontsize=10, color='green')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Optimization Benefits')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png')


def save_trajectories_as_array(pos_hist, ctrl_hist, save_path=None):
    """
    Convert position and control histories to structured numpy array
    
    Args:
        pos_hist: Dictionary with obstacle names as keys and position arrays as values
        ctrl_hist: Dictionary with obstacle names as keys and control arrays as values
        save_path: Optional path to save the array
    
    Returns:
        numpy array with shape [batch_size, obstacles, position + controller input, timestep]
        where:
        - batch_size = 1 (single simulation run)
        - obstacles = number of obstacles
        - position + controller input = 4 (2 for position, 2 for control)
        - timestep = number of time steps
    """
    
    # Get number of obstacles and time steps
    obstacle_names = sorted(pos_hist.keys())  # Sort to ensure consistent ordering
    num_obstacles = len(obstacle_names)
    num_timesteps = pos_hist[obstacle_names[0]].shape[0]
    
    print(f"Converting trajectories to structured array:")
    print(f"  - Number of obstacles: {num_obstacles}")
    print(f"  - Number of timesteps: {num_timesteps}")
    print(f"  - Data dimensions: 4 (x_pos, y_pos, u_x, u_y)")
    
    # Initialize the structured array [batch_size, obstacles, features, timesteps]
    batch_size = 1
    num_features = 4  # x_pos, y_pos, u_x, u_y
    
    trajectory_array = np.zeros((batch_size, num_obstacles, num_features, num_timesteps))
    
    # Fill the array
    for i, obstacle_name in enumerate(obstacle_names):
        # Get position and control data for this obstacle
        positions = pos_hist[obstacle_name]  # shape: (timesteps, 2)
        controls = ctrl_hist[obstacle_name]   # shape: (timesteps, 2)
        
        # Fill the array: [batch=0, obstacle=i, :, :]
        trajectory_array[0, i, 0, :] = positions[:, 0]  # x_position
        trajectory_array[0, i, 1, :] = positions[:, 1]  # y_position
        trajectory_array[0, i, 2, :] = controls[:, 0]   # u_x (control x)
        trajectory_array[0, i, 3, :] = controls[:, 1]   # u_y (control y)
        
        print(f"  - {obstacle_name}: ✓")
    
    print(f"Final array shape: {trajectory_array.shape}")
    print(f"Array structure: [batch_size={batch_size}, obstacles={num_obstacles}, features={num_features}, timesteps={num_timesteps}]")
    
    # Save if path provided
    if save_path:
        np.save(save_path, trajectory_array)
        print(f"Array saved to: {save_path}")
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"  - Position range X: [{trajectory_array[0, :, 0, :].min():.3f}, {trajectory_array[0, :, 0, :].max():.3f}]")
    print(f"  - Position range Y: [{trajectory_array[0, :, 1, :].min():.3f}, {trajectory_array[0, :, 1, :].max():.3f}]")
    print(f"  - Control range X: [{trajectory_array[0, :, 2, :].min():.3f}, {trajectory_array[0, :, 2, :].max():.3f}]")
    print(f"  - Control range Y: [{trajectory_array[0, :, 3, :].min():.3f}, {trajectory_array[0, :, 3, :].max():.3f}]")
    
    return trajectory_array


def load_and_extract_trajectories(array_path):
    """
    Load a saved trajectory array and extract position and control histories
    
    Args:
        array_path: Path to the saved numpy array
        
    Returns:
        tuple: (pos_hist, ctrl_hist) in original dictionary format
    """
    
    # Load the array
    trajectory_array = np.load(array_path)
    batch_size, num_obstacles, num_features, num_timesteps = trajectory_array.shape
    
    print(f"Loaded array shape: {trajectory_array.shape}")
    
    # Convert back to dictionary format
    pos_hist = {}
    ctrl_hist = {}
    
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i+1}"
        
        # Extract positions and controls
        positions = trajectory_array[0, i, 0:2, :].T  # shape: (timesteps, 2)
        controls = trajectory_array[0, i, 2:4, :].T   # shape: (timesteps, 2)
        
        pos_hist[obstacle_name] = positions
        ctrl_hist[obstacle_name] = controls
    
    return pos_hist, ctrl_hist


if __name__ == "__main__":
    print("🚀 Running optimized simulations with RANDOM trajectory parameters...\n")
    
    print("Default random trajectories (seed=42) are already loaded!")
    print("Current trajectory parameters and initial positions:")
    print("=" * 60)
    for i, (param, init_pos) in enumerate(zip(trajectory_params, initial_positions)):
        print(f"Obstacle {i+1}: r={param.r:.3f}, ω={param.omega:.2f}, φ={param.phi:.2f}, "
              f"center=({param.x_c:.3f}, {param.y_c:.3f}), init=({init_pos[0]:.3f}, {init_pos[1]:.3f})")
    
    # Example: Create new random trajectories with different seed
    print("\n🎲 EXAMPLE: Creating new random trajectories with seed=123")
    print("=" * 60)
    new_params, new_positions = create_new_random_trajectories(seed=123)
    
    # Use either the default random trajectories or the new ones
    print(f"\n📋 Choose which trajectories to simulate:")
    print("1. Default random trajectories (seed=42)")
    print("2. New random trajectories (seed=123)")
    print("Using default trajectories for simulation...\n")
    
    # Method 1: Object-oriented approach with pre-generation
    print("=" * 60)
    print("METHOD 1: Object-oriented with random pre-generated references")
    print("=" * 60)
    time_hist1, pos_hist1, ctrl_hist1, ref_hist1 = create_optimized_simulation()
    plot_optimized_results(time_hist1, pos_hist1, ctrl_hist1, "Method 1: Random OOP Pre-generated")
    
    print("\n" + "=" * 60)
    print("METHOD 2: Fully vectorized pre-computation with random parameters")
    print("=" * 60)
    time_hist2, results2 = create_fully_precomputed_simulation()
    
    # Extract positions and controls for plotting
    pos_hist2 = {name: data['positions'] for name, data in results2.items()}
    ctrl_hist2 = {name: data['controls'] for name, data in results2.items()}
    
    plot_optimized_results(time_hist2, pos_hist2, ctrl_hist2, "Method 2: Random Fully Vectorized")
    
    # Save trajectories as structured numpy arrays
    print("\n" + "=" * 60)
    print("SAVING RANDOM TRAJECTORIES AS NUMPY ARRAYS")
    print("=" * 60)
    
    # Save Method 1 results
    print("\nSaving Method 1 random trajectories...")
    trajectory_array_1 = save_trajectories_as_array(pos_hist1, ctrl_hist1, "trajectory_method1.npy")
    
    # Save Method 2 results
    print("\nSaving Method 2 random trajectories...")
    trajectory_array_2 = save_trajectories_as_array(pos_hist2, ctrl_hist2, "trajectory_method2.npy")
    
    print("\n" + "=" * 60)
    print("RANDOM TRAJECTORY GENERATION SUMMARY")
    print("=" * 60)
    print("✅ Random trajectory parameters with seed control for reproducibility!")
    print("✅ Constraint satisfaction:")
    print(f"   • X-axis bounds: [0.2, 0.8]")
    print(f"   • Y-axis bounds: [-0.4, 0.4]")
    print(f"   • Circular trajectories stay within bounds")
    print("✅ Randomized parameters:")
    print("   • Radius (r): 0.05 to 0.15")
    print("   • Angular velocity (ω): 1.0 to 3.0 rad/s")
    print("   • Phase offset (φ): 0 to 2π")
    print("   • Center coordinates: dynamically constrained")
    print("   • Noise levels: 0.0 to 0.05")
    print("✅ Easy regeneration with different seeds")
    print("✅ Both methods use random parameters efficiently!")
    print(f"✅ Trajectory arrays saved with shape: {trajectory_array_2.shape}")
    print("• Array format: [batch_size, obstacles, features, timesteps]")
    print("• Features: [x_pos, y_pos, u_x, u_y]")
    
    # Example: Load and verify saved arrays
    print("\n" + "=" * 60)
    print("EXAMPLE: LOADING SAVED ARRAYS")
    print("=" * 60)
    
    # Load the saved array
    print("Loading trajectory_method2.npy...")
    loaded_pos_hist, loaded_ctrl_hist = load_and_extract_trajectories("trajectory_method2.npy")
    
    # Verify the data matches
    print("\nVerifying loaded data matches original...")
    verification_passed = True
    for obstacle_name in pos_hist2.keys():
        pos_match = np.allclose(pos_hist2[obstacle_name], loaded_pos_hist[obstacle_name])
        ctrl_match = np.allclose(ctrl_hist2[obstacle_name], loaded_ctrl_hist[obstacle_name])
        if not (pos_match and ctrl_match):
            verification_passed = False
            print(f"  ❌ {obstacle_name}: Data mismatch")
        else:
            print(f"  ✅ {obstacle_name}: Data verified")
    
    if verification_passed:
        print("\n🎉 All trajectory data successfully saved and verified!")
    else:
        print("\n❌ Some data verification failed!")
    
    print("\n" + "=" * 60)
    print("HOW TO GENERATE NEW RANDOM TRAJECTORIES")
    print("=" * 60)
    print("To generate new random trajectories, use one of these methods:")
    print()
    print("1. Quick generation with new seed:")
    print("   new_params, new_positions = create_new_random_trajectories(seed=999)")
    print()
    print("2. Custom parameters:")
    print("   new_params = generate_random_trajectory_params(")
    print("       num_obstacles=8,")
    print("       seed=456,")
    print("       x_bounds=(0.1, 0.9),")
    print("       y_bounds=(-0.5, 0.5),")
    print("       r_range=(0.03, 0.20)")
    print("   )")
    print()
    print("3. Then use in simulation:")
    print("   # Replace global variables")
    print("   trajectory_params = new_params")
    print("   initial_positions = new_positions")
    print("   # Run simulation...")
    print()
    print("Access pattern examples for saved arrays:")
    print("  - All obstacles at timestep 0: array[0, :, :, 0]")
    print("  - Obstacle 1 positions: array[0, 0, 0:2, :]") 
    print("  - Obstacle 1 controls: array[0, 0, 2:4, :]")
    print("  - All X positions: array[0, :, 0, :]")
    print("  - All Y positions: array[0, :, 1, :]")