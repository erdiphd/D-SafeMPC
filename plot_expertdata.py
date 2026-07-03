import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

data_directory = '/home/erdi/Storage/publications/erdi_dpcc_test/dpcc_original/d3il/environments/dataset/data/avoiding/data/'

state_files = os.listdir(data_directory)
plt.figure(figsize=(10, 8))

# Plot for a subset of files if there are too many

files_to_plot = state_files

for file in files_to_plot:
    with open(os.path.join(data_directory, file), 'rb') as f:
        env_state = pickle.load(f)
        
        robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
        robot_c_pos = env_state['robot']['c_pos'][:, :2]
        
        # # Create masks for points outside the 0.2-0.3 y-range
        # valid_y_mask = ~((0.2 <= robot_des_pos[:, 1]) & (robot_des_pos[:, 1] <= 0.5))
        
        # # Filter the trajectories
        # robot_des_pos = robot_des_pos[valid_y_mask]
        # robot_c_pos = robot_c_pos[valid_y_mask]
        
        # Plot desired trajectory
        plt.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], 'b-', alpha=0.3, label='Desired trajectory' if file == files_to_plot[0] else "")
        
        # Plot actual trajectory
        plt.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], 'r-', alpha=0.3, label='Actual trajectory' if file == files_to_plot[0] else "")
        
        # Plot start points
        plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^', label='Start points' if file == files_to_plot[0] else "")
        
        # Plot end points
        plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x', label='End points' if file == files_to_plot[0] else "")
        
# Define obstacle positions
obstacle_positions = {
    'l1_obs': np.array([0.6, -0.05,  0. ]),
    'l2_top_obs': np.array([0.4, -0.05 , 0.   ]),
    'l2_bottom_obs': np.array([0.5, 0.08 , 0. ]),
    'l3_top_obs': np.array([0.35, 0.08, 0.  ]),
    'l3_mid_obs': np.array([0.65 , 0.08, 0. ]),
    'l3_bottom_obs': np.array([0.5, 0.3, 0. ])
}
# Plot obstacles as circles
for pos in obstacle_positions.values():
    plt.gca().add_patch(plt.Circle((pos[0], pos[1]), 0.025, color='r'))


plt.title('Expert Dataset Trajectories')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio
plt.savefig('expert_trajectories.png', dpi=300)
plt.xlim([0.2, 0.8])  # Fixed x-axis limits matching the environment bounds
plt.ylim([-0.3, 0.6]) # Fixed y-axis limits matching the environment bounds
plt.show()