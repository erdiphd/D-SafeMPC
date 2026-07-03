import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

data_directory = '/home/erdi/Storage/publications/erdi_dpcc_test/dpcc_original/d3il/environments/dataset/data/avoiding/data/'

state_files = os.listdir(data_directory)

# First pass: determine environment boundaries by checking all trajectories
min_x, max_x = float('inf'), float('-inf')
for file in state_files:
    with open(os.path.join(data_directory, file), 'rb') as f:
        try:
            env_state = pickle.load(f)
            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            
            # Update min and max x values
            min_x = min(min_x, np.min(robot_des_pos[:, 0]))
            max_x = max(max_x, np.max(robot_des_pos[:, 0]))
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Calculate midpoint of x-axis
midpoint_x = (min_x + max_x) / 2
print(f"Environment x-range: {min_x:.3f} to {max_x:.3f}, midpoint: {midpoint_x:.3f}")

# Second pass: identify left-side trajectories (with a stricter criteria)
left_side_files = []
for file in state_files:
    with open(os.path.join(data_directory, file), 'rb') as f:
        try:
            env_state = pickle.load(f)
            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            
            # Check if at least 75% of points are on the left side for a stricter filter
            points_on_left = np.sum(robot_des_pos[:, 0] < midpoint_x)
            if points_on_left / len(robot_des_pos) >= 0.75:
                left_side_files.append(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")

print(f"Found {len(left_side_files)} trajectories with at least 75% of points on the left side.")
print("\nLeft side files:")
for file in left_side_files:
    print(f"- {file}")

# Plot only the left portions of trajectories
plt.figure(figsize=(10, 8))

for file in left_side_files:
    with open(os.path.join(data_directory, file), 'rb') as f:
        env_state = pickle.load(f)
        
        robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
        robot_c_pos = env_state['robot']['c_pos'][:, :2]
        
        # Create masks for points on the left side only
        left_mask_des = robot_des_pos[:, 0] < midpoint_x
        left_mask_actual = robot_c_pos[:, 0] < midpoint_x
        
        # Plot only the points on the left side
        if np.any(left_mask_des):
            # For continuous lines, we need to break the trajectory into segments
            # whenever it crosses the midpoint
            segments_des = []
            current_segment = []
            for i, (is_left, pos) in enumerate(zip(left_mask_des, robot_des_pos)):
                if is_left:
                    current_segment.append(pos)
                elif len(current_segment) > 0:
                    segments_des.append(np.array(current_segment))
                    current_segment = []
            if len(current_segment) > 0:
                segments_des.append(np.array(current_segment))
            
            # Plot each segment
            for segment in segments_des:
                if len(segment) > 1:  # Need at least 2 points to make a line
                    plt.plot(segment[:, 0], segment[:, 1], 'b-', alpha=0.3)
        
        if np.any(left_mask_actual):
            # Same segmentation for actual trajectory
            segments_actual = []
            current_segment = []
            for i, (is_left, pos) in enumerate(zip(left_mask_actual, robot_c_pos)):
                if is_left:
                    current_segment.append(pos)
                elif len(current_segment) > 0:
                    segments_actual.append(np.array(current_segment))
                    current_segment = []
            if len(current_segment) > 0:
                segments_actual.append(np.array(current_segment))
            
            # Plot each segment
            for segment in segments_actual:
                if len(segment) > 1:
                    plt.plot(segment[:, 0], segment[:, 1], 'r-', alpha=0.3)
        
        # Plot start points if on left side
        if robot_des_pos[0, 0] < midpoint_x:
            plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^')
        
        # Plot end points if on left side
        if robot_des_pos[-1, 0] < midpoint_x:
            plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x')
        
        # If obstacles exist in the data, plot them if they're on the left side
        if 'obstacles' in env_state:
            for obs in env_state['obstacles']:
                if 'position' in obs and 'size' in obs:
                    pos = obs['position'][:2]  # Get X,Y coordinates
                    if pos[0] < midpoint_x:  # Only plot if on left side
                        plt.scatter(pos[0], pos[1], c='black', s=100, marker='o')

# Add legend items manually since we're not using labels in the loop anymore
plt.plot([], [], 'b-', label='Desired trajectory')
plt.plot([], [], 'r-', label='Actual trajectory')
plt.scatter([], [], c='green', s=50, marker='^', label='Start points')
plt.scatter([], [], c='black', s=50, marker='x', label='End points')
plt.scatter([], [], c='black', s=100, marker='o', label='Obstacle')

plt.title('Strictly Left Side Expert Dataset Trajectories')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axvline(x=midpoint_x, color='k', linestyle='--', label='Midpoint')
plt.xlim(min_x, midpoint_x + 0.05)  # Set x-axis limit to focus on left side
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio
plt.savefig('strictly_left_side_trajectories.png', dpi=300)
plt.show()

# Save the list of left side files
with open('strictly_left_side_files.txt', 'w') as f:
    for file in left_side_files:
        f.write(f"{file}\n")