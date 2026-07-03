import numpy as np
import torch
from scipy.optimize import minimize, Bounds
import copy
import time
import os
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
# Set font sizes and style for scientific publication
plt.rcParams.update({
'font.size': 12,
'font.family': 'serif',
'text.usetex': False,
'axes.labelsize': 14,
'axes.titlesize': 14,
'xtick.labelsize': 12,
'ytick.labelsize': 12,
'legend.fontsize': 12
})

from diffuser.models.reward import single_cbf_reward_fn_pairwise, single_clf_reward_fn
import yaml


# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)


class Projector:

    def __init__(self, horizon, transition_dim, action_dim=0, goal_dim=0, constraint_list=[], normalizer=None, variant='states', 
                 dt=0.1, cost_dims=None, skip_initial_state=True, diffusion_timestep_threshold=0.5, gradient=False, gradient_weights=None,
                 device='cuda', solver='proxsuite', parallelize=False ,env=None, method=None ,obstacle_constraints=None):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dt = torch.tensor(dt, device=device)
        self.skip_initial_state = skip_initial_state
        # self.only_last = only_last
        self.diffusion_timestep_threshold = diffusion_timestep_threshold
        self.gradient = gradient
        self.gradient_weights = gradient_weights
        self.device = device
        self.solver = solver
        self.parallelize = parallelize
        self.method = method
        self.env = env
        self.config_obstacle_constraints = obstacle_constraints
        self.plot_data = {}
        self.plot_debug = True
        
        # Determine whether to include actions in the projection
        if normalizer is None:
            self.normalizer = None
        elif variant == 'states':
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'], goal_dim=goal_dim)
        elif variant == 'states_actions':
        # elif transition_dim != normalizer.normalizers['observations'].maxs.size:
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'], 
                                                   action_normalizer=normalizer.normalizers['actions'], goal_dim=goal_dim)
        else:
            KeyError('Invalid variant. Choose either "states" or "states_actions".')            

        # Quadratic cost
        if cost_dims is not None:
            costs = torch.ones(transition_dim, device=self.device)
            for idx in cost_dims:
                costs[idx] = 1
            self.Q = torch.diag(torch.tile(costs, (self.horizon, )))
        else:
            self.Q = torch.eye(transition_dim * horizon, device=self.device)

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Equality constraints
        self.b = torch.empty(0, device=self.device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Inequality constraints
        self.d = torch.empty(0, device=self.device)

        self.safety_constraints = SafetyConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer, 
                                                 skip_initial_state=self.skip_initial_state, action_dim=action_dim, device=self.device)
        self.dynamic_constraints = DynamicConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                      skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)
        self.obstacle_constraints = ObstacleConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                        skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)
        self.cbf_constraints = CBF(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                        skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)
        for constraint_spec in constraint_list:
            if constraint_spec[0] == 'deriv':
                self.dynamic_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'lb' or constraint_spec[0] == 'ub' or constraint_spec[0] == 'eq' or constraint_spec[0] == 'ineq':
                self.safety_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'sphere_inside' or constraint_spec[0] == 'sphere_outside':
                self.obstacle_constraints.constraint_list.append(constraint_spec)
                self.cbf_constraints.constraint_list.append(constraint_spec)

        
        safety_constraints = copy.deepcopy(self.obstacle_constraints.constraint_list[0])
        safety_constraints_cbf = copy.deepcopy(self.cbf_constraints.constraint_list[0])
        config_constraints = copy.deepcopy(safety_constraints)
        all_obstacles_in_env = env.get_obstacle_position()
        for obstacle_name in all_obstacles_in_env:
            safety_constraints[2] = all_obstacles_in_env[obstacle_name][:2].tolist()
            safety_constraints[3] = 0.03
            self.obstacle_constraints.constraint_list.append(copy.deepcopy(safety_constraints))
            safety_constraints_cbf[2] = all_obstacles_in_env[obstacle_name][:2].tolist()
            safety_constraints_cbf[3] = 0.03
            self.cbf_constraints.constraint_list.append(copy.deepcopy(safety_constraints_cbf))

        self.safety_constraints.build_matrices()
        self.dynamic_constraints.build_matrices()
        self.obstacle_constraints.build_matrices()
        self.cbf_constraints.build_matrices()
        self.append_linear_constraint(self.safety_constraints)
        self.append_linear_constraint(self.dynamic_constraints)
        self.add_numpy_constraints()
        
        trajectory_file = config.get('trajectory_file', 'generated_trajectories.npy')
        self.external_obstacle_pos =  np.load(trajectory_file)
        self.external_obstacle_pos_tensor = torch.tensor(self.external_obstacle_pos, device=self.device, dtype=torch.float32)
        self.single_cbf_grad_fn = torch.func.grad(single_cbf_reward_fn_pairwise)
        self.batched_cbf_grad_fn = torch.vmap(self.single_cbf_grad_fn, in_dims=(None, 0, None, None, None))


    def compute_projection_costs(self, sol_np, trajectory_np, Q, r_np, i):
            # Original cost terms
            # quadratic_cost = 0.5 * sol_np @ Q @ sol_np
            # linear_cost = r_np @ sol_np
            # reference_cost = 0.5 * trajectory_np @ Q @ trajectory_np
            
            sol_np = sol_np.reshape(self.horizon, self.transition_dim)
            sol_np = self.normalizer.unnormalize(sol_np).flatten()
            
            # Smoothness cost
            # smoothness_cost = 0.0
            # for t in range(1, self.horizon-1):
            #     curr_state = sol_np[t*self.transition_dim:(t+1)*self.transition_dim]
            #     next_state = sol_np[(t+1)*self.transition_dim:(t+2)*self.transition_dim]
            #     smoothness_cost += np.linalg.norm(next_state - curr_state)**2
            
            # Safety margin cost
            safety_cost = 0.0
            min_safe_dist = 0.1  # Minimum safe distance to obstacles
            # obstacle_env_data = [self.env.get_obstacle_position()]
            for t in range(self.horizon):
                state = sol_np[t*self.transition_dim:(t+1)*self.transition_dim]
                pos = state[4:6]  # Assuming positions are at indices 4 and 5
                # fig_debug, ax_debug = plt.subplots(figsize=(10, 10))
                # ax_debug.scatter(pos[0], pos[1], c='blue', alpha=0.5, s=20)
                # centers = []
                # for center_pos_key in list(obstacle_env_data[-1].keys())[:6]:  # First 6 are obstacles
                #     centers.append(obstacle_env_data[-1][center_pos_key][:2])
                # for center in centers:
                #     ax_debug.add_patch(matplotlib.patches.Circle(center, 0.025, color='r'))
                
                # for k in range(len(self.config_obstacle_constraints)):
                #     pos_i = self.config_obstacle_constraints[k]['center']
                #     r_i = self.config_obstacle_constraints[k]['radius']
                #     ax_debug.add_patch(matplotlib.patches.Circle(pos_i, r_i, color='b', alpha=0.2))
        
                # # Plot finish line
                # ax_debug.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
                # plt.savefig(f'test.png', format='png', dpi=300, bbox_inches='tight')
                # plt.close(fig_debug)

                for i in range(len(self.obstacle_constraints.constraint_list)):
                    center = self.obstacle_constraints.constraint_list[i][2]
                    radius = self.obstacle_constraints.constraint_list[i][3]
                    dist = np.linalg.norm(pos - center) - radius
                    if dist < 0.01:
                        safety_cost += np.exp(10 * (min_safe_dist - dist))
            
            # Velocity cost (penalize high velocities near obstacles)
            # velocity_cost = 0.0
            # for t in range(self.horizon):
            #     state = sol_np[t*self.transition_dim:(t+1)*self.transition_dim]
            #     vel = state[:2]  # Assuming velocities are at indices 0 and 1
                
            #     # Weight velocity cost by proximity to obstacles
            #     for i in range(len(self.obstacle_constraints.constraint_list)):
            #         center = self.obstacle_constraints.constraint_list[i][2]
            #         radius = self.obstacle_constraints.constraint_list[i][3]
            #         dist = np.linalg.norm(state[4:6] - center) - radius
            #         if dist < min_safe_dist:
            #             velocity_cost += np.linalg.norm(vel)**2 * (min_safe_dist / (dist + 1e-6))
            
            # Combine all costs with weights
            # w_obstacle = 0.3
            # w_smooth = 0.0
            w_safety = 2
            # w_velocity = 0.0
            
            total_cost = w_safety * safety_cost
            
            return total_cost


    def equality_constraint_cost(self, x):
        # x is normalized trajectory
        return self.A_np @ x - self.b_np

    def inequality_constraint_cost(self, x):
        # x is normalized trajectory
        return self.C_np @ x - self.d_np   
        

    def compute_distance_to_obstacle_torch(self, x):
        num_constraints = len(self.obstacle_constraints.P_list)
        distances = torch.zeros((num_constraints, self.horizon-1), device=self.device)

        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = torch.tensor(self.obstacle_constraints.P_list[constraint_idx], device=self.device, dtype=torch.float32)
            q = torch.tensor(self.obstacle_constraints.q_list[constraint_idx], device=self.device, dtype=torch.float32)
            v = torch.tensor(self.obstacle_constraints.v_list[constraint_idx], device=self.device, dtype=torch.float32)
            
            for t in range(1, self.horizon):
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                x_t = x[start_idx:end_idx]
                dist = -x_t @ P @ x_t - q @ x_t + v
                distances[constraint_idx, t-1] = dist
        
        return distances


    def obstacle_constraint_cost(self, x):
        # x is normalized trajectory

        # x_plot = x.reshape(self.horizon,self.transition_dim)
        # x_plot = self.normalizer.unnormalize(x_plot).flatten()

        num_constraints = len(self.obstacle_constraints.P_list)
        distances = np.zeros((num_constraints, self.horizon-1))
        
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = self.obstacle_constraints.P_list[constraint_idx]
            q = self.obstacle_constraints.q_list[constraint_idx]
            v = self.obstacle_constraints.v_list[constraint_idx]
            # center_tmp = self.obstacle_constraints.constraint_list[constraint_idx][2]
            # radius_tmp = self.obstacle_constraints.constraint_list[constraint_idx][3]
            for t in range(1, self.horizon):
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                x_t = x[start_idx:end_idx]
                # x_tmp = x_plot[start_idx:end_idx]
                # pos_tmp = x_tmp[4:6]         
                dist = - x_t @ P @ x_t - q @ x_t + v
                distances[constraint_idx, t-1] = dist
                # dist_tmp = np.linalg.norm(pos_tmp - center_tmp)**2 - radius_tmp**2
        
        return distances


    def diffmpc_cost(self, z, Q, p):
        if type(z) == torch.Tensor:
            # Original cost terms
            # tracking_cost = 0.5 * z @ Q @ z + torch.cos(cost_lambda) * p @ z /(torch.linalg.norm(z)* torch.linalg.norm(p) + 1e-6)
            tracking_cost = (0.5 * z @ (0.01 * Q) @ z)
            # Obstacle avoidance cost
            # obstacle_distances = self.compute_distance_to_obstacle_torch(z)
            # obstacle_cost = -torch.sum(torch.minimum(obstacle_distances, torch.zeros_like(obstacle_distances)))
            
            cost = tracking_cost
        else:   
            # Original cost terms
            # tracking_cost = 0.5 * z @ Q @ z + np.cos(cost_lambda) * p @ z /(np.linalg.norm(z)* np.linalg.norm(p) + 1e-6)
            tracking_cost = (0.5 * z @ (0.01 * Q) @ z)
            # Obstacle avoidance cost
            # obstacle_distances = self.obstacle_constraint_cost(z)
            # obstacle_cost = -np.sum(np.minimum(obstacle_distances, np.zeros_like(obstacle_distances)))
            
            cost = tracking_cost
        return cost



    def objective(self, x, Q, p, model ,cond, apply_conditioning, timesteps):
        timesteps = torch.tensor([x[-1]],dtype=torch.int32,device=self.device)
        x = x[:-1]
        x = torch.tensor(x,dtype=torch.float32,device=self.device).reshape(-1,self.horizon,self.transition_dim)
        # Get diffusion model output
        x = model(x,cond[0][0:1,:],timesteps[0:1],projector=self ,env=self.env)
        # x[:,:,2:] = cond[0][0,:]
        x = x.detach().cpu().numpy().reshape(self.horizon*self.transition_dim)
        

        # eq_violation = self.equality_constraint_cost(x)
        # ineq_violation = self.inequality_constraint_cost(x)
        # obstacle_distance = self.obstacle_constraint_cost(x)
        cost = self.diffmpc_cost(x, Q, p)
        self.plot_data['cost'].append(cost)
        self.plot_data['timesteps'].append(timesteps.clone().detach().cpu().numpy()[0])
        return cost

    def jacobian(self, x, Q, p, model,cond, apply_conditioning, timesteps):
        torch.set_grad_enabled(True)
        # Extract trajectory and timestep from x
        z_flat = x[:-1]  # Trajectory part
        t_val = x[-1]    # Timestep part
        
        # Convert to tensors with gradients enabled
        z = torch.tensor(z_flat, dtype=torch.float32, device=self.device).reshape(-1, self.horizon, self.transition_dim)
        timesteps = torch.tensor([t_val], dtype=torch.float32, device=self.device)

        z.requires_grad_(True)
        timesteps.requires_grad_(True)
        
        # Compute model output
        d = model(z, cond[0][0:1,:], timesteps, projector=self, env=self.env)
        # d[:,:,2:] = cond[0][0,:]
        d = d.reshape(self.horizon * self.transition_dim)
        # Compute the cost
        Q_torch = torch.tensor(Q, dtype=torch.float32, device=self.device)
        p_torch = torch.tensor(p, dtype=torch.float32, device=self.device)

        # obstacle_distance = self.compute_distance_to_obstacle_torch(d)
        cost = self.diffmpc_cost(d, Q_torch, p_torch)
        # Compute gradients with respect to z and timesteps
        grad_z = torch.autograd.grad(cost, z, create_graph=False, retain_graph=True)[0]
        grad_timesteps = torch.autograd.grad(cost, timesteps, create_graph=False)[0]
        
        
        # grad_timesteps is shape (1,), take the scalar value
        grad_t = grad_timesteps[0]
        
        # Flatten grad_z and concatenate with grad_t
        grad_z_flat = grad_z.reshape(-1)  # Size: horizon * transition_dim
        grad_x = torch.cat([grad_z_flat, grad_t.unsqueeze(0)], dim=0)  # Size: horizon * transition_dim + 1
        
        return grad_x.detach().cpu().numpy()

                 
    def project_mpcdiff(self, trajectory, constraints=None, model=None, apply_conditioning=None ,cond=None, timesteps=None, save_dir=None):
            """
                trajectory: np.ndarray of shape (batch_size, horizon, transition_dim)
                Solve an optimization problem of the form 
                    \hat z =   argmin_z 1/2 z^T Q z + r^T z
                            subject to  Az  = b
                                        Cz <= d
                where z = (o_0, o_1, ..., o_{H-1}) is the trajectory in vector form. The matrices A, b, C, and d are defined by the dynamic and safety constraints.
                                        
            """
            
            # projection_costs = np.ones(trajectory.shape[0], dtype=np.float32)
            # return trajectory, projection_costs
            dims = trajectory.shape

            # Reshape the trajectory to a batch of vectors (from B x H x T to B x (HT)
            batch_size = trajectory.shape[0]
            trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)

            # Cost
            r = - trajectory_reshaped @ self.Q
            r_np = r.detach().cpu().numpy()
            Q = self.Q_np.astype('double')
            trajectory_np = trajectory_reshaped.detach().cpu().numpy()

            # Constraints
            A = self.A_np.astype('double')
            A = np.concatenate([A, np.zeros((A.shape[0], 1), dtype=A.dtype)], axis=1)
            b = self.b_np.astype('double')
            C = self.C_np.astype('double')
            C = np.concatenate([C, np.zeros((C.shape[0], 1), dtype=C.dtype)], axis=1)
            d = self.d_np.astype('double')

            if self.skip_initial_state:
                s_0 = trajectory_reshaped[0, :self.transition_dim]
                if self.solver == 'proxsuite' or self.solver == 'gurobi':
                    s_0 = s_0.cpu().numpy()
                counter = 0
                for constraint in self.dynamic_constraints.constraint_list:
                    if constraint[0] == 'deriv':
                        x_idx = int(constraint[1][0])
                        b[counter * self.horizon] = s_0[x_idx]
                        counter += 1

            r_np_double = r_np.astype('double')
            trajectory_np_double = trajectory_np.astype('double')
            # Constraints
            constraints = ()

            for obs_idx, obs_key in enumerate(self.env.get_obstacle_position().keys()):
                obstacle_pos = self.env.get_obstacle_position()[obs_key]
                self.obstacle_constraints.constraint_list[obs_idx +  1][2] = obstacle_pos[:2]


            self.obstacle_constraints.build_matrices()
            for constraint_idx in range(len(self.obstacle_constraints.P_list)):
                P = self.obstacle_constraints.P_list[constraint_idx]
                q = self.obstacle_constraints.q_list[constraint_idx]
                v = self.obstacle_constraints.v_list[constraint_idx]
                for t in range(1, self.horizon):                        # Obstacle constraints
                    start_idx = t * self.transition_dim
                    end_idx = (t + 1) * self.transition_dim
                    constraints += ({'type': 'ineq', 'fun': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q, v=v: -x[start_idx: end_idx] @ P @ x[start_idx: end_idx] - q @ x[start_idx: end_idx] + v,
                                        'jac': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q: np.concatenate([np.zeros(start_idx), -2 * P @ x[start_idx: end_idx] - q, np.zeros(len(x) - end_idx)])},)

            # for constraint_idx in range(len(self.cbf_constraints.P_cbf_list)):
            #     cbf_center = self.cbf_constraints.constraint_list[constraint_idx][2]
            #     cbf_radius = self.cbf_constraints.constraint_list[constraint_idx][3]
            #     for t in range(1, self.horizon):
            #         start_idx = t * self.transition_dim
            #         end_idx = (t + 1) * self.transition_dim
            #         def constraint_fun_local(x, start_idx=start_idx, end_idx=end_idx,cbf_center=cbf_center, cbf_radius=cbf_radius, constraints=self.cbf_constraints, unnormalizer=self.normalizer.unnormalize, dt = self.cbf_constraints.dt.detach().cpu().numpy().tolist()):
            #             lambda_cbf = 1
            #             alpha_cbf = 1
            #             state = x[:-1].reshape(constraints.horizon, constraints.transition_dim)
            #             state = unnormalizer(state).flatten()
            #             hpk_in =  ((state[start_idx + 4] - cbf_center[0])**2 + (state[start_idx +5] - cbf_center[1])**2 - cbf_radius**2)
            #             hpktuk_in = ((dt * state[start_idx] +state[start_idx+4] - cbf_center[0])**2 + (dt * state[start_idx+1] + state[start_idx+5] - cbf_center[1])**2 - cbf_radius**2)
            #             hpk = hpk_in
            #             hpktuk = hpktuk_in
            #             alpha_hpk = alpha_cbf * hpk
            #             return hpktuk - hpk + alpha_hpk
            #             # return -x[start_idx:end_idx] @ P @ x[start_idx:end_idx] - q @ x[start_idx:end_idx] + v
                    
            #         def constraint_jac_local(x, start_idx=start_idx, end_idx=end_idx,cbf_center=cbf_center, cbf_radius=cbf_radius,constraints=self.cbf_constraints, unnormalizer=self.normalizer.unnormalize, dt = self.cbf_constraints.dt.detach().cpu().numpy().tolist()):
            #             # x_cbf = x[:-1].reshape(constraints.horizon, constraints.transition_dim)
            #             # x_cbf = self.normalizer.unnormalize(x_cbf)
            #             # x_cbf = torch.tensor(x_cbf, device=self.device, dtype=torch.float32)

            #             # grad_cbf = self.single_cbf_grad_fn(x_cbf[:,:2], x_cbf[:,4:6], self.external_obstacle_pos_tensor[0,:,:2, self.env.step_counter:self.env.step_counter+self.horizon], self.external_obstacle_pos_tensor[0,:,2:, self.env.step_counter:self.env.step_counter+self.horizon], 0.04)
            #             lambda_cbf = 1
            #             alpha_cbf = 1
            #             state = x[:-1].reshape(constraints.horizon, constraints.transition_dim)
            #             state = unnormalizer(state).flatten()
            #             gradient_array = np.zeros(len(x))
            #             # for constraint_idx in range(len(constraints.P_cbf_list)):
            #             cbf_center =constraints.constraint_list[constraint_idx][2]
            #             # cbf_radius =constraints.constraint_list[constraint_idx][3]
            #             grad_vx = 2*dt*(dt*state[start_idx] - cbf_center[0] + state[start_idx+ 4])
            #             grad_vy = 2*dt*(dt*state[start_idx + 1] - cbf_center[1] + state[start_idx + 5])
            #             grad_x =  2* alpha_cbf * (state[start_idx+ 4] - cbf_center[0]) + 2 *dt *state[start_idx]
            #             grad_gx =  2* alpha_cbf * (state[start_idx+ 2] - cbf_center[0]) + 2 *dt *state[start_idx]
            #             grad_y =  2* alpha_cbf * (state[start_idx+ 5] - cbf_center[1]) + 2 *dt *state[start_idx + 1]
            #             grad_gy =  2* alpha_cbf * (state[start_idx+ 3] - cbf_center[1]) + 2 *dt *state[start_idx + 1]
            #             gradient_array[start_idx:end_idx] = np.array([grad_vx, grad_vy, grad_gx, grad_gy, grad_x, grad_y])
            #             return gradient_array
                            
            #         constraints += ({'type': 'ineq', 
            #                         'fun': constraint_fun_local,
            #                         'jac': constraint_jac_local},)
            if C.size > 0:
                constraints += ({'type': 'ineq', 'fun': lambda x: -C @ x + d, 'jac': lambda x: -C},)
            if A.size > 0:
                constraints += ({'type': 'eq', 'fun': lambda x: A @ x - b, 'jac': lambda x: A},)   
            
            projection_costs = np.ones(batch_size, dtype=np.float32)
            sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)
            

            plot_log_data = {}
            for i in range(batch_size):

                self.plot_data['cost'] = []
                self.plot_data['timesteps'] = []
                # Cost
                cost_fun = self.objective # + (A_double @ x - b_double) @ (A_double @ x - b_double)
                jac_cost_fun = self.jacobian
                x0 = np.concatenate([trajectory_np_double[i], np.array([5])])
                lower_bound =-5 * np.ones_like(x0)
                upper_bound = 5 * np.ones_like(x0)
                lower_bound[-1] = 0.0
                upper_bound[-1] = 5.0
                res = minimize(fun=cost_fun, 
                                x0=x0,
                                args=(self.Q_np, trajectory_np_double[i] ,model, cond, apply_conditioning, timesteps),  # Pass arguments for the objective function
                                constraints=constraints, 
                                method='SLSQP', 
                                jac=jac_cost_fun, 
                                bounds=Bounds(lower_bound, upper_bound),
                                tol=1e-6,
                                options={'maxiter': 100, 'disp': False})

                sol_np[i] = res.x[:-1]
                projection_costs[i] = self.compute_projection_costs(sol_np[i], trajectory_np[i], self.Q_np, trajectory_np_double[i], i)
                # if not res.success:
                #     print(f'Projection failed for trajectory {i}')
                #     projection_costs[i] += 1000000000

                plot_log_data[f'cost{i}'] = self.plot_data['cost'].copy()
                plot_log_data[f'timesteps{i}'] = self.plot_data['timesteps'].copy()


                # if np.linalg.norm(A_double @ res.x - b_double) > 1e-3:
                #     print('Equality constraints not satisfied!')
                # if np.any(C_double @ res.x > d_double + 1e-3):
                #     print('Inequality constraints not satisfied!')

                # for constraint_idx in range(len(self.cbf_constraints.P_cbf_list)):
                #     P = self.cbf_constraints.P_cbf_list[constraint_idx]
                #     q = self.cbf_constraints.q_cbf_list[constraint_idx]
                #     v = self.cbf_constraints.v_cbf_list[constraint_idx]
                #     for t in range(1, self.horizon):
                #         start_idx = t * self.transition_dim
                #         end_idx = (t + 1) * self.transition_dim
                #         x_sol = sol_np[i, start_idx:end_idx]
                #         constraint_val = -x_sol @ P @ x_sol - q @ x_sol + v
                #         print(f"  Constraint {constraint_idx}, t={t}: Value={constraint_val:.6f}")

            sol = torch.tensor(sol_np, device=self.device).reshape(dims)
            if self.plot_debug:
                # Create first figure with 3 plots per batch item
                fig, ax = plt.subplots(batch_size, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 2]})
                colors = plt.cm.rainbow(np.linspace(0, 1, batch_size))
                for i in range(batch_size):
                    ax[i, 0].plot(range(len(plot_log_data[f'timesteps{i}'])), plot_log_data[f'timesteps{i}'], color='#1f77b4', linewidth=2, marker='o', markersize=4)
                    ax[i, 0].set_title('Evolution of Timesteps During Optimization')
                    ax[i, 0].set_xlabel('Iteration Number')
                    ax[i, 0].set_ylabel('Timestep Value')
                    ax[i, 0].grid(True, linestyle='--', alpha=0.7)
                    ax[i, 0].spines['top'].set_visible(False)

                    ax[i, 1].plot(range(len(plot_log_data[f'cost{i}'])), plot_log_data[f'cost{i}'], color='#ff7f0e', linewidth=2, marker='o', markersize=4)
                    ax[i, 1].set_title('Optimization Cost Over Iterations')
                    ax[i, 1].set_xlabel('Iteration Number')
                    ax[i, 1].set_ylabel('Cost Value')
                    ax[i, 1].grid(True, linestyle='--', alpha=0.7)
                    ax[i, 1].spines['top'].set_visible(False)

                plt.tight_layout()
                # Save the first figure (optimization plots)
                timestamp = int(time.time() * 1000)
                os.makedirs(f'{save_dir}/mpc', exist_ok=True)
                optimization_path = f'{save_dir}/mpc/optimization_{timestamp:020d}.png'
                fig.savefig(optimization_path, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Create second figure for trajectory visualization
                fig_debug = plt.figure(figsize=(10, 10))
                ax_debug = fig_debug.add_subplot(111)
                
                # Plot trajectories
                for i in range(batch_size):
                    x_plot = sol_np[i].reshape(self.horizon, self.transition_dim)
                    x_plot = self.normalizer.unnormalize(x_plot)
                    ax_debug.scatter(x_plot[:, 4], x_plot[:, 5], 
                                    color=colors[i], 
                                    s=30, 
                                    label=f'Trajectory Cost {projection_costs[i]:.2f}',
                                    alpha=0.7)
                
                # Add obstacles to trajectory plot
                obstacle_env_data = [self.env.get_obstacle_position()]
                centers = []
                for center_pos_key in list(obstacle_env_data[-1].keys())[:6]:
                    centers.append(obstacle_env_data[-1][center_pos_key][:2])
                for center in centers:
                    ax_debug.add_patch(matplotlib.patches.Circle(center, 0.025, color='r'))
                
                # Add constraint circles
                for k in range(len(self.config_obstacle_constraints)):
                    pos_i = self.config_obstacle_constraints[k]['center']
                    r_i = self.config_obstacle_constraints[k]['radius']
                    ax_debug.add_patch(matplotlib.patches.Circle(pos_i, r_i, color='b', alpha=0.2))
                
                # Add finish line to trajectory plot
                ax_debug.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
                
                # Add legend and title to trajectory plot
                ax_debug.legend()
                ax_debug.set_title('Optimized Trajectories')
                ax_debug.set_xlabel('X Position')
                ax_debug.set_ylabel('Y Position')
                ax_debug.grid(True, linestyle='--', alpha=0.7)
                ax_debug.set_aspect('equal')
                
                # Save the second figure (trajectory plot)
                trajectory_path = f'{save_dir}/mpc/trajectory_{timestamp:020d}.png'
                fig_debug.savefig(trajectory_path, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig_debug)
            # Project each solution in the batch onto the feasible set
            # sol_np_proj = np.zeros_like(sol_np)
            # for i in range(batch_size):
            #     sol_np_proj[i] = self.project_to_constraints(sol_np[i])
            # sol = torch.tensor(sol_np_proj, device=self.device).reshape(dims)
            
            # print(f'Projection time {self.solver}:', time.time() - start_time)
            return sol, projection_costs    # only implemented for proxsuite and scipy and parallelize=False
        

    def project(self, trajectory, constraints=None):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim)
            Solve an optimization problem of the form 
                \hat z =   argmin_z 1/2 z^T Q z + r^T z
                        subject to  Az  = b
                                    Cz <= d
            where z = (o_0, o_1, ..., o_{H-1}) is the trajectory in vector form. The matrices A, b, C, and d are defined by the dynamic and safety constraints.
                                    
        """
        
        dims = trajectory.shape

        # Reshape the trajectory to a batch of vectors (from B x H x T to B x (HT)
        batch_size = trajectory.shape[0]
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)

        # Cost
        r = - trajectory_reshaped @ self.Q
        r_np = r.cpu().numpy()
        Q = self.Q_np.astype('double')
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints
        A = self.A_np.astype('double')
        b = self.b_np.astype('double')
        C = self.C_np.astype('double')
        d = self.d_np.astype('double')

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            if self.solver == 'proxsuite' or self.solver == 'gurobi':
                s_0 = s_0.cpu().numpy()
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        r_np_double = r_np.astype('double')
        trajectory_np_double = trajectory_np.astype('double')
        # Constraints
        constraints = ()
        for obs_idx, obs_key in enumerate(self.env.get_obstacle_position().keys()):
                obstacle_pos = self.env.get_obstacle_position()[obs_key]
                self.obstacle_constraints.constraint_list[obs_idx +  1][2] = obstacle_pos[:2]
        
        self.obstacle_constraints.build_matrices()               
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = self.obstacle_constraints.P_list[constraint_idx]
            q = self.obstacle_constraints.q_list[constraint_idx]
            v = self.obstacle_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):                        # Obstacle constraints
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                constraints += ({'type': 'ineq', 'fun': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q, v=v: -x[start_idx: end_idx] @ P @ x[start_idx: end_idx] - q @ x[start_idx: end_idx] + v,
                                    'jac': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q: np.concatenate([np.zeros(start_idx), -2 * P @ x[start_idx: end_idx] - q, np.zeros(len(x) - end_idx)])},)

        if C.size > 0:
            constraints += ({'type': 'ineq', 'fun': lambda x: -C @ x + d, 'jac': lambda x: -C},)
        if A.size > 0:
            constraints += ({'type': 'eq', 'fun': lambda x: A @ x - b, 'jac': lambda x: A},)   
        
        projection_costs = np.ones(batch_size, dtype=np.float32)
        sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)
        for i in range(batch_size):
            # Cost
            cost_fun = lambda x: 0.5 * x @ Q @ x + r_np_double[i] @ x # + (A_double @ x - b_double) @ (A_double @ x - b_double)
            jac_cost_fun = lambda x: Q @ x + r_np_double[i]
            res = minimize(fun=cost_fun, 
                            x0=trajectory_np_double[i],
                            constraints=constraints, 
                            method='SLSQP', 
                            jac=jac_cost_fun, 
                            bounds=Bounds(-5 * np.ones_like(trajectory_np_double[i]), 5 * np.ones_like(trajectory_np_double[i])),
                            tol=1e-6,
                            options={'maxiter': 1000, 'disp': False})

            sol_np[i] = res.x
            projection_costs[i] = 0.5 * sol_np[i] @ Q @ sol_np[i] + r_np[i] @ sol_np[i] + 0.5 * trajectory_np[i] @ Q @ trajectory_np[i]

            # if np.linalg.norm(A_double @ res.x - b_double) > 1e-3:
            #     print('Equality constraints not satisfied!')
            # if np.any(C_double @ res.x > d_double + 1e-3):
            #     print('Inequality constraints not satisfied!')

        sol = torch.tensor(sol_np, device=self.device).reshape(dims)

        # print(f'Projection time {self.solver}:', time.time() - start_time)
        return sol, projection_costs    # only implemented for proxsuite and scipy and parallelize=False
    
    def compute_gradient(self, trajectory, constraints=None):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim) or (horizon, transition_dim)
            Calculate the (weighted) gradients for the following cost functions:
            c_1 = ||A * tau - b||^2                             --> grad_1 = 2 * A^T (A * tau - b)
            c_2 = max(0, C * tau - d)^2                         --> grad_2 = 2 * C^T max(0, C * tau - d)
            c_3 = sum_{t=1}^{H-1} (s_t^T P s_t + q^T s_t - v)^2 --> grad_3 = 2 * (2 * P s_t + q) * (s_t^T P s_t + q^T s_t - v)                 
        """
        
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints
        A, b, C, d = self.A, self.b, self.C, self.d

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        # Equality and polytopic constraints
        grad1 = torch.zeros_like(trajectory_reshaped)
        grad2 = torch.zeros_like(trajectory_reshaped)
        for i in range(trajectory.shape[0]):
            grad1[i] = - A.T @ (A @ trajectory_reshaped[i] - b)
            grad2[i] = - C.T @ torch.max(torch.zeros_like(C @ trajectory_reshaped[i] - d), C @ trajectory_reshaped[i] - d)
        grad1 = grad1.reshape(trajectory.shape)
        grad2 = grad2.reshape(trajectory.shape)

        # Obstacle constraints
        grad3 = np.zeros_like(trajectory_np)
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = self.obstacle_constraints.P_list[constraint_idx]
            q = self.obstacle_constraints.q_list[constraint_idx]
            v = self.obstacle_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                for i in range(trajectory.shape[0]):
                    if trajectory_np[i, start_idx: end_idx] @ P @ trajectory_np[i, start_idx: end_idx] + q @ trajectory_np[i, start_idx: end_idx] <= v:
                        continue
                    else:
                        grad3[i, start_idx: end_idx] -= 2 * P @ trajectory_np[i, start_idx: end_idx] + q   
        grad3 = torch.tensor(grad3, device=self.device).reshape(trajectory.shape)
        
        if self.gradient_weights is not None:
            grad1 = grad1 * self.gradient_weights[0]
            grad2 = grad2 * self.gradient_weights[1]
            grad3 = grad3 * self.gradient_weights[2]
        
        return grad1 + grad2 + grad3 

    def append_linear_constraint(self, constraint):
        self.C = torch.cat([self.C, constraint.C], dim=0)
        self.d = torch.cat([self.d, constraint.d], dim=0)
        self.A = torch.cat([self.A, constraint.A], dim=0)
        self.b = torch.cat([self.b, constraint.b], dim=0)
        if constraint.__class__.__name__ == 'SafetyConstraints':
            self.A_safe, self.b_safe, self.C_safe, self.d_safe = constraint.A, constraint.b, constraint.C, constraint.d
        elif constraint.__class__.__name__ == 'DynamicConstraints':
            self.A_dyn, self.b_dyn, self.C_dyn, self.d_dyn = constraint.A, constraint.b, constraint.C, constraint.d

    def add_numpy_constraints(self):
        self.A_np = self.A.cpu().numpy()
        self.b_np = self.b.cpu().numpy()     
        self.C_np = self.C.cpu().numpy()
        self.d_np = self.d.cpu().numpy()
        self.Q_np = self.Q.cpu().numpy()


class Constraints:

    def __init__(self, horizon, transition_dim, normalizer=None, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.normalizer = normalizer
        self.device = device

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.b = torch.empty(0, device=device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.d = torch.empty(0, device=device)

    def build_matrices(self):
        pass

class SafetyConstraints(Constraints):

    def __init__(self, skip_initial_state=True, action_dim=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.action_dim = action_dim
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('lb', [-1.0, -inf, 0]), ('ub', [1.0, 2.0, inf]), ('eq', ([0, 1, 1], 1.5)), ('ineq': ([1, 0, 0], 0.5))] -->
                    x_0 in [-1, 1], x_1 in [-inf, 2], x_2 in [0, inf], 0 * x_0 + 1 * x_1 + 1 * x_2 = 1.5, 1 * x_0 + 0 * x_1 + 0 * x_2 <= 0.5,
                    where x_i is the i-th dimension of the state or state-action vectors
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
                lb: horizon * transition_dim
            C consists of n_bounds blocks of shape (horizon, transition_dim * horizon), where each block corresponds to a 
            constraint (ub or lb) on a specific dimension. The block has a 1 or -1 at the corresponding dimension and time step.
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        for constraint in constraint_list:
            type = constraint[0]
            bound = constraint[1]
            if type == 'lb' or type == 'ub':
                for dim in range(len(bound)):
                    if bound[dim] == -np.inf or bound[dim] == np.inf:
                        continue
                    
                    mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
                    vec_append = torch.zeros(self.horizon, device=self.device)

                    sign = 1 if type == 'ub' else -1
                    for t in range(self.horizon):
                        mat_append[t, t * self.transition_dim + dim] = sign
                        vec_append[t] = sign * bound[dim]
                        
                    if self.normalizer is not None:
                        x_min = self.normalizer.mins[dim]
                        x_max = self.normalizer.maxs[dim]
                        mat_append = mat_append * (x_max - x_min) / 2
                        vec_append = vec_append - sign * (x_min + x_max) / 2

                    if self.skip_initial_state and dim >= self.action_dim:
                        mat_append = mat_append[1:]
                        vec_append = vec_append[1:]

                    self.C = torch.cat((self.C, mat_append), dim=0)
                    self.d = torch.cat((self.d, vec_append), dim=0)
                continue         

            # type == 'eq' or 'ineq'
            mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
            vec_append = torch.zeros(self.horizon, device=self.device)

            for i in range(self.horizon):
                if self.normalizer is not None:
                    x_min = self.normalizer.mins
                    x_max = self.normalizer.maxs

                    # Unnormalize the constraints. TODO: Extend to actions by checking if dim <= action_dim
                    # We have Cs <= d, where s is the unnormalized state vector. This is converted to C's_n <= d', where s_n is the normalized state vector,
                    # by using the fact that s = (s_n + 1) * (s_max - s_min) / 2 + s_min.
                    a = bound[0] * (x_max - x_min) / 2
                    b = bound[1] - bound[0] @ (x_max + x_min) / 2
                else:
                    a = bound[0]
                    b = bound[1]
                
                mat_append[i, i * self.transition_dim: (i + 1) * self.transition_dim] = torch.tensor(a, device=self.device)
                vec_append[i] = torch.tensor(b, device=self.device)

            if self.skip_initial_state:
                mat_append = mat_append[1:]
                vec_append = vec_append[1:]

            if type == 'eq':
                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)
            else:
                self.C = torch.cat((self.C, mat_append), dim=0)
                self.d = torch.cat((self.d, vec_append), dim=0)         

class DynamicConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('deriv', [0, 2]), ('deriv', [1, 3])] -->
                    x_0[t+1] = x_0[t] + self.dt * x_2[t], x_1[t+1] = x_1[t] + self.dt * x_3[t]                                      (explicit Euler) or
                    x_0[t+1] = x_0[t] + self.dt * (x_2[t] + x_2[t+1]) / 2, x_1[t+1] = x_1[t] + self.dt * (x_3[t] + x_3[t+1]) / 2    (variant of trapezoidal rule)
                    where x_i[t] is the i-th dimension of the state or state-action vectors at time t
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)
        
        for constraint in constraint_list:
            type = constraint[0]
            vals = constraint[1]
            if 'deriv' in type:
                x_idx = int(vals[0])
                dx_idx = int(vals[1])
            
                mat_append = torch.zeros(self.horizon - 1, self.transition_dim * self.horizon, device=self.device)
                vec_append = torch.zeros(self.horizon - 1, device=self.device)

                # Calculate multiplicative factors needed for normalization
                if self.normalizer is not None: 
                    x_min = self.normalizer.mins[x_idx]
                    x_max = self.normalizer.maxs[x_idx]
                    dx_min = self.normalizer.mins[dx_idx]
                    dx_max = self.normalizer.maxs[dx_idx]
                    x_diff = x_max - x_min
                    dx_diff = dx_max - dx_min
                    dx_sum = dx_max + dx_min

                for i in range(self.horizon - 1):
                    if self.normalizer is not None:
                        mat_append[i, i * self.transition_dim + x_idx] = 1 * x_diff
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt * dx_diff
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1 * x_diff
                        vec_append[i] = - dx_sum * self.dt
                    else:
                        mat_append[i, i * self.transition_dim + x_idx] = 1
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1
                        vec_append[i] = 0

                if self.skip_initial_state:     # --> Do that in the projection method because it needs the current state. For that, record the relevant rows
                    mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)    # Fix the initial state
                    mat_fix_initial[0, x_idx] = 1
                    mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                    vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)          # Must be changed to current state in each iteration!

                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)

class ObstacleConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('sphere_inside', [0, 2] [-1, 5], 1), ('sphere_outside', [1, 3], [0, 1], 4)] -->
                    (x_0 + 1)^2 + (x_2 - 5)^2 <= 1, x_1^2 + (x_3 - 1)^2 >= 4
                    where x_i is the i-th dimension of the state or state-action vectors at time t
            Generate the matrix P and the vector q for:
                s_i^T P s_i + q^T s_i <= v
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        self.P_list = []
        self.q_list = []
        self.v_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            center = constraint[2]
            radius = constraint[3]

            P = np.zeros((self.transition_dim, self.transition_dim))
            q = np.zeros(self.transition_dim)
            v = radius ** 2

            dim_counter = 0
            for dim in dims:
                if self.normalizer is not None:
                    delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                    s_min = self.normalizer.mins[dim]
                    P[dim, dim] = delta_s ** 2 / 4
                    q[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                    v -= delta_s**2 / 4 + delta_s * (s_min - center[dim_counter]) + (s_min - center[dim_counter]) ** 2
                else:
                    P[dim, dim] = 1
                    q[dim] = -2 * center[dim_counter]
                    v -= center[dim_counter] ** 2
                dim_counter += 1

            if type == 'sphere_outside':
                P = -P
                q = -q
                v = -v

            self.P_list.append(P)
            self.q_list.append(q)
            self.v_list.append(v)    

class CBF(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('sphere_inside', [0, 2] [-1, 5], 1), ('sphere_outside', [1, 3], [0, 1], 4)] -->
                    (x_0 + 1)^2 + (x_2 - 5)^2 <= 1, x_1^2 + (x_3 - 1)^2 >= 4
                    where x_i is the i-th dimension of the state or state-action vectors at time t
            Generate the matrix P and the vector q for:
                s_i^T P s_i + q^T s_i <= v
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        self.P_cbf_list = []
        self.q_cbf_list = []
        self.v_cbf_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            center = constraint[2]
            radius = constraint[3]

            P_cbf = np.zeros((self.transition_dim, self.transition_dim))
            q_cbf = np.zeros(self.transition_dim)
            alpha = 1
            b_cbf = 0
            
            #Barrier function
            
            dim_counter = 0
            for dim in dims:
                if self.normalizer is not None:
                    delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                    s_min = self.normalizer.mins[dim]
                    P_cbf[dim, dim] = alpha * delta_s ** 2 / 4
                    P_cbf[dim-4, dim] = delta_s ** 2 / 4
                    P_cbf[dim, dim-4] = delta_s ** 2 / 4
                    q_cbf[dim] = -alpha * delta_s * center[dim_counter] + alpha * delta_s * s_min + 0.5 * alpha * delta_s**2
                    q_cbf[dim-4] = 0.5 * delta_s**2 + delta_s * s_min - delta_s * center[dim_counter]
                    # q_cbf[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                    b_cbf = b_cbf + alpha * center[dim_counter]**2 - (2*alpha * s_min * center[dim_counter] + alpha * delta_s *center[dim_counter]) + alpha * s_min**2 + alpha*(delta_s**2)/4 +  delta_s*alpha*s_min 
                else:
                    P_cbf[dim, dim] = (2 + alpha)
                    q_cbf[dim] = -2 * center[dim_counter] * (alpha + 1)
                    b_cbf =  b_cbf + alpha * center[dim_counter] ** 2
                dim_counter += 1

            b_cbf = b_cbf -alpha  * radius ** 2
            if type == 'sphere_outside':
                P_cbf = -P_cbf
                q_cbf = -q_cbf
                b_cbf = -b_cbf

            self.P_cbf_list.append(P_cbf)
            self.q_cbf_list.append(q_cbf)
            self.v_cbf_list.append(b_cbf)  

class ProjectionNormalizer():
    def __init__(self, observation_normalizer=None, action_normalizer=None, goal_dim=0):
        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer
        self.goal_dim = goal_dim
        self.get_limits()

    def get_limits(self):
        if self.observation_normalizer is not None and self.action_normalizer is not None:
            x_max_obs = self.observation_normalizer.maxs[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.maxs
            x_min_obs = self.observation_normalizer.mins[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.mins
            x_max = np.concatenate([self.action_normalizer.maxs, x_max_obs])
            x_min = np.concatenate([self.action_normalizer.mins, x_min_obs])
        elif self.observation_normalizer is not None:
            x_max = self.observation_normalizer.maxs[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.maxs
            x_min = self.observation_normalizer.mins[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.mins
        elif self.action_normalizer is not None:
            x_max = self.action_normalizer.maxs
            x_min = self.action_normalizer.mins
        
        self.maxs = x_max
        self.mins = x_min

    def normalize(self, x):
        x_normalized = (x - self.mins) / (self.maxs - self.mins) * 2 - 1
        return x_normalized
    
    def unnormalize(self, x_normalized):
        x = (x_normalized + 1) * (self.maxs - self.mins) / 2 + self.mins
        return x
                