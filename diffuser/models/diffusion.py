import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
import os
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # For colormaps
import numpy as np  # For array ops
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
from diffuser.models.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import yaml

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, goal_dim=0, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
        loss_discount=1.0, loss_weights=None, returns_condition=False, condition_guidance_w=0.1,):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        trajectory_file = config.get('trajectory_file', 'generated_trajectories.npy')
        self.external_obstacle_pos =  np.load(trajectory_file)
        single_cbf_grad_fn = torch.func.grad(single_cbf_reward_fn_pairwise)
        single_clf_grad_fn = torch.func.grad(single_clf_reward_fn)
        self.batched_cbf_grad_fn = torch.vmap(single_cbf_grad_fn, in_dims=(0, 0, None, None, None))
        self.batched_clf_grad_fn = torch.vmap(single_clf_grad_fn, in_dims=(0, 0, None))

        self.batched_over_obst = torch.vmap(self.compute_h, in_dims=(None, 0, None))
        self.batched_over_batch = torch.vmap(self.batched_over_obst, in_dims=(0, None, None))

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None, env=None, config_obstacle_constraints=None, polytopic_constraints=None, save_dir=None, iteration=None):
        # if self.model.calc_energy:
        #     assert self.predict_epsilon
        #     x = torch.tensor(x, requires_grad=True)
        #     t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        #     returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)

        if projector is not None and projector.gradient:
            if self.goal_dim > 0:
                grad = projector.compute_gradient(x_recon[:,:,:-self.goal_dim], constraints)
            else:
                grad = projector.compute_gradient(x_recon, constraints)
            model_mean = model_mean + grad

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, projector=None, constraints=None, env=None, config_obstacle_constraints=None, polytopic_constraints=None, save_dir=None, iteration=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns, projector=projector, constraints=constraints)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    def compute_h(self, ego_s, obst_s, r):
        pairwise = torch.cat([ego_s.T, obst_s], dim=0)
        ego_state = pairwise[:2, :]
        obst = pairwise[2:4, :]
        return (ego_state[0] - obst[0])**2 + (ego_state[1] - obst[1])**2 - r**2


    def compute_grad_cbf(self, x_cbf, external_obstacle_pos, env):

        grad_clf = self.batched_clf_grad_fn(x_cbf[:,:,:2],x_cbf[:,:,5:6],torch.tensor([[0.35]]).to(x_cbf.device))
        # single_clf_reward_fn(x_cbf[0,:,:2],x_cbf[0,:,4:6],torch.tensor([[0.5,0.35]]).to(x_cbf.device))
        # single_clf_grad_fn = torch.func.grad(single_clf_reward_fn)

        grad_cbf = self.batched_cbf_grad_fn(x_cbf[:,:,:2], x_cbf[:,:,4:6], external_obstacle_pos[0,:,:2, env.step_counter:env.step_counter+self.horizon], external_obstacle_pos[0,:,2:, env.step_counter:env.step_counter+self.horizon], 0.04)
        h_x_all = self.batched_over_batch(x_cbf[:,:,4:6], external_obstacle_pos[0,:,:2, env.step_counter:env.step_counter+self.horizon], 0.04)
        grad_cbf_norm = torch.norm(grad_cbf, dim=-1, keepdim=True)
        grad_clf_norm = torch.norm(grad_clf, dim=-1, keepdim=True)
        normalized_grad_cbf = grad_cbf / (grad_cbf_norm+1e-8)
        normalized_grad_clf = grad_clf / (grad_clf_norm+1e-8)
        epsilon_threshold = 0.01
        mask = (h_x_all.min(dim=-1).values.min(dim=-1).values <= epsilon_threshold).unsqueeze(-1).unsqueeze(-1).float()
        normalized_grad_cbf = normalized_grad_cbf * mask.float()

        min_h = h_x_all.min(dim=-1).values.min(dim=-1).values  # Shape: [batch], or adjust dims as needed
        min_h = min_h.clamp(min=-1e6, max=1e6)  # Prevent extremes
        # Adaptive scalar: increases as min_h decreases (closer to obstacle)
        base_value = 5.0  # Minimum when far (tune this)
        amp = 10.0  # Max boost (tune: higher = stronger push when close)
        k = 1.0  # Sensitivity (tune: higher = faster growth)
        epsilon = 1e-3  # Avoid div-by-zero
        adaptive_scalar_cbf = base_value + amp * torch.exp(-k / (min_h + epsilon))
        adaptive_scalar_cbf = adaptive_scalar_cbf.clamp(min=1.0, max=15.0)  #


        return normalized_grad_cbf, normalized_grad_clf, adaptive_scalar_cbf


    def cbf_plot(self, x_cbf, external_obstacle_pos, env, normalized_grad_cbf, normalized_grad_clf, config_obstacle_constraints, polytopic_constraints, save_dir):
        colors = plt.cm.rainbow(np.linspace(0, 1, x_cbf.shape[0]))
        
        pos_x_all = x_cbf[:, :, 4].clone().detach().cpu().numpy()  # [batch=4, horizon=8]
        pos_y_all = x_cbf[:, :, 5].clone().detach().cpu().numpy()
        normalized_grad_cbf_np = normalized_grad_cbf.clone().detach().cpu().numpy()  # [4,8,2] assuming (batch, horizon, xy)
        normalized_grad_clf_np = normalized_grad_clf.clone().detach().cpu().numpy()  # [4,8,2]
        grad_x_all = normalized_grad_cbf_np[:, :, 0]  # [4,8]
        grad_y_all = normalized_grad_cbf_np[:, :, 1]  # [4,8]
        grad_clf_x_all = normalized_grad_clf_np[:, :, 0]  # [4,8]
        grad_clf_y_all = normalized_grad_clf_np[:, :, 1]  # [4,8]
        
        # Create figure (single plot; use subplots if you want one per batch)
        fig_debug, ax_debug = plt.subplots(figsize=(10, 10))
        
        # Global magnitudes for colorbar (for CBF; adapt for CLF if needed)
        cbf_gradients = np.stack([grad_x_all, grad_y_all], axis=-1)  # [4,8,2]
        all_magnitudes = np.linalg.norm(cbf_gradients, axis=-1).flatten()  # Flatten for global min/max
        min_mag = all_magnitudes.min()
        max_mag = all_magnitudes.max()
        
        for batch_idx in range(x_cbf.shape[0]):
            # Per-batch data
            batch_pos_x = pos_x_all[batch_idx]  # [8]
            batch_pos_y = pos_y_all[batch_idx]  # [8]
            batch_grad_x = grad_x_all[batch_idx]  # [8]
            batch_grad_y = grad_y_all[batch_idx]  # [8]
            batch_grad_clf_x = grad_clf_x_all[batch_idx]  # [8]
            batch_grad_clf_y = grad_clf_y_all[batch_idx]  # [8]
            
            # Stack to vectors [8,2]
            batch_cbf_grad = np.stack([batch_grad_x, batch_grad_y], axis=-1)  # [8,2]
            batch_clf_grad = np.stack([batch_grad_clf_x, batch_grad_clf_y], axis=-1)  # [8,2]
            
            # Explicit normalization for uniform arrow lengths (unit vectors)
            cbf_norms = np.linalg.norm(batch_cbf_grad, axis=-1, keepdims=True)  # [8,1]
            cbf_norms[cbf_norms == 0] = 1.0  # Avoid div-by-zero
            batch_cbf_grad_norm = batch_cbf_grad / cbf_norms  # Unit length [8,2]
            
            clf_norms = np.linalg.norm(batch_clf_grad, axis=-1, keepdims=True)  # [8,1]
            clf_norms[clf_norms == 0] = 1.0
            batch_clf_grad_norm = batch_clf_grad / clf_norms  # Unit length [8,2]
            
            # Magnitudes for coloring (use original, non-normalized for color)
            cbf_magnitudes = np.linalg.norm(batch_cbf_grad, axis=-1)  # [8] (original lengths for color)
            cbf_magnitudes_norm = (cbf_magnitudes - min_mag) / (max_mag - min_mag + 1e-8)  # [8], [0,1]
            
            # Colormap for CBF (per batch)
            cmap = cm.jet
            cbf_colors = cmap(cbf_magnitudes_norm)  # [8,4] RGBA
            
            # Plot CBF arrows (normalized for uniform length, colored by original magnitude)
            ax_debug.quiver(
                batch_pos_x, batch_pos_y,  # X, Y
                batch_cbf_grad_norm[:, 0], batch_cbf_grad_norm[:, 1],  # Normalized U, V
                color=cbf_colors,  # Color by magnitude
                scale=20.0,  # Uniform scaling (tune: smaller = longer arrows)
                width=0.005, alpha=0.5
            )
            
            # Plot CLF arrows (normalized, fixed color 'green')
            ax_debug.quiver(
                batch_pos_x, batch_pos_y,
                batch_clf_grad_norm[:, 0], batch_clf_grad_norm[:, 1],
                color='green',  # Fixed color
                scale=20.0,  # Same scale as CBF for consistency
                width=0.005
            )
            
            # Scatter trajectory points (using your colors)
            ax_debug.scatter(batch_pos_x, batch_pos_y, color=colors[batch_idx], s=30, label=f'Trajectory {batch_idx+1}', alpha=0.7)
        
        # Plot obstacles (your existing code)
        ax_debug.scatter(self.external_obstacle_pos[0,:,0, env.step_counter:env.step_counter+self.horizon].flatten(), 
                         self.external_obstacle_pos[0,:,1, env.step_counter:env.step_counter+self.horizon].flatten(), 
                         color='red', s=150, label='Obstacle', alpha=0.7)
        
        # Plot finish line
        ax_debug.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
        
        # Plot additional constraint areas

        for k in range(len(config_obstacle_constraints)):
            pos_i = config_obstacle_constraints[k]['center']
            r_i = config_obstacle_constraints[k]['radius']
            ax_debug.add_patch(matplotlib.patches.Circle(pos_i, r_i, color='b', alpha=0.2))

        # Add legend
        ax_debug.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_limits = [[0.2, 0.8], [-0.3, 0.4]]   # TODO: get from env
        utils.plot_halfspace_constraints('avoiding-d3il', polytopic_constraints, ax_debug, ax_limits)
        # Set axis limits and grid
        ax_debug.set_xlim(ax_limits[0])
        ax_debug.set_ylim(ax_limits[1])
        ax_debug.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=min_mag, vmax=max_mag))
        cbar = fig_debug.colorbar(sm, ax=ax_debug, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('CBF Gradient Magnitude')

        # Save the figure with timestep in filename
        timestamp = int(time.time() * 1000)  
        debug_save_path = f'{save_dir}/diffusion/arrows_{timestamp:04d}.png'
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)

        data_directory = save_dir[: save_dir.find('/logs/')] + "/d3il/environments/dataset/data/avoiding/data/"

        state_files = os.listdir(data_directory)
        files_to_plot = state_files

        for file in files_to_plot:
            with open(os.path.join(data_directory, file), 'rb') as f:
                env_state = pickle.load(f)
                
                robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
                robot_c_pos = env_state['robot']['c_pos'][:, :2]
                
                # Plot desired trajectory
                plt.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], 'b-', alpha=0.05, label='Desired trajectory' if file == files_to_plot[0] else "")
                
                # Plot actual trajectory
                plt.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], 'r-', alpha=0.05, label='Actual trajectory' if file == files_to_plot[0] else "")
                
                # Plot start points
                plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^', label='Start points' if file == files_to_plot[0] else "")
                
                # Plot end points
                plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x', label='End points' if file == files_to_plot[0] else "")

        plt.savefig(debug_save_path, bbox_inches='tight', dpi=150)
        plt.close(fig_debug)


    def p_mean_variance_diffmpc(self, x, cond, t, returns=None, projector=None, constraints=None, env=None, config_obstacle_constraints=None, polytopic_constraints=None, save_dir=None, iteration=None):

        device = self.betas.device
        external_obstacle_pos = torch.tensor(self.external_obstacle_pos, device=device, dtype=torch.float32)

        x_cbf = x.clone().detach().cpu().numpy()
        x_cbf = projector.normalizer.unnormalize(x_cbf)
        x_cbf = torch.tensor(x_cbf, device=device, dtype=torch.float32)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:

            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)

        if projector is not None and projector.gradient:
            if self.goal_dim > 0:
                grad = projector.compute_gradient(x_recon[:,:,:-self.goal_dim], constraints)
            else:
                grad = projector.compute_gradient(x_recon, constraints)
            model_mean = model_mean + grad

        if env.step_counter > 5:
            if iteration is not None and iteration  < 10:  #this will called from reverse diffusion loop
                posterior_var_t = self.posterior_variance[t]
                posterior_var_t = posterior_var_t.unsqueeze(1).unsqueeze(1)
                normalized_grad_cbf, normalized_grad_clf, adaptive_scalar_cbf = self.compute_grad_cbf(x_cbf, external_obstacle_pos, env)
                scaled_grad_cbf = adaptive_scalar_cbf.reshape_as(posterior_var_t) * posterior_var_t * normalized_grad_cbf
                scaled_grad_clf = -2 * posterior_var_t * normalized_grad_clf
                # self.cbf_plot(x_cbf, external_obstacle_pos, env, scaled_grad_cbf, scaled_grad_clf, config_obstacle_constraints, polytopic_constraints, save_dir)
            elif iteration is None:  #this will called from MPC model
                posterior_var_t = self.posterior_variance[t]
                posterior_var_t = posterior_var_t.unsqueeze(1).unsqueeze(1)
                normalized_grad_cbf, normalized_grad_clf, adaptive_scalar_cbf = self.compute_grad_cbf(x_cbf, external_obstacle_pos, env)
                # scaled_grad_cbf = adaptive_scalar_cbf.reshape_as(posterior_var_t) * posterior_var_t * normalized_grad_cbf
                scaled_grad_cbf = 0
                scaled_grad_clf = -2 * posterior_var_t * normalized_grad_clf
            else:
                scaled_grad_cbf = 0
                scaled_grad_clf = 0

            model_mean[:, :, :2] = model_mean[:, :, :2] + scaled_grad_cbf + scaled_grad_clf
        else:
            pass
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample_diffmpc(self, x, cond, t, returns=None, projector=None, constraints=None, env=None, config_obstacle_constraints=None, polytopic_constraints=None, save_dir=None, iteration=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_diffmpc(x=x, cond=cond, t=t, returns=returns, projector=projector, constraints=constraints, env=env, config_obstacle_constraints=config_obstacle_constraints, polytopic_constraints=polytopic_constraints, save_dir=save_dir, iteration=iteration)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    def p_sample_loop_diffmpc(self, shape, cond, returns=None, return_diffusion=False, projector=None, constraints=None, repeat_last=0 ,env=None ,config_obstacle_constraints=None ,polytopic_constraints=None, save_dir=None, variant=None):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion: diffusion = [x]
        costs = {}

        # Denoising process
        last_timestep = -repeat_last if repeat_last > 0 and projector is not None else 0
        for i in reversed(range(last_timestep + 5, self.n_timesteps)):
            t = i if i >= 0 else 0
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if projector is not None and projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                x = self.p_sample_diffmpc(x, cond, timesteps, returns, projector=projector, constraints=constraints)
            else:
                x = self.p_sample_diffmpc(x, cond, timesteps, returns, projector=projector ,env=env, config_obstacle_constraints=config_obstacle_constraints ,polytopic_constraints=polytopic_constraints, save_dir=save_dir, iteration=i)

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion: diffusion.append(x)

        colors = plt.cm.rainbow(np.linspace(0, 1, batch_size))
        colors_projected = plt.cm.viridis(np.linspace(0, 1, batch_size))
        x_diffusion = x.clone().detach().cpu().numpy()
        x_diffusion = projector.normalizer.unnormalize(x_diffusion)
        x, projection_costs = projector.project_mpcdiff(x, constraints, self.p_sample_diffmpc, apply_conditioning, cond, timesteps, save_dir)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
        infos = {}
        if return_diffusion: infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = projection_costs

        # Debug plot
        fig_debug, ax_debug = plt.subplots(figsize=(10, 10))
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
        x_after_projection = x.clone().detach().cpu().numpy()
        x_after_projection = projector.normalizer.unnormalize(x_after_projection)
        obstacle_env_data = [env.get_obstacle_position()]
        for batch_idx in range(x_after_projection.shape[0]):
        # Plot trajectory points
            ax_debug.scatter(x_diffusion[batch_idx, :, 4], x_diffusion[batch_idx, :, 5], 
                        color=colors[batch_idx], 
                        s=30, 
                        label=f'Trajectory {batch_idx+1}',
                        alpha=0.7)
            ax_debug.scatter([projector.normalizer.unnormalize(np.concatenate([cond[0].clone().detach().numpy()[0,-2:],cond[0].clone().detach().numpy()[0,-2:],cond[0].clone().detach().numpy()[0,-2:]]))[-2]], [projector.normalizer.unnormalize(np.concatenate([cond[0].clone().detach().numpy()[0,-2:],cond[0].clone().detach().numpy()[0,-2:],cond[0].clone().detach().numpy()[0,-2:]]))[-1]], 
                        color='black', 
                        s=70, 
                        label=f'Actual State',
                                        marker='h',
                alpha=0.7)
            if projection_costs is not None and batch_idx == np.argmin(projection_costs):
                ax_debug.scatter(x_after_projection[batch_idx, :, 4], x_after_projection[batch_idx, :, 5],
                            color=colors_projected[batch_idx],
                            s=40,
                            marker='s',
                            label=f'Projected {batch_idx+1}')
            else:
                ax_debug.scatter(x_after_projection[batch_idx, :, 4], x_after_projection[batch_idx, :, 5],
                            color=colors_projected[batch_idx],
                            s=40,
                            marker='*',
                            label=f'Projected {batch_idx+1}')
        centers = []
        for center_pos_key in list(obstacle_env_data[-1].keys())[:6]:  # First 6 are obstacles
            centers.append(obstacle_env_data[-1][center_pos_key][:2])
        for center in centers:
            ax_debug.add_patch(matplotlib.patches.Circle(center, 0.025, color='r'))
        
        # Plot finish line
        ax_debug.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
        
        # Plot additional constraint areas

        for k in range(len(config_obstacle_constraints)):
            pos_i = config_obstacle_constraints[k]['center']
            r_i = config_obstacle_constraints[k]['radius']
            ax_debug.add_patch(matplotlib.patches.Circle(pos_i, r_i, color='b', alpha=0.2))

        # Add legend
        ax_debug.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_limits = [[0.2, 0.8], [-0.3, 0.4]]   # TODO: get from env
        utils.plot_halfspace_constraints('avoiding-d3il', polytopic_constraints, ax_debug, ax_limits)
        # Set axis limits and grid
        ax_debug.set_xlim(ax_limits[0])
        ax_debug.set_ylim(ax_limits[1])
        ax_debug.grid(True, alpha=0.3)

        # Save the figure with timestep in filename
        timestamp = int(time.time() * 1000)  
        debug_save_path = f'{save_dir}/diffusion/diff_{timestamp:04d}.png'
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)

        data_directory = save_dir[: save_dir.find('/logs/')] + "/d3il/environments/dataset/data/avoiding/data/"

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
                plt.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], 'b-', alpha=0.05, label='Desired trajectory' if file == files_to_plot[0] else "")
                
                # Plot actual trajectory
                plt.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], 'r-', alpha=0.05, label='Actual trajectory' if file == files_to_plot[0] else "")
                
                # Plot start points
                plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^', label='Start points' if file == files_to_plot[0] else "")
                
                # Plot end points
                plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x', label='End points' if file == files_to_plot[0] else "")

        plt.savefig(debug_save_path, bbox_inches='tight', dpi=150)
        plt.close(fig_debug)
            
        return x, infos


    @torch.no_grad()
    def p_sample_loop_diffmpc_project(self, shape, cond, returns=None, return_diffusion=False, projector=None, constraints=None, repeat_last=0, 
                      env=None ,config_obstacle_constraints=None ,polytopic_constraints=None, save_dir=None, variant=None):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion: diffusion = [x]
        costs = {}

        # Denoising process
        last_timestep = -repeat_last if repeat_last > 0 and projector is not None else 0
        for i in reversed(range(last_timestep, self.n_timesteps)):
            t = i if i >= 0 else 0
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if projector is not None and projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                x = self.p_sample_diffmpc(x, cond, timesteps, returns, projector=projector, constraints=constraints)
            else:
                x = self.p_sample_diffmpc(x, cond, timesteps, returns, projector=projector ,env=env, config_obstacle_constraints=config_obstacle_constraints ,polytopic_constraints=polytopic_constraints, save_dir=save_dir, iteration=i)

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
            x_diffusion = x.clone().detach().cpu().numpy()
            x_diffusion = projector.normalizer.unnormalize(x_diffusion)
            if projector is not None and not projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                if self.goal_dim > 0:
                    x[:,:,:-self.goal_dim], projection_costs = projector.project(x[:,:,:-self.goal_dim], constraints)
                    costs[i] = projection_costs
                else:
                    x, projection_costs = projector.project(x, constraints)
                    costs[i] = projection_costs

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion: diffusion.append(x)

        infos = {}
        if return_diffusion: infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs
        colors = plt.cm.rainbow(np.linspace(0, 1, batch_size))
        obstacle_env_data = [env.get_obstacle_position()]
        x_after_projection = x.clone().detach().cpu().numpy()
        x_after_projection = projector.normalizer.unnormalize(x_after_projection)
        fig_debug, ax_debug = plt.subplots(figsize=(10, 10))
        for batch_idx in range(x_after_projection.shape[0]):
        # Plot trajectory points
            ax_debug.scatter(x_diffusion[batch_idx, :, 4], x_diffusion[batch_idx, :, 5], 
                color=colors[batch_idx], 
                s=30, 
                label=f'Trajectory {batch_idx+1}',
                alpha=0.7)
            if costs and batch_idx == np.argmin(list(costs.values())):
                ax_debug.scatter(x_after_projection[batch_idx, :, 4], x_after_projection[batch_idx, :, 5],
                            color=colors[batch_idx],
                            s=40,
                            marker='s',
                            label=f'Projected {batch_idx+1}')
            else:
                ax_debug.scatter(x_after_projection[batch_idx, :, 4], x_after_projection[batch_idx, :, 5],
                            color=colors[batch_idx],
                            s=40,
                            marker='*',
                            label=f'Projected {batch_idx+1}')
        centers = []
        for center_pos_key in list(obstacle_env_data[-1].keys())[:6]:  # First 6 are obstacles
            centers.append(obstacle_env_data[-1][center_pos_key][:2])
        for center in centers:
            ax_debug.add_patch(matplotlib.patches.Circle(center, 0.025, color='r'))
        
        # Plot finish line
        ax_debug.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
        
        # Plot additional constraint areas

        for k in range(len(config_obstacle_constraints)):
            pos_i = config_obstacle_constraints[k]['center']
            r_i = config_obstacle_constraints[k]['radius']
            ax_debug.add_patch(matplotlib.patches.Circle(pos_i, r_i, color='b', alpha=0.2))

        # Add legend
        ax_debug.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_limits = [[0.2, 0.8], [-0.3, 0.4]]   # TODO: get from env
        utils.plot_halfspace_constraints('avoiding-d3il', polytopic_constraints, ax_debug, ax_limits)
        # Set axis limits and grid
        ax_debug.set_xlim(ax_limits[0])
        ax_debug.set_ylim(ax_limits[1])
        ax_debug.grid(True, alpha=0.3)

        # Save the figure with timestep in filename
        timestamp = int(time.time() * 1000)  
        debug_save_path = f'{save_dir}/diffusion/diff_{timestamp:04d}.png'
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)


        data_directory = save_dir[: save_dir.find('/logs/')] + "/d3il/environments/dataset/data/avoiding/data/"

        state_files = os.listdir(data_directory)
        files_to_plot = state_files

        for file in files_to_plot:
            with open(os.path.join(data_directory, file), 'rb') as f:
                env_state = pickle.load(f)
                
                robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
                robot_c_pos = env_state['robot']['c_pos'][:, :2]
                
                # Plot desired trajectory
                plt.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], 'b-', alpha=0.05, label='Desired trajectory' if file == files_to_plot[0] else "")
                
                # Plot actual trajectory
                plt.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], 'r-', alpha=0.05, label='Actual trajectory' if file == files_to_plot[0] else "")
                
                # Plot start points
                plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^', label='Start points' if file == files_to_plot[0] else "")
                
                # Plot end points
                plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x', label='End points' if file == files_to_plot[0] else "")

        plt.savefig(debug_save_path, bbox_inches='tight', dpi=150)
        plt.close(fig_debug)
        return x, infos



    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, return_diffusion=False, projector=None, constraints=None, repeat_last=0, 
                      env=None ,config_obstacle_constraints=None ,polytopic_constraints=None, save_dir=None, variant=None):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion: diffusion = [x]
        costs = {}

        # Denoising process
        last_timestep = -repeat_last if repeat_last > 0 and projector is not None else 0
        for i in reversed(range(last_timestep, self.n_timesteps)):
            t = i if i >= 0 else 0
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if projector is not None and projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                x = self.p_sample(x, cond, timesteps, returns, projector=projector, constraints=constraints)
            elif variant == 'cobl':
                x = self.p_sample_diffmpc(x, cond, timesteps, returns, projector=projector, constraints=constraints, env=env, config_obstacle_constraints=config_obstacle_constraints, polytopic_constraints=polytopic_constraints, save_dir=save_dir, iteration=i)
            else:
                x = self.p_sample(x, cond, timesteps, returns, projector=projector ,env=env, config_obstacle_constraints=config_obstacle_constraints ,polytopic_constraints=polytopic_constraints, save_dir=save_dir, iteration=i)

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
            if not variant in ['diffuser','cobl']:
                x_diffusion = x.clone().detach().cpu().numpy()
                x_diffusion = projector.normalizer.unnormalize(x_diffusion)
            if projector is not None and not projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                if variant == 'cobl':
                    pass
                else:
                    if self.goal_dim > 0:
                        x[:,:,:-self.goal_dim], projection_costs = projector.project(x[:,:,:-self.goal_dim], constraints)
                        costs[i] = projection_costs
                    else:
                        x, projection_costs = projector.project(x, constraints)
                        costs[i] = projection_costs

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion: diffusion.append(x)

        infos = {}
        if return_diffusion: infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs
        if not variant in ['diffuser','cobl']:
            colors = plt.cm.rainbow(np.linspace(0, 1, batch_size))
            obstacle_env_data = [env.get_obstacle_position()]
            x_after_projection = x.clone().detach().cpu().numpy()
            x_after_projection = projector.normalizer.unnormalize(x_after_projection)
            fig_debug, ax_debug = plt.subplots(figsize=(10, 10))
            for batch_idx in range(x_after_projection.shape[0]):
            # Plot trajectory points
                ax_debug.scatter(x_diffusion[batch_idx, :, 4], x_diffusion[batch_idx, :, 5], 
                    color=colors[batch_idx], 
                    s=30, 
                    label=f'Trajectory {batch_idx+1}',
                    alpha=0.7)
                if costs and batch_idx == np.argmin(list(costs.values())):
                    ax_debug.scatter(x_after_projection[batch_idx, :, 4], x_after_projection[batch_idx, :, 5],
                                color=colors[batch_idx],
                                s=40,
                                marker='s',
                                label=f'Projected {batch_idx+1}')
                else:
                    ax_debug.scatter(x_after_projection[batch_idx, :, 4], x_after_projection[batch_idx, :, 5],
                                color=colors[batch_idx],
                                s=40,
                                marker='*',
                                label=f'Projected {batch_idx+1}')
            centers = []
            for center_pos_key in list(obstacle_env_data[-1].keys())[:6]:  # First 6 are obstacles
                centers.append(obstacle_env_data[-1][center_pos_key][:2])
            for center in centers:
                ax_debug.add_patch(matplotlib.patches.Circle(center, 0.025, color='r'))
            
            # Plot finish line
            ax_debug.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
            
            # Plot additional constraint areas

            for k in range(len(config_obstacle_constraints)):
                pos_i = config_obstacle_constraints[k]['center']
                r_i = config_obstacle_constraints[k]['radius']
                ax_debug.add_patch(matplotlib.patches.Circle(pos_i, r_i, color='b', alpha=0.2))

            # Add legend
            ax_debug.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_limits = [[0.2, 0.8], [-0.3, 0.4]]   # TODO: get from env
            utils.plot_halfspace_constraints('avoiding-d3il', polytopic_constraints, ax_debug, ax_limits)
            # Set axis limits and grid
            ax_debug.set_xlim(ax_limits[0])
            ax_debug.set_ylim(ax_limits[1])
            ax_debug.grid(True, alpha=0.3)

            # Save the figure with timestep in filename
            timestamp = int(time.time() * 1000)  
            debug_save_path = f'{save_dir}/diffusion/diff_{timestamp:04d}.png'
            os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)


            data_directory = save_dir[: save_dir.find('/logs/')] + "/d3il/environments/dataset/data/avoiding/data/"

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
                    plt.plot(robot_des_pos[:, 0], robot_des_pos[:, 1], 'b-', alpha=0.05, label='Desired trajectory' if file == files_to_plot[0] else "")
                    
                    # Plot actual trajectory
                    plt.plot(robot_c_pos[:, 0], robot_c_pos[:, 1], 'r-', alpha=0.05, label='Actual trajectory' if file == files_to_plot[0] else "")
                    
                    # Plot start points
                    plt.scatter(robot_des_pos[0, 0], robot_des_pos[0, 1], c='green', s=50, marker='^', label='Start points' if file == files_to_plot[0] else "")
                    
                    # Plot end points
                    plt.scatter(robot_des_pos[-1, 0], robot_des_pos[-1, 1], c='black', s=50, marker='x', label='End points' if file == files_to_plot[0] else "")

            plt.savefig(debug_save_path, bbox_inches='tight', dpi=150)
            plt.close(fig_debug)
        return x, infos

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        if hasattr(kwargs['projector'], 'method') and kwargs['projector'].method == "diffmpc":
            return self.p_sample_loop_diffmpc(shape, cond, returns, *args, **kwargs)
        elif hasattr(kwargs['projector'], 'method') and kwargs['projector'].method == "diffmpc_project":
            return self.p_sample_loop_diffmpc_project(shape, cond, returns, *args, **kwargs)
        else:
            return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion: diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        if self.predict_epsilon:
            # Cause we condition on obs at t=0
            # noise[:, 0, self.action_dim:] = 0
            noise = apply_conditioning(noise, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim, goal_dim=self.goal_dim)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim, goal_dim=self.goal_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
