import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        self.alphas = alphas = 1. - betas
        self.sigmas = sigmas = (1 - alphas ** 2).sqrt()
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

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, guide=False, guide_step=1, plan_i=-1, pg=False, task_name='maze2d-medium'):
        
        if pg:
            # print(xt.shape)
            # robot_info = xt[..., :9].view(64, 20, xt.shape[1], 9) # kitchen
            # print(xt.shape)
            robot_info = x[:, :, self.action_dim:self.action_dim+2] # [B, H, D]
            diff = robot_info.unsqueeze(1) - robot_info.unsqueeze(0)
                    # sim = (robot_info - robot_info_mean).unsqueeze(1) * (robot_info - robot_info_mean).unsqueeze(0)
                    # diff = (robot_info.max(dim=0, keepdim=True).values - robot_info_mean).unsqueeze(1) * (robot_info.max(dim=0, keepdim=True).values - robot_info_mean).unsqueeze(0) - sim
                    # remove the diag, make distance with shape [N * N-1 * H * D]
                    # print(f"diff: {diff.shape}")
            diff = diff[~torch.eye(diff.shape[0], device=x.device).bool()].reshape(diff.shape[0], -1, diff.shape[-2], diff.shape[-1])
                        
            distance = torch.norm(diff, p=2, dim=(-2, -1), keepdim=True).to(x.device)
                    # square_distance = torch.sum(diff, dim=(-2, -1), keepdim=True)
            num_traj = diff.shape[0]
                    # print(diff.shape, distance.shape)
            h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / np.log(num_traj - 1)
                    # h_t = (robot_info_mean) ** 2 / np.log(num_traj - 1)
            weights = torch.exp(- (distance ** 2 / h_t)).to(x.device)
            
            # print(x.shape, weights.shape, diff.shape, self.sigmas[t.to(x.device)].shape)
            # print(self.sigmas[t], diff.device, h_t.device)
            grad_phi = 2 * weights * diff / h_t * (self.sigmas[t.to(x.device)][0] ** 2) * 1.0
            grad_phi = grad_phi.sum(dim=1)
                    # print(f"grad_phi: {grad_phi.shape}")
            grad_phi = grad_phi.view_as(x[:, :, :2])

        if guide:
            # gold_easy_location = np.load("gold_medium_single_location.npy")
            if task_name == 'maze2d-medium':
                gold_location = np.load("gold_medium_single_location.npy")
            elif task_name == 'maze2d-large':
                gold_location = np.load("gold_single_location.npy")
            elif task_name == 'multi2d-medium':
                gold_location = np.load("gold_medium_location.npy")
            elif task_name == 'multi2d-large':
                gold_location = np.load("gold_location.npy")
            with torch.enable_grad():
                x.requires_grad_()
                # guide distance between planned states and subgoal state s_c
                # gold_state_location = torch.tensor([-0.23648487,  0.75055706]).to(x.device) # [3, 7]
                # gold_state_location = torch.tensor([0.35002944, 1.15965169]).to(x.device) # [5, 11]
                # gold_state_location = torch.tensor([-0.82299918,  0.95510438]).to(x.device) # upper right
                # gold_state_location = torch.tensor([-0.23648487, -0.88582143]).to(x.device) # lower middle
                # gold_state_location = torch.tensor([-0.82299918,  0.13691513]).to(x.device) # [1, 6] 
                gold_state_location = torch.from_numpy(gold_location[plan_i]).to(x.device) # [3, 8] 
                # subgold_state_location = torch.tensor([0.93654375, -0.2721795]).to(x.device) # lower middle
                # internally divided center from gold to subgold
                # one_two_center = (2 * subgold_state_location + gold_state_location) / 3
                # dist = - torch.min(torch.min(torch.abs(x[:, :, self.action_dim:self.action_dim+2] - gold_state_location), dim=-1).values, dim=-1).values
                ############################################
                # norm_reward_one_goal
                dist = -torch.abs(x[:, :, self.action_dim:self.action_dim+2] - gold_state_location).mean(dim=-1).mean(dim=-1)
                scalar = 125.0 * 2 / 2 / guide_step
                ############################################
                
                ############################################
                # min_reward_one_goal
                # dist = -torch.min(torch.abs(x[:, :, self.action_dim:self.action_dim+2] - gold_state_location).mean(dim=-1), dim=-1).values
                # scalar = 125.0
                ############################################

                # reward = torch.min(dist, 2 * torch.min(torch.min(torch.abs(x[:, :, self.action_dim:self.action_dim+2] - subgold_state_location), dim=-1).values, dim=-1).values
                # reward = reward + torch.abs(x[:, :, self.action_dim:self.action_dim+2] - one_two_center).mean(dim=-1).mean(dim=-1)
                grad = torch.autograd.grad([dist.sum()], [x])[0]
                # print(f"grad: {grad}")
                x.detach()
        else:
            scalar = 0
            grad = 0
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        var = (model_log_variance).exp()
        # print(f"sigma: {sigma}")
        # print(f"var: {var}")
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if pg:
            # print("HERE")
            model_mean[:, :, self.action_dim:self.action_dim+2] += nonzero_mask * grad_phi / self.alphas[t.to(x.device)][0] * 0.5

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise + var * grad * scalar

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, guide=False, guide_step=1, plan_i=-1, verbose=True, return_diffusion=False, pg=False, task_name='maze2d-medium'):
        device = self.betas.device
        self.sigmas = self.sigmas.to(device)
        self.alphas = self.alphas.to(device)
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            for j in range(guide_step):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x = self.p_sample(x, cond, timesteps, guide=guide, guide_step=guide_step, plan_i=plan_i, pg=pg, task_name=task_name)
                x = apply_conditioning(x, cond, self.action_dim)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

