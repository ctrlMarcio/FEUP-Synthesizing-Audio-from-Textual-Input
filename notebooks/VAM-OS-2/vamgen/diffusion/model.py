import torch
from torch import nn
from torch.distributions import Normal
from math import sqrt

from vamgen.diffusion.unet import UNet


class DiffusionModel(nn.Module):

    def __init__(self, diffusion_steps, scheduler="linear", depth=3, conv_layers_per_block=2):
        super().__init__()

        self.diffusion_steps = diffusion_steps
        self.scheduler = scheduler

        # make the betas
        self.betas = self._make_betas()

        # define the unet
        self.unet = UNet(3, 1, depth=depth, conv_layers_per_block=conv_layers_per_block)

    def forward(self, x):
        # x is the initial pure noise data
        # apply the unet
        pass

    def _make_betas(self):
        if self.scheduler == "linear":
            betas = torch.linspace(0, 1, self.diffusion_steps)
        else:
            raise NotImplementedError

        return betas

    def forward_diffusion(self, x0, timestep):
        # x0 is the initial clean data

        x = x0
        for t in range(timestep):

            # Get current noise level
            beta_t = self.betas[t]

            # Sample noise
            noise = torch.randn_like(x)

            # Add noise
            x_t = sqrt(1 - beta_t) * x + sqrt(beta_t) * noise

            # Update to use x_t for next step
            x = x_t

        return x

    def get_noise(self, x, timestep):
        # calculates a noise tensor with the same shape as x
        # according to the beta value at the given timestep

        # current beta
        beta_t = self.betas[timestep]

        # sample noise
        noise = torch.randn_like(x)

        # return noise
        return sqrt(beta_t) * noise
