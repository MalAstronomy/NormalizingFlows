import torch
import torch.nn as nn
from torch.distributions import Normal


class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self, flow, name = 'realnvp', device = 'cuda'):
        super().__init__()
        self.flow = nn.ModuleList(flow)
        self.device = device
        self.name = name
        
    @property    
    def base_dist(self):
        return Normal(
            loc=torch.zeros(2, device=self.device),
            scale=torch.ones(2, device=self.device),
        )
    
    def flow_outputs(self, x):
        log_det = torch.zeros(x.shape[0], device=self.device)
        for bijection in self.flow:
            z, ldj = bijection(x)
            log_det += ldj
            
        return z, log_det

    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        for bijection in reversed(self.flow):
            z = bijection.inverse(z)
        return z
    
        
        
    
    