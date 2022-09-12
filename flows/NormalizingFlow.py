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
        
        if ((self.name == 'realnvp') | (self.name == 'planar')):
            log_det = torch.zeros(x.shape[0], device=self.device)
        elif self.name == 'continuous':
            log_det = torch.zeros(torch.Size([x[0].shape[0],x[1].shape[0],1]), device=self.device)
            
        z = x
        
        for bijection in self.flow:
            z, ldj = bijection(z)
            if ((self.name == 'realnvp') | (self.name == 'planar')):
                log_det += ldj
            elif self.name == 'continuous':
                log_det += ldj
            
        return z, log_det

    def sample(self, num_samples):
        if ((self.name == 'realnvp') | (self.name == 'planar')):
                z = self.base_dist.sample((num_samples,))
        elif self.name == 'continuous':
            ts = torch.tensor([0, 1]).type(torch.float32).to(self.device)
            z0 = self.base_dist.sample((num_samples,))
            logp_diff_t0 = torch.zeros(z0.size()[0], 1).type(torch.float32).to(self.device)
            z = (ts, z0, logp_diff_t0)
        for bijection in reversed(self.flow):
            z = bijection.inverse(z)
        return z
    
        
        
    
    