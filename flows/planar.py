import torch
import torch.nn as nn

class Planar(nn.Module):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """
    
    def __init__(self, net):
        super().__init__()
        
        self.net = net
        
    def g(self,z):
        # g = f^-1
        x = self.net(z)
            
        for name, param in self.net.named_parameters():
            if name == 'u' : 
                self.u = param
            elif name == 'w' : 
                self.w = param
            elif name == 'b' : 
                self.b = param
        
        affine = torch.mm(z, self.w) + self.b
        psi = (1 - nn.Tanh()(affine) ** 2) * self.w.T
        abs_det = (1 + torch.mm(psi, self.u)).abs()
        log_det = torch.log(1e-4 + abs_det).squeeze(1)
        
        return x, log_det
    
    def f(self, x):
        return print('Not implemented')

    def inverse(self, z):
        return self.g(z)[0]

    def forward(self, x):
        return self.g(x)
    
    
    