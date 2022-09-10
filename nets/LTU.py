import torch
from torch import nn
from typing import List

class encoder(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        
        
        hidden_sizes = [2] + hidden_sizes
        
        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)

        
    def forward(self, x):
        return self.net(x)
    
class decoder(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        
        hidden_sizes = [2] + hidden_sizes
        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(nn.GELU())
        
        self.net = nn.Sequential(*self.net)

    def forward(self, z):
        return self.net(z)


class Matmul(nn.Module):
    def forward(self, *args):
        return torch.matmul(*args)

class Matadd(nn.Module):
    def forward(self, *args):
        return torch.add(*args)
    
class LTU(nn.Module):
    """
    A Linear Transition Unit with a tanh nonlinearity
    """
    def __init__(self, ):
        super().__init__()
        
#         self.u = nn.Parameter(torch.empty(2,1))
#         self.w = nn.Parameter(torch.empty(2,1))
#         self.b = nn.Parameter(torch.empty(1))
        
#         nn.init.normal_(self.w)
#         nn.init.normal_(self.u)
#         nn.init.normal_(self.b)
        
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, 0.1))
        
        if (torch.mm(self.u, self.w.T)< -1).any():   
            self.get_u_hat()

        self.layer1 = Matmul()
        self.layer2 = Matadd() 
        self.layer3 = nn.Tanh()
        self.layer4 = Matmul()
        
    def get_u_hat(self):
        """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition 
        for invertibility of the transformation f(z). See Appendix A.1.
        """
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        self.u.data = (self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2)
   
    def forward(self, z):
        
        x = self.layer1(z, self.w.T) #self.w     # 2*1
        x = self.layer2(x, self.b)               # 2*1
        x = self.layer3(x)                       # 2*1
        x = self.layer4(x, self.u) #u.T          # 2*2
        x = z + x                                # 2*2
        
        return x
    
  
    
    
    