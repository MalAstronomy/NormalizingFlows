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
            self.net.append(nn.ReLU())
        
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
            self.net.append(nn.ReLU())
        
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
        
        self.u = nn.Parameter(torch.empty(2,1))
        self.w = nn.Parameter(torch.empty(2,1))
        self.b = nn.Parameter(torch.empty(1))
        
        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

        self.layer1 = Matmul()
        self.layer2 = Matadd() 
        self.layer3 = nn.Tanh()
        self.layer4 = Matmul()
   
        
    def forward(self, z):
        
        
        x = self.layer1(z, self.w)
        x = self.layer2(x, self.b)
        x = self.layer3(x)
        x = self.layer4(x, self.u.T)
        x = z + x
        
        return x
    
  
    
    
    