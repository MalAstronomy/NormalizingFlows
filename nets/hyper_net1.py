import torch
import torch.nn as nn


class Matmul(nn.Module):
    def forward(self, *args):
        return torch.matmul(*args)
    
class Matadd(nn.Module):
    def forward(self, *args):
        return torch.add(*args)

class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.
    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim
        
        self.layer = nn.Sequential(nn.Linear(1,hidden_dim), nn.Tanh(),
                       nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                       nn.Linear(hidden_dim,2))
        
        self.tanh = nn.Tanh()
        self.matmul = Matmul()
        self.matadd = Matadd()
        

        self.width = width
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.blocksize  = blocksize
        

    def get_weights(self, params):
        ''' Computes hypernetwork weights. See the forward() function for hypernet output
            Inputs
                t - current time
            Outputs
                W - [width,d,1]
                B - [width,1,d]
                U - [width,1,d]
        '''

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)
        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        U = U * torch.sigmoid(G)
        
        return [W, B, U]
         
    
    def defining_Z(self, z):
        # copy the state for each hidden unit
        Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1) # [width,N,d]
        return Z
    
    
    def forward(self, t, z):
        ''' takes current time and state as input and computes hypernet output: U * h(W.T*Z+B) '''
        
        params = self.layer(t.reshape(1, 1))
        
        W, B, U = self.get_weights(params) # [width,d,1], [width,1,1], [width,1,d]
        Z = self.defining_Z(z)
        
        r = self.matmul(Z,W)
        r = self.matadd(r,B)
        r = self.tanh(r)
        r = self.matmul(r,U)
        
        return r.mean(0)
        
    
    
    
    
    
    