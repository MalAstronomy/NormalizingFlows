## Building flows

from flows import realnvp, planar
from flows.realnvp import CouplingBijection, ReverseBijection
from flows.planar import Planar
from flows.cnf import CNF
from nets import MLP
from flows.NormalizingFlow import Flow

class build_flow():
    def __init__(self, name='realnvp', net = MLP, dim=5, device='cuda'):
        self.name = name
        self.net = net
        self.dim = dim
        self.device = device
        self.bijections = []   
        
        if self.name == 'realnvp':
            self.build_realNVP()
        elif self.name == 'planar':
            self.build_Planar()   
        elif self.name == 'continuous':
            self.build_continuous()
        
        self.flow = Flow(self.bijections, self.name,  self.device)    
        

    def build_realNVP(self):
        # flows for nvp
           
        for i in range(self.dim):
            if i%2 == 0:
                self.bijections.append(CouplingBijection(self.net))
            else:
                self.bijections.append(ReverseBijection())

        
    def build_Planar(self):
        # flows for planar
        for i in range(self.dim):
            self.bijections += [Planar(self.net)]
            
    def build_continuous(self):
        self.bijections = [CNF(self.net)]
        
    def print_flow(self):
        return print(self.flow)
    