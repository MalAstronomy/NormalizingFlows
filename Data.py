import pandas as pd
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

datapath = Path('datasets')

class Data(Dataset):
    
    def __init__(self, d= 'two_moons.csv', s=None):
        self.d = d
        self.read_csv()
        
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
        
    def read_csv(self):
        self.data = torch.Tensor(pd.read_csv(datapath / self.d).values)
        return self.data
    
    def plot_samples(self, s):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(26, 12))
        ax[0].axis('off'); ax[1].axis('off')
        ax[0].set_title('Data', fontsize=24); ax[1].set_title('Samples', fontsize=24)
        ax[0].hist2d(self.data[...,0].numpy(), self.data[...,1].numpy(), bins=256, range=[[-4, 4], [-4, 4]])
        if s is not None:
            s = s.detach().cpu().numpy()
            ax[1].hist2d(s[...,0], s[...,1], bins=256, range=[[-4, 4], [-4, 4]])
        return plt.show()
    
    def plot_scatter(self):
        return plt.scatter(self.data[:,0], self.data[:,1])
    
    def return_dataloaders(self, split=[0.9, 0.1], batch_size = 4, num_workers=2):
        indices = list(range(self.__len__()))
        s = int(np.floor(split[1] * self.__len__()))

        #shuffling
        np.random.seed(111)
        np.random.shuffle(indices)
        train_indices, test_indices = indices[s:], indices[:s]
        train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)
        train_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler) 
        test_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

        return train_dataloader, test_dataloader
    
 
   
            

    
    