import torch
from torch.utils.data import dataset
from torchvision.datasets import MNIST
import numpy as np

class MNIST_Dataset():
    def __init__(self,
                 root="data/", # data 하위에 MNIST 디렉터리 생김
                 download=True, # root로 지정한 위치에 없으면 다운, 있으면 패스
                 mode='train'): 
        
        if mode not in ['train', 'test']:
            raise Exception("mode should be 'train' or 'test'")
        
        # Train = 60,000 samples, Test = 10,000 samples
        self.data_li = MNIST(root=root,
                             download=download,
                             train=True if mode=='train' else False)
        
        self.data_li = list(self.data_li)
        
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        image, label = self.data_li[idx]

        image = np.array(image.getdata()).reshape(28, 28).astype(np.float32)
        image /= 255.
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.int64) # long

        return image, label
    
    @classmethod
    def from_config(cls, cfg):
        return cls(root=cfg.get('root', 'data/'),
                   download=cfg.get('download', True),
                   mode=cfg.get('mode', 'train'))