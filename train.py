from models import load_model

import torch

cfg = {
    'name': 'ODENet'
}

model = load_model(cfg)

x =  torch.randn((16, 1, 28, 28))
print(x.shape)
print(model(x).shape)

cfg = {
    'name': 'ResNet'
}

model = load_model(cfg)

x =  torch.randn((16, 1, 28, 28))
print(x.shape)
print(model(x).shape)