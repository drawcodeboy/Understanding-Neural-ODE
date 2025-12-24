from .NeuralODE.odenet import ODENet
from .ResNet.resnet import ResNet

def load_model(cfg):
    if cfg['name'] == 'ODENet':
        return ODENet.from_config(cfg)
    elif cfg['name'] == 'ResNet':
        return ResNet.from_config(cfg)