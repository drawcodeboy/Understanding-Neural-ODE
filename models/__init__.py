from .NeuralODE.odenet import ODENet
from .ResNet.resnet import ResNet
from .OfficialNeuralODE.model import OfficialODENet

def load_model(cfg):
    if cfg['name'] == 'ODENet':
        return ODENet.from_config(cfg)
    elif cfg['name'] == 'ResNet':
        return ResNet.from_config(cfg)
    elif cfg['name'] == 'OfficialODENet':
        return OfficialODENet.from_config(cfg)