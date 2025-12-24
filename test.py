import torch

import argparse
import time, sys, os, yaml, logging

from utils import evaluate
from models import load_model
from datasets import load_dataset

def add_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    
    return parser

def get_logger(expr_name):
    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_h = logging.FileHandler(f"logs/{expr_name}.log", mode='w')
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(formatter)
    logger.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    logger.addHandler(console_h)

    return logger

def main(cfg):
    logger = get_logger(cfg['expr'])
    logger.info(f"=====================[{cfg['expr']}]=====================")
    
    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else: 
        device = 'cpu'
    logger.info(f"device: {device}")
    
    # Hyperparameter Settings
    hp_cfg = cfg['hyperparameters']
    
    # Load Dataset
    data_cfg = cfg['data']
    test_ds = load_dataset(data_cfg)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=hp_cfg['batch_size'])
    logger.info(f"Load Dataset {data_cfg['dataset']}")
    
    # Load Model
    save_cfg = cfg['save']
    model_cfg = cfg['model']
    model = load_model(model_cfg).to(device)
    ckpt = torch.load(os.path.join(save_cfg['weights_path'], save_cfg['weights_filename']),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    start_time = int(time.time())
    result = evaluate(model, test_dl, device)
    test_time = int(time.time() - start_time)
    logger.info(f"Test Time: {test_time//60:02d}m {test_time%60:02d}s")
    
    for key, value in result.items():
        logger.info(f"{key}: {value*100:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/test/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)