import os, sys
sys.path.append(os.getcwd())

from models import load_model

def get_params(cfg):
    model = load_model(cfg)
    p_sum = 0
    for p in model.parameters():
        p_sum += p.numel()
    
    return p_sum

def main():
    cfg = {'name': 'ResNet', 'layers': 6}
    print(f"ResNet # Params: {get_params(cfg)/1000000:.2f}M")

    cfg = {'name': 'ResNet', 'layers': 1}
    print(f"ResNet (One Block) # Params: {get_params(cfg)/1000000:.2f}M")

    cfg = {'name': 'ODENet'}
    print(f"ODE-Net (Euler) # Params: {get_params(cfg)/1000000:.2f}M")

    cfg = {'name': 'OfficialODENet', 'solver': 'implicit_adams', 'adjoint': True}
    print(f"ODE-Net (Adams) # Params: {get_params(cfg)/1000000:.2f}M")

    cfg = {'name': 'OfficialODENet', 'solver': 'dopri5', 'adjoint': True}
    print(f"ODE-Net (RK) # Params: {get_params(cfg)/1000000:.2f}M")

    cfg = {'name': 'OfficialODENet', 'solver': 'dopri', 'adjoint': False}
    print(f"RK-Net # Params: {get_params(cfg)/1000000:.2f}M")

if __name__ == '__main__':
    main()