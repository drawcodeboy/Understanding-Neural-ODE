import os, sys
sys.path.append(os.getcwd())

from models import load_model

def get_params(model_name):
    model = load_model({"name": model_name})
    p_sum = 0
    for p in model.parameters():
        p_sum += p.numel()
    
    return p_sum

def main():
    print(f"ODE-Net # Params: {get_params("ODENet")/1000000:.2f}M")

    print(f"ResNet # Params: {get_params("ResNet")/1000000:.2f}M")

if __name__ == '__main__':
    main()