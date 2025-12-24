import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    plt.figure(figsize=(8, 4))

    ode_loss = np.load(f"saved/loss/train_loss_odenet_mnist.npy")
    res_loss = np.load(f"saved/loss/train_loss_resnet_mnist.npy")

    ode_epochs = np.array([i + 1 for i in range(0, len(ode_loss))])
    res_epochs = np.array([i + 1 for i in range(0, len(res_loss))])

    plt.plot(ode_epochs, ode_loss, label='ODE-Net', color='blue')
    plt.plot(res_epochs, res_loss, label='ResNet', color='red')

    plt.title(f"Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(-0.01, 0.1)

    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"assets/loss_curve.jpg", dpi=500)

if __name__ == '__main__':
    main()