import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    plt.figure(figsize=(8, 4))

    ode_euler_loss = np.load(f"saved/loss/train_loss_odenet_euler_mnist.npy")
    ode_adams_loss = np.load(f"saved/loss/train_loss_odenet_adams_mnist.npy")
    ode_adams_adam_loss = np.load(f"saved/loss/train_loss_odenet_adams_adam_mnist.npy")
    ode_rk_loss = np.load(f"saved/loss/train_loss_odenet_rk_mnist.npy")
    rk_loss = np.load(f"saved/loss/train_loss_rknet_mnist.npy")
    res_loss = np.load(f"saved/loss/train_loss_resnet_mnist.npy")
    one_res_loss = np.load(f"saved/loss/train_loss_one_resnet_mnist.npy")

    ode_euler_epochs = np.array([i + 1 for i in range(0, len(ode_euler_loss))])
    ode_adams_epochs = np.array([i + 1 for i in range(0, len(ode_adams_loss))])
    ode_adams_adam_epochs = np.array([i + 1 for i in range(0, len(ode_adams_adam_loss))])
    ode_rk_epochs = np.array([i + 1 for i in range(0, len(ode_rk_loss))])
    rk_epochs = np.array([i + 1 for i in range(0, len(rk_loss))])
    res_epochs = np.array([i + 1 for i in range(0, len(res_loss))])
    one_res_epochs = np.array([i + 1 for i in range(0, len(one_res_loss))])

    line_cfg = {
        'alpha': 0.7,
        'linestyle': '--'
    }
    
    plt.plot(ode_euler_epochs, ode_euler_loss, label='ODE-Net (Euler)', color='blue', **line_cfg)
    plt.plot(ode_adams_epochs, ode_adams_loss, label='ODE-Net (Adams)', color='green', **line_cfg)
    plt.plot(ode_adams_adam_epochs, ode_adams_adam_loss, label='ODE-Net (Adams with Adam)', color='cyan', **line_cfg)
    plt.plot(ode_rk_epochs, ode_rk_loss, label='ODE-Net (RK)', color='purple', **line_cfg)
    plt.plot(rk_epochs, rk_loss, label='RK-Net', color='orange', **line_cfg)
    plt.plot(res_epochs, res_loss, label='ResNet', color='red', **line_cfg)
    plt.plot(one_res_epochs, one_res_loss, label='ResNet (One Block)', color='magenta', **line_cfg)
    plt.title(f"Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(-0.01, 0.15)

    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"assets/loss_curve.jpg", dpi=500)

if __name__ == '__main__':
    main()