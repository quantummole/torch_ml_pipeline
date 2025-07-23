import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class DiffusionProcess:
    def __init__(self, noise_schedule):
        assert noise_schedule[0] == 0, "Noise schedule must start with 0"
        assert noise_schedule[-1] > 0, "Noise schedule must end with a positive value"
        assert len(noise_schedule) > 1, "Noise schedule must have more than one value"
        self.betas = noise_schedule
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
    def forward_process(self, x0, t):
        noise = torch.randn(*x0.shape).numpy()
        alpha_t = self.alpha_cumprod[t]
        std = np.sqrt(1 - alpha_t)
        mean = np.sqrt(alpha_t) * x0
        return mean + std * noise, noise, 
    def backward_process(self, xt, noise, t):
        alpha_t = self.alpha_cumprod[t]
        std = np.sqrt(1 - alpha_t)
        x0 = (xt - std*noise)/np.sqrt(alpha_t)
        return x0
    

if __name__ == "__main__":
    import cv2
    from matplotlib import pyplot as plt
    noise_schedule = np.linspace(0, 0.02, 100)
    diffusion = DiffusionProcess(noise_schedule)
    
    # Example usage
    x0 =  cv2.imread(r"""C:\Users\rkvai\OneDrive\Pictures\Screenshots\Screenshot 2025-07-13 135001.png""")/255. # Example input image
    t = 50  # Example time step
    xt, noise = diffusion.forward_process(x0, t)
    x0_reconstructed = diffusion.backward_process(xt, noise, t)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(x0)    
    ax[1].imshow(x0_reconstructed)
    ax[0].set_title("Original Image")
    ax[1].set_title("Reconstructed Image")
    plt.show()
    assert np.allclose(x0, x0_reconstructed), "Reconstruction failed"
    print("Original shape:", x0.shape)
    print("Noisy shape:", xt.shape)
    print("Reconstructed shape:", x0_reconstructed.shape)