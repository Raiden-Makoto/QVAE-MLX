import mlx.core as mx
import mlx.nn as nn

class Reparameterize(nn.Module):
    """
    Reparameterization trick for VAE.
    Samples z ~ N(mu, sigma^2) where sigma = exp(0.5 * logvar)
    using z = mu + sigma * eps, where eps ~ N(0, 1)
    """
    
    def __init__(self):
        super().__init__()

    def __call__(self, mu, logvar):
        """
        Args:
            mu: Mean tensor of shape (batch, latent_dim)
            logvar: Log variance tensor of shape (batch, latent_dim)
        
        Returns:
            z: Sampled latent vector of shape (batch, latent_dim)
        """
        batch, latent_dim = mu.shape
        # Generate epsilon ~ N(0, 1)
        eps = mx.random.normal((batch, latent_dim), loc=0.0, scale=1.0)
        # Compute sigma = exp(0.5 * logvar)
        sigma = mx.exp(logvar * 0.5)
        # Reparameterization: z = mu + sigma * eps
        return mu + sigma * eps