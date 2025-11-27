import mlx.core as mx
import mlx.nn as nn
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.reparameterize import Reparameterize
from utils.preprocess import graph_to_molecule
from utils.one_hot import one_hot

class VAE(nn.Module):
    def __init__(self, encoder, decoder, reparameterize):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reparameterize = reparameterize
        # Property prediction: use mu (latent representation)
        # mu shape: (batch, latent_dim)
        # Get latent_dim from encoder's fc_mu layer weight shape (output_dims, input_dims)
        latent_dim = encoder.fc_mu.weight.shape[0]
        self.property = nn.Linear(latent_dim, 1)

    def __call__(self, inputs):
        mu, logvar = self.encoder(inputs)
        z = self.reparameterize(mu, logvar)
        adjacency, features = self.decoder(z)
        property = self.property(mu)
        return mu, logvar, adjacency, features, property

    def inference(self, batch_size):
        # Get latent_dim and shapes from encoder
        latent_dim = self.encoder.fc_mu.weight.shape[0]
        adjacency_shape = self.encoder.adjacency_shape if hasattr(self.encoder, 'adjacency_shape') else (5, 90, 90)
        feature_shape = self.encoder.feature_shape if hasattr(self.encoder, 'feature_shape') else (90, 11)
        
        # Sample z from prior
        z = mx.random.normal((batch_size, latent_dim), loc=0.0, scale=1.0)
        
        # Decode
        adjacency, features = self.decoder(z)
        
        # Convert to discrete: argmax then one-hot
        # Adjacency: (batch, bond_dim, num_atoms, num_atoms) -> argmax over bond_dim -> (batch, num_atoms, num_atoms)
        adjacency_argmax = mx.argmax(adjacency, axis=1)
        # One-hot: (batch, num_atoms, num_atoms) -> (batch, bond_dim, num_atoms, num_atoms)
        adjacency_onehot = one_hot(adjacency_argmax, depth=adjacency_shape[0], axis=1)
        
        # Remove self-loops
        # adjacency_onehot: (batch, bond_dim, num_atoms, num_atoms)
        # eye should be (num_atoms, num_atoms) = (adjacency_shape[1], adjacency_shape[2])
        num_atoms = adjacency_shape[1]
        eye = mx.eye(num_atoms)
        # Broadcast eye to (batch, bond_dim, num_atoms, num_atoms)
        eye_broadcast = mx.broadcast_to(eye[None, None, :, :], adjacency_onehot.shape)
        adjacency_onehot = adjacency_onehot * (1.0 - eye_broadcast)
        
        # Features: (batch, num_atoms, atom_dim) -> argmax over atom_dim -> (batch, num_atoms)
        features_argmax = mx.argmax(features, axis=2)
        # One-hot: (batch, num_atoms) -> (batch, num_atoms, atom_dim)
        features_onehot = one_hot(features_argmax, depth=feature_shape[1], axis=2)
        
        # Convert to molecules
        return [
            graph_to_molecule((adjacency_onehot[i], features_onehot[i])) for i in range(batch_size)
        ]

