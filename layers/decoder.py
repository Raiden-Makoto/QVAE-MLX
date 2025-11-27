import mlx.core as mx
import mlx.nn as nn
from layers.reshape import Reshape

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        dense_units,
        dropout_rate,
        adjacency_shape,
        feature_shape,
    ):
        super().__init__()
        # Build dense layers with proper input/output dimensions
        self.dense_layers = []
        input_dim = latent_dim
        for units in dense_units:
            layer = nn.Linear(input_dim, units)
            self.dense_layers.append(layer)
            input_dim = units
        
        # Projection layers for adjacency and features
        adj_flat_size = adjacency_shape[0] * adjacency_shape[1] * adjacency_shape[2]
        feat_flat_size = feature_shape[0] * feature_shape[1]
        
        self.adj_proj = nn.Linear(input_dim, adj_flat_size)
        self.feature_proj = nn.Linear(input_dim, feat_flat_size)
        
        self.dropout_rate = dropout_rate
        self.adj_shape = adjacency_shape
        self.feature_shape = feature_shape

    def __call__(self, inputs):
        x = inputs  # (batch, latent_dim)
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.Dropout(self.dropout_rate)(x)
        
        # Project to adjacency and features
        x_adjacency_flat = self.adj_proj(x)  # (batch, adj_flat_size)
        x_adjacency = Reshape((x.shape[0],) + self.adj_shape)(x_adjacency_flat)  # (batch, bond_dim, num_atoms, num_atoms)
        
        x_feature_flat = self.feature_proj(x)  # (batch, feat_flat_size)
        x_feature = Reshape((x.shape[0],) + self.feature_shape)(x_feature_flat)  # (batch, num_atoms, atom_dim)
        
        # Make adjacency symmetric and apply softmax
        x_adjacency = (x_adjacency + mx.transpose(x_adjacency, (0, 1, 3, 2))) / 2
        x_adjacency = mx.softmax(x_adjacency, axis=1)
        
        # Apply softmax to features
        x_feature = mx.softmax(x_feature, axis=2)
        
        return x_adjacency, x_feature