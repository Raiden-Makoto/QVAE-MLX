import mlx.core as mx
import mlx.nn as nn
from layers.dense import QuantumDense
from layers.rgcn import RelationalGraphConvLayer

class Encoder(nn.Module):
    def __init__(
        self,
        gconv_units,
        adjacency_shape,
        feature_shape,
        dense_units,
        dropout_rate,
        latent_dim: int=435,
    ):
        super().__init__()
        bond_dim = adjacency_shape[0]
        atom_dim = feature_shape[1]
        
        # Build gconv layers with proper dimensions
        self.gconv_layers = []
        input_atom_dim = atom_dim
        for units in gconv_units:
            layer = RelationalGraphConvLayer(
                bond_dim=bond_dim,
                atom_dim=input_atom_dim,
                units=units
            )
            self.gconv_layers.append(layer)
            input_atom_dim = units  # Next layer's input is this layer's output
        
        # Pooling: we will mean-pool over atoms to get a single feature vector
        # per graph with size equal to the last GConv units (e.g., 512 = 2^latent_dim).
        self.pool_kernel_size = adjacency_shape[1]
        self.pool = nn.AvgPool1d(kernel_size=self.pool_kernel_size)
        
        self.q_dense = QuantumDense(latent_dim, n_layers=1, initializer="normal")
        
        # Build dense layers
        # QuantumDense outputs (batch, 2^latent_dim)
        quantum_output_dim = 2 ** 9
        
        self.dense_layers = []
        input_dim = quantum_output_dim
        for units in dense_units:
            layer = nn.Linear(input_dim, units)
            # Better initialization: Xavier/Glorot for encoder dense layers
            if hasattr(layer, 'weight'):
                init_fn = nn.init.glorot_normal()
                layer.weight = init_fn(layer.weight)
            self.dense_layers.append(layer)
            input_dim = units
        
        self.dropout_rate = dropout_rate
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Kaiming/He initialization for VAE latent layers
        # fc_mu: Use He normal initialization to prevent zero outputs
        if hasattr(self.fc_mu, 'weight'):
            init_fn = nn.init.he_normal()
            self.fc_mu.weight = init_fn(self.fc_mu.weight)
        if hasattr(self.fc_mu, 'bias') and self.fc_mu.bias is not None:
            # Zero bias for mu (prior mean is 0)
            self.fc_mu.bias = mx.zeros_like(self.fc_mu.bias)
        
        # fc_logvar: Use He normal initialization
        if hasattr(self.fc_logvar, 'weight'):
            init_fn = nn.init.he_normal()
            self.fc_logvar.weight = init_fn(self.fc_logvar.weight)
        if hasattr(self.fc_logvar, 'bias') and self.fc_logvar.bias is not None:
            # Initialize logvar bias to 2 (variance = exp(2) ≈ 7.39, higher initial variance)
            self.fc_logvar.bias = mx.array([2.0] * latent_dim, dtype=mx.float32)
        
    def __call__(self, inputs):
        adjacency, features = inputs
        # adjacency: (batch, bond_dim, num_atoms, num_atoms)
        # features: (batch, num_atoms, atom_dim)
        batch_size = adjacency.shape[0]
        
        # Process in smaller sub-batches to avoid memory issues
        # Match quantum layer micro-batch size
        sub_batch_size = min(20, batch_size)  # Try 20 as requested
        
        features_transformed_list = []
        for sub_start in range(0, batch_size, sub_batch_size):
            sub_end = min(sub_start + sub_batch_size, batch_size)
            sub_adj = adjacency[sub_start:sub_end]
            sub_feat = features[sub_start:sub_end]
            
            # Process each sample in sub-batch
            sub_features_list = []
            for b in range(sub_end - sub_start):
                batch_adj = sub_adj[b]  # (bond_dim, num_atoms, num_atoms)
                batch_features = sub_feat[b]  # (num_atoms, atom_dim)
                
                batch_features_transformed = batch_features
                for layer in self.gconv_layers:
                    batch_features_transformed = layer((batch_adj, batch_features_transformed))
                
                sub_features_list.append(batch_features_transformed)
            
            # Stack sub-batch and add to list
            sub_features = mx.stack(sub_features_list, axis=0)
            features_transformed_list.append(sub_features)
        
        # Concatenate all sub-batches: (batch, num_atoms, last_units)
        features_transformed = mx.concatenate(features_transformed_list, axis=0)
        
        # Pool over num_atoms dimension using the pooling layer
        # features_transformed: (batch, num_atoms, last_gconv_units)
        # AvgPool1d in MLX expects (batch, length, channels) and pools over length.
        x = self.pool(features_transformed)  # (batch, 1, last_gconv_units)
        x = mx.squeeze(x, axis=1)  # (batch, last_gconv_units)
        
        # QuantumDense: amplitude embedding + entangling circuit + measurement
        #x = self.q_dense(x)  # (batch, last_gconv_units)
        
        # Regular dense layers with activations
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.relu(x)
            x = nn.Dropout(self.dropout_rate)(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar