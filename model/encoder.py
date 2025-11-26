import mlx.core as mx
import mlx.nn as nn
from model.dense import QuantumDense
from model.rgcn import RelationalGraphConvLayer

class Encoder(nn.Module):
    def __init__(
        self,
        gconv_units,
        latent_dim,
        adjacency_shape,
        feature_shape,
        dense_units,
        dropout_rate,
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
        
        # Pooling: need to pool over num_atoms dimension
        # GConv output: (batch, num_atoms, last_units)
        # After transpose: (batch, last_units, num_atoms)
        # After pooling: (batch, last_units, 1) -> squeeze -> (batch, last_units)
        self.pool_kernel_size = adjacency_shape[1]
        self.pool = nn.AvgPool1d(kernel_size=self.pool_kernel_size)
        
        # Map from last gconv units to latent_dim for quantum dense
        last_gconv_units = gconv_units[-1] if gconv_units else atom_dim
        self.pre_q_linear = nn.Linear(last_gconv_units, latent_dim)
        
        self.q_dense = QuantumDense(latent_dim, n_layers=1, initializer="normal")
        
        # Build dense layers
        # QuantumDense outputs (batch, 2^latent_dim)
        quantum_output_dim = 2 ** latent_dim
        
        self.dense_layers = []
        input_dim = quantum_output_dim
        for units in dense_units:
            layer = nn.Linear(input_dim, units)
            self.dense_layers.append(layer)
            input_dim = units
        
        self.dropout_rate = dropout_rate
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
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
        
        # Pool over num_atoms dimension: average over atoms
        # features_transformed: (batch, num_atoms, units)
        x = mx.mean(features_transformed, axis=1)  # (batch, units)
        
        # Map to latent_dim for quantum dense
        x = self.pre_q_linear(x)  # (batch, latent_dim)
        
        # Quantum dense layer
        x = self.q_dense(x)  # (batch, 2^latent_dim)
        
        # Regular dense layers
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.Dropout(self.dropout_rate)(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar