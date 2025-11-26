import mlx.core as mx
import mlx.nn as nn
from quantum.layers import SimpleEntangler, EncoderLayer
import numpy as np


class QuantumDense(nn.Module):
    """
    Fully trainable quantum dense layer.
    Takes input features, initializes entangled state with trainable weights,
    encodes input features, and returns quantum state probabilities.
    """
    
    def __init__(
        self,
        wires: int,
        n_layers: int = 1,
        initializer: str = "uniform"
    ):
        """
        Args:
            wires: Number of qubits
            n_layers: Number of rotation layers for entanglement
            use_entanglement: Whether to apply CNOT gates for entanglement
            rotation: Rotation axis - 'X', 'Y', or 'Z' (default: 'Y')
            initializer: Parameter initializer - 'uniform' or 'normal'
        """
        super().__init__()
        
        self.wires = wires
        self.n_layers = n_layers
        self.state_dim = 2 ** wires
        
        # Trainable rotation angles: (n_layers, wires)
        if initializer == "uniform":
            # Initialize angles uniformly in [0, 2π]
            self.weights = mx.random.uniform(0, 2*np.pi, (n_layers, wires), dtype=mx.float32)
        elif initializer == "normal":
            # Initialize angles from normal distribution
            self.weights = mx.random.normal((n_layers, wires), loc=0.0, scale=0.1, dtype=mx.float32)
        else:
            raise ValueError(f"Unsupported initializer: {initializer}")
    
    def __call__(self, inputs: mx.array) -> mx.array:
        """
        Forward pass.
        
        Args:
            inputs: Input features of shape (batch_size, wires)
        
        Returns:
            Output probabilities of shape (batch_size, 2^wires)
        """
        # Process in smaller batches to avoid memory issues
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        
        batch_size = inputs.shape[0]
        max_batch = 20  # Try 20 as requested
        
        if batch_size <= max_batch:
            # Small batch - process directly
            dev = SimpleEntangler(self.wires, weights=self.weights, rotation='Z')
            encoded_probs = EncoderLayer(self.wires, dev=dev, features=inputs, rotation='Y')
            return encoded_probs
        else:
            # Large batch - process in chunks
            results = []
            for i in range(0, batch_size, max_batch):
                end_idx = min(i + max_batch, batch_size)
                batch_inputs = inputs[i:end_idx]
                
                dev = SimpleEntangler(self.wires, weights=self.weights, rotation='Z')
                batch_probs = EncoderLayer(self.wires, dev=dev, features=batch_inputs, rotation='Y')
                results.append(batch_probs)
            
            return mx.concatenate(results, axis=0)
