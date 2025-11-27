import mlx.core as mx
import mlx.nn as nn
from quantum.layers import SimpleEntangler, EncoderLayer
from quantum.gates import device
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
            inputs: Input features of shape (batch_size, 2^wires) for amplitude embedding
        
        Returns:
            Output probabilities of shape (batch_size, 2^wires)
        """
        # Process in smaller batches to avoid memory issues
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        
        batch_size, input_dim = inputs.shape
        if input_dim != self.state_dim:
            raise ValueError(
                f"QuantumDense expects input of shape (batch, {self.state_dim}) for amplitude embedding, got last dim {input_dim}"
            )
        
        max_batch = 20  # Try 20 as requested
        results = []
        
        for i in range(0, batch_size, max_batch):
            end_idx = min(i + max_batch, batch_size)
            batch_inputs = inputs[i:end_idx]  # (sub_batch, state_dim)
            
            # Use EncoderLayer to perform amplitude embedding and return complex state
            states = EncoderLayer(
                self.wires,
                dev=None,
                features=batch_inputs,
                return_state=True,
            )  # (sub_batch, state_dim), complex64
            
            sub_results = []
            for b in range(batch_inputs.shape[0]):
                state_vec = states[b]
                
                # Prepare quantum device with embedded state
                dev = device(self.wires)
                dev.state = state_vec
                
                # Apply SimpleEntangler *after* amplitude embedding without resetting
                dev = SimpleEntangler(
                    self.wires,
                    dev=dev,
                    weights=self.weights,
                    rotation='Z',
                    reset=False,
                )
                
                # Measure probabilities
                from quantum import state as _state
                probs = mx.abs(_state(dev)) ** 2  # (state_dim,)
                sub_results.append(probs)
            
            sub_results = mx.stack(sub_results, axis=0)  # (sub_batch, state_dim)
            results.append(sub_results)
        
        return mx.concatenate(results, axis=0)
