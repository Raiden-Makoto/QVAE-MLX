from .gates import device, Hadamard, CNOT, RX, RY, RZ, MLXQuantumDevice
import mlx.core as mx
from typing import Callable, Optional

def SimpleEntangler(
    wires: int,
    dev: MLXQuantumDevice = None,
    weights: mx.array = None,
    rotation: str = "RX",
    reset: bool = True,
) -> MLXQuantumDevice:
    """
    Implements a simple quantum layer that initializes qubits with Hadamard gates
    then entangles them in a ring topology, with optional multi-layer rotations.
    
    Args:
        wires: Number of qubits
        dev: Optional quantum device (creates new one if None)
        weights: Rotation weights of shape (n_layers, n_qubits). If None, no rotations applied.
        rotation: Rotation axis - 'X', 'Y', 'Z', 'RX', 'RY', or 'RZ' (default: 'RX')
    
    Returns:
        The quantum device with the circuit applied
    """
    if dev is None:
        dev = device(wires)
    elif reset:
        dev.reset()
    
    for i in range(wires):
        Hadamard(i, dev)
    
    for i in range(wires):
        target = (i + 1) % wires
        CNOT(i, target, dev)

    # Apply rotation gates if weights provided
    if weights is not None:
        weights = mx.array(weights) if not isinstance(weights, mx.array) else weights
        
        # Handle both (n_qubits,) and (n_layers, n_qubits) shapes
        if weights.ndim == 1:
            # Single layer: (n_qubits,) -> (1, n_qubits)
            weights = weights[None, :]
        elif weights.ndim != 2:
            raise ValueError(f"Weights must be 1D or 2D, got {weights.ndim}D")
        
        n_layers, n_qubits = weights.shape
        if n_qubits != wires:
            raise ValueError(f"Number of qubits in weights ({n_qubits}) must match number of wires ({wires})")
        
        rotation = rotation.upper()
        if rotation == "X" or rotation == "RX":
            rotation_gate = RX
        elif rotation == "Y" or rotation == "RY":
            rotation_gate = RY
        elif rotation == "Z" or rotation == "RZ":
            rotation_gate = RZ
        else:
            raise ValueError(f"Rotation must be 'X', 'Y', 'Z', 'RX', 'RY', or 'RZ', got '{rotation}'")
        
        # Apply rotations for each layer
        for layer in range(n_layers):
            for i in range(wires):
                angle = float(weights[layer, i].item() if hasattr(weights[layer, i], 'item') else weights[layer, i])
                rotation_gate(angle, i, dev)

    return dev

def EncoderLayer(
    wires: int,
    dev: MLXQuantumDevice = None,
    features: Optional[mx.array] = None,
    rotation: str = "Y",
    return_state: bool = True,
) -> mx.array:
    """
    Quantum data encoding layer using **amplitude embedding only**, with
    **no measurement**.
    
    Interprets the input as amplitudes of a 2**wires-dimensional state,
    L2-normalizes them, and returns the complex state vector(s).
    
    Args:
        wires: Number of qubits.
        dev: Unused (kept for API compatibility).
        features: Input features to encode. Can be:
            - 1D array: (2**wires,)        → single state
            - 2D array: (batch, 2**wires) → batched states
            If None, uses zeros of length 2**wires.
        rotation: Unused (kept for API compatibility).
        return_state: Kept for compatibility; always returns state.
    
    Returns:
        - Single sample: complex state vector of shape (2**wires,)
        - Batched: complex state matrix of shape (batch, 2**wires)
    """
    state_dim = 2**wires
    
    if features is None:
        features = mx.zeros(state_dim, dtype=mx.float32)
    
    features = mx.array(features) if not isinstance(features, mx.array) else features
    
    if features.ndim == 1:
        num_features = features.shape[0]
        if num_features != state_dim:
            raise ValueError(
                f"1D features must have length {state_dim} for amplitude embedding, "
                f"got {num_features}"
            )
        
        amps = mx.array(features, dtype=mx.float32)
        norm = mx.linalg.norm(amps)
        norm = mx.maximum(norm, 1e-8)
        amps = amps / norm
        return mx.array(amps, dtype=mx.complex64)
    
    elif features.ndim == 2:
        batch_size, num_features = features.shape
        if num_features != state_dim:
            raise ValueError(
                f"2D features must have last dim {state_dim} for amplitude embedding, "
                f"got {num_features}"
            )
        
        amps = mx.array(features, dtype=mx.float32)
        norms = mx.linalg.norm(amps, axis=1, keepdims=True)
        norms = mx.maximum(norms, 1e-8)
        amps = amps / norms  # (batch, state_dim)
        return mx.array(amps, dtype=mx.complex64)
    
    else:
        raise ValueError(f"Features must be 1D or 2D array, got {features.ndim}D")