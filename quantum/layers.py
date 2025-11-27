from .gates import device, Hadamard, CNOT, RX, RY, RZ, MLXQuantumDevice
import mlx.core as mx
from typing import Callable, Optional

def SimpleEntangler(wires: int, dev: MLXQuantumDevice = None, weights: mx.array = None, rotation: str = "RX") -> MLXQuantumDevice:
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
    if dev is None: dev = device(wires)
    else: dev.reset()
    
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
    rotation: str = "Y"
) -> mx.array:
    """
    Implements angle embedding (PennyLane equivalent).
    Encodes classical features into quantum states using rotation gates.
    Supports both single samples and batched data (vectorized like PennyLane).
    
    Args:
        wires: Number of qubits
        dev: Optional quantum device (creates new one if None, reused for batches)
        features: Input features to encode. Can be:
            - 1D array: (wires,) for single sample → returns (2^wires,)
            - 2D array: (batch_size, wires) for batched data → returns (batch_size, 2^wires)
            If None, uses zeros.
        rotation: Rotation axis - 'X', 'Y', or 'Z' (default: 'Y')
    
    Returns:
        - Single sample: quantum state array of shape (2^wires,)
        - Batched: quantum state array of shape (batch_size, 2^wires)
    """
    from quantum import state
    
    if features is None:
        features = mx.zeros(wires, dtype=mx.float32)
    
    features = mx.array(features) if not isinstance(features, mx.array) else features
    
    rotation = rotation.upper()
    if rotation == "X":
        rotation_gate = RX
    elif rotation == "Y":
        rotation_gate = RY
    elif rotation == "Z":
        rotation_gate = RZ
    else:
        raise ValueError(f"Rotation must be 'X', 'Y', or 'Z', got '{rotation}'")
    
    if features.ndim == 1:
        if features.shape[0] != wires:
            raise ValueError(f"Number of features ({features.shape[0]}) must match number of wires ({wires})")
        
        if dev is None:
            dev = device(wires)
        else:
            dev.reset()
        
        for i in range(wires):
            angle = float(features[i].item() if hasattr(features[i], 'item') else features[i])
            rotation_gate(angle, i, dev)
        
        # Return probabilities (|amplitude|^2) instead of complex state
        return mx.abs(state(dev)) ** 2
    
    elif features.ndim == 2:
        batch_size, num_features = features.shape
        if num_features != wires:
            raise ValueError(f"Number of features ({num_features}) must match number of wires ({wires})")
        
        # Vectorized batch processing
        state_dim = 2 ** wires
        
        # Initialize batched state: (batch_size, 2^wires)
        # All samples start from |0...0⟩
        if dev is None:
            # Initialize all to |0...0⟩ = [1, 0, 0, ..., 0]
            batched_state = mx.zeros((batch_size, state_dim), dtype=mx.complex64)
            batched_state[:, 0] = 1.0 + 0j
        else:
            # Copy initial state from provided device for all samples
            initial_state = mx.array(dev.state)
            batched_state = mx.broadcast_to(initial_state[None, :], (batch_size, state_dim))
        
        # Vectorized angle processing
        angles = features  # (batch_size, wires)
        
        # For each wire, apply rotation gates to all batch samples simultaneously
        for wire_idx in range(wires):
            # Get angles for this wire across all batch samples: (batch_size,)
            wire_angles = angles[:, wire_idx]
            
            # Compute rotation matrices for all batch samples
            half_angles = wire_angles / 2.0
            
            if rotation == "X":
                # RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
                cos_half = mx.cos(half_angles)  # (batch_size,)
                sin_half = mx.sin(half_angles)  # (batch_size,)
                
                # Build rotation matrices: (batch_size, 2, 2)
                rot_matrices = mx.zeros((batch_size, 2, 2), dtype=mx.complex64)
                rot_matrices[:, 0, 0] = cos_half
                rot_matrices[:, 0, 1] = -1j * sin_half
                rot_matrices[:, 1, 0] = -1j * sin_half
                rot_matrices[:, 1, 1] = cos_half
                
            elif rotation == "Y":
                # RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
                cos_half = mx.cos(half_angles)  # (batch_size,)
                sin_half = mx.sin(half_angles)  # (batch_size,)
                
                rot_matrices = mx.zeros((batch_size, 2, 2), dtype=mx.complex64)
                rot_matrices[:, 0, 0] = cos_half
                rot_matrices[:, 0, 1] = -sin_half
                rot_matrices[:, 1, 0] = sin_half
                rot_matrices[:, 1, 1] = cos_half
                
            elif rotation == "Z":
                # RZ(θ) = [[exp(-i*θ/2), 0], [0, exp(i*θ/2)]]
                exp_neg = mx.exp(-1j * half_angles)  # (batch_size,)
                exp_pos = mx.exp(1j * half_angles)  # (batch_size,)
                
                rot_matrices = mx.zeros((batch_size, 2, 2), dtype=mx.complex64)
                rot_matrices[:, 0, 0] = exp_neg
                rot_matrices[:, 1, 1] = exp_pos
            
            # Build full operator for this wire: (batch_size, 2^wires, 2^wires)
            # Tensor product: I ⊗ ... ⊗ R ⊗ ... ⊗ I
            identity = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.complex64)
            full_operators = []
            
            for b in range(batch_size):
                # Build operator for this batch sample
                op = mx.array([[1.0]], dtype=mx.complex64)
                for i in range(wires):
                    if i == wire_idx:
                        op = mx.kron(op, rot_matrices[b])
                    else:
                        op = mx.kron(op, identity)
                full_operators.append(op)
            
            # Stack to (batch_size, 2^wires, 2^wires)
            full_op = mx.stack(full_operators, axis=0)
            
            # Apply to all batch samples using batched matrix-vector multiplication
            # (batch_size, 2^wires, 2^wires) @ (batch_size, 2^wires, 1) -> (batch_size, 2^wires, 1)
            # Use stop_gradient on the state to prevent complex64 tracking
            # Gradients will still flow through rotation matrices (which depend on float32 angles)
            state_input = mx.stop_gradient(batched_state)
            batched_state = mx.squeeze(mx.matmul(full_op, state_input[:, :, None]), axis=2)
            # Wrap result in stop_gradient to prevent tracking, but gradients flow through full_op
            batched_state = mx.stop_gradient(batched_state)
        
        # Return probabilities (|amplitude|^2) instead of complex state
        # This converts complex64 to float32, which is what we need for gradients
        return mx.abs(batched_state) ** 2
    
    else:
        raise ValueError(f"Features must be 1D or 2D array, got {features.ndim}D")