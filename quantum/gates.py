import mlx.core as mx
from typing import Callable, Any
from functools import wraps

# This file contains a PennyLane-like interface for quantum computing with MLX

class MLXQuantumDevice:
    """A quantum device that manages state and executes quantum circuits."""
    
    def __init__(self, wires: int, shots: int = None):
        """
        Initialize a quantum device.
        
        Args:
            wires: Number of qubits
            shots: Number of measurement shots (None for exact simulation)
        """
        self.wires = wires
        self.shots = shots
        self.state = None
        self.reset()
    
    def reset(self):
        """Reset the device to the |0...0⟩ state."""
        # Initialize to |00...0⟩ = [1, 0, 0, ..., 0]
        self.state = mx.zeros(2**self.wires, dtype=mx.complex64)
        self.state[0] = 1.0 + 0j
    
    def apply_operation(self, operation_matrix: mx.array):
        """Apply a unitary operation to the current state."""
        if self.state is None:
            self.reset()
        # State evolution: |ψ⟩ → U|ψ⟩
        self.state = operation_matrix @ self.state
    
    def get_state(self) -> mx.array:
        """Get the current quantum state."""
        if self.state is None:
            self.reset()
        return self.state
    
    def measure_probabilities(self) -> mx.array:
        """Measure the probabilities of all computational basis states."""
        state = self.get_state()
        # Probabilities are |⟨i|ψ⟩|²
        return mx.abs(state) ** 2
    
    def measure_expectation(self, observable: mx.array) -> mx.array:
        """Measure the expectation value of an observable: ⟨ψ|O|ψ⟩."""
        state = self.get_state()
        # ⟨ψ|O|ψ⟩ = state† @ O @ state
        # state is a vector, so we need: state.conj() @ observable @ state
        result = mx.real(state.conj() @ observable @ state)
        # Convert to scalar if needed
        if result.ndim == 0:
            return result.item()
        return result[0]
    
    def sample(self, wire: int) -> int:
        """Sample a measurement result for a specific wire."""
        probs = self.measure_probabilities()
        # For multi-qubit, we need to marginalize over other qubits
        # This is a simplified version - full implementation would marginalize
        if self.shots is None:
            # Return expectation value
            z_op = self._pauli_z_operator(wire)
            return self.measure_expectation(z_op)
        else:
            # Sample from distribution
            idx = mx.random.categorical(probs)
            # Extract bit for this wire
            return (idx // (2 ** (self.wires - wire - 1))) % 2
    
    def _pauli_z_operator(self, wire: int) -> mx.array:
        """Get the Pauli-Z operator for a specific wire."""
        op = mx.array([[1.0, 0.0], [0.0, -1.0]], dtype=mx.complex64)
        return self._tensor_product_operator(op, wire)
    
    def _tensor_product_operator(self, single_qubit_op: mx.array, target_wire: int) -> mx.array:
        """Create a multi-qubit operator by tensoring with identity on other wires."""
        # MLX doesn't support complex64 for eye, so we create identity manually
        identity = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.complex64)
        result = mx.array([[1.0]], dtype=mx.complex64)
        for i in range(self.wires):
            if i == target_wire:
                result = mx.kron(result, single_qubit_op)
            else:
                result = mx.kron(result, identity)
        return result


class QuantumNode:
    """Context manager for quantum circuit execution."""
    
    def __init__(self, device: MLXQuantumDevice):
        self.device = device
        self.operations = []
    
    def __enter__(self):
        self.device.reset()
        return self
    
    def __exit__(self, *args):
        # Execute all operations
        for op_matrix in self.operations:
            self.device.apply_operation(op_matrix)
        return False


# Global device registry
_default_device = None

def device(wires: int, shots: int = None) -> MLXQuantumDevice:
    """Create or get a quantum device (similar to qml.device)."""
    global _default_device
    _default_device = MLXQuantumDevice(wires, shots)
    return _default_device


def qnode(func: Callable) -> Callable:
    """Decorator for quantum node functions (similar to @qml.qnode)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _default_device
        if _default_device is None:
            raise RuntimeError("No device specified. Call device() first.")
        
        # Reset device state before executing circuit
        _default_device.reset()
        
        # Execute the function (operations are applied immediately)
        result = func(*args, **kwargs)
        
        return result
    
    return wrapper


# Gate operations (similar to qml.Hadamard, qml.PauliX, etc.)

def Hadamard(wire: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply Hadamard gate to a wire."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    H = mx.array([[1.0, 1.0], [1.0, -1.0]], dtype=mx.complex64) / mx.sqrt(2.0)
    op = device._tensor_product_operator(H, wire)
    device.apply_operation(op)
    return op


def PauliX(wire: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply Pauli-X (NOT) gate to a wire."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    X = mx.array([[0.0, 1.0], [1.0, 0.0]], dtype=mx.complex64)
    op = device._tensor_product_operator(X, wire)
    device.apply_operation(op)
    return op


def PauliY(wire: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply Pauli-Y gate to a wire."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    Y = mx.array([[0.0, -1j], [1j, 0.0]], dtype=mx.complex64)
    op = device._tensor_product_operator(Y, wire)
    device.apply_operation(op)
    return op


def PauliZ(wire: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply Pauli-Z gate to a wire."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    Z = mx.array([[1.0, 0.0], [0.0, -1.0]], dtype=mx.complex64)
    op = device._tensor_product_operator(Z, wire)
    device.apply_operation(op)
    return op


def CNOT(control: int, target: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply CNOT gate (control on 'control', target on 'target')."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    # CNOT matrix in computational basis (2-qubit)
    cnot_2q = mx.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=mx.complex64)
    
    if device.wires == 2:
        op = cnot_2q
    else:
        # For multi-qubit, construct CNOT matrix directly
        # CNOT flips target qubit when control is |1⟩
        dim = 2 ** device.wires
        op = mx.zeros((dim, dim), dtype=mx.complex64)
        
        for i in range(dim):
            control_val = (i >> (device.wires - control - 1)) & 1
            
            if control_val == 0:
                # Control is 0: target unchanged
                op[i, i] = 1.0
            else:
                # Control is 1: flip target qubit
                j = i ^ (1 << (device.wires - target - 1))
                op[i, j] = 1.0
    
    device.apply_operation(op)
    return op


def RX(angle, wire: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply rotation around X-axis: RX(θ) = exp(-i*θ*X/2)."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    # Convert angle to MLX array if needed
    if not isinstance(angle, mx.array):
        angle = mx.array(float(angle), dtype=mx.float32)
    
    # RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
    half_angle = angle / 2.0
    cos_half = mx.cos(half_angle)
    sin_half = mx.sin(half_angle)
    
    # Extract scalar values for matrix construction
    c = float(cos_half.item() if hasattr(cos_half, 'item') else cos_half)
    s = float(sin_half.item() if hasattr(sin_half, 'item') else sin_half)
    
    RX_matrix = mx.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=mx.complex64)
    
    op = device._tensor_product_operator(RX_matrix, wire)
    device.apply_operation(op)
    return op


def RY(angle, wire: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply rotation around Y-axis: RY(θ) = exp(-i*θ*Y/2)."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    # Convert angle to MLX array if needed
    if not isinstance(angle, mx.array):
        angle = mx.array(float(angle), dtype=mx.float32)
    
    # RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    half_angle = angle / 2.0
    cos_half = mx.cos(half_angle)
    sin_half = mx.sin(half_angle)
    
    # Extract scalar values for matrix construction
    c = float(cos_half.item() if hasattr(cos_half, 'item') else cos_half)
    s = float(sin_half.item() if hasattr(sin_half, 'item') else sin_half)
    
    RY_matrix = mx.array([
        [c, -s],
        [s, c]
    ], dtype=mx.complex64)
    
    op = device._tensor_product_operator(RY_matrix, wire)
    device.apply_operation(op)
    return op


def RZ(angle, wire: int, device: MLXQuantumDevice = None) -> mx.array:
    """Apply rotation around Z-axis: RZ(θ) = exp(-i*θ*Z/2)."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    # Convert angle to MLX array if needed
    if not isinstance(angle, mx.array):
        angle = mx.array(float(angle), dtype=mx.float32)
    
    # RZ(θ) = [[exp(-i*θ/2), 0], [0, exp(i*θ/2)]]
    half_angle = angle / 2.0
    exp_neg = mx.exp(-1j * half_angle)
    exp_pos = mx.exp(1j * half_angle)
    
    # Extract scalar values for matrix construction
    e_neg = complex(exp_neg.item() if hasattr(exp_neg, 'item') else exp_neg)
    e_pos = complex(exp_pos.item() if hasattr(exp_pos, 'item') else exp_pos)
    
    RZ_matrix = mx.array([
        [e_neg, 0.0],
        [0.0, e_pos]
    ], dtype=mx.complex64)
    
    op = device._tensor_product_operator(RZ_matrix, wire)
    device.apply_operation(op)
    return op


# Measurement operations

def expval(observable, device: MLXQuantumDevice = None) -> mx.array:
    """Measure expectation value of an observable."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    if isinstance(observable, str):
        # Handle string observables like "PauliZ(0)"
        if observable.startswith("PauliZ"):
            wire = int(observable.split("(")[1].split(")")[0])
            obs = device._pauli_z_operator(wire)
        else:
            raise ValueError(f"Unknown observable: {observable}")
    else:
        obs = observable
    
    return device.measure_expectation(obs)


def probs(wires: list = None, device: MLXQuantumDevice = None) -> mx.array:
    """Measure probabilities of computational basis states."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    return device.measure_probabilities()


def state(device: MLXQuantumDevice = None) -> mx.array:
    """Get the current quantum state."""
    if device is None:
        device = _default_device
    if device is None:
        raise RuntimeError("No device specified.")
    
    return device.get_state()
