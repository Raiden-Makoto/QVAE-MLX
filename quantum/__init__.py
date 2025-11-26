"""
MLX Quantum - A PennyLane-like interface for quantum computing with MLX
"""

from .gates import (
    device,
    qnode,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    CNOT,
    RX,
    RY,
    RZ,
    expval,
    probs,
    state,
    MLXQuantumDevice,
)

__all__ = [
    "device",
    "qnode",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "CNOT",
    "RX",
    "RY",
    "RZ",
    "expval",
    "probs",
    "state",
    "MLXQuantumDevice",
]

# For convenience, create aliases similar to PennyLane
H = Hadamard
X = PauliX
Y = PauliY
Z = PauliZ

