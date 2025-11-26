"""
Example usage of MLX Quantum - demonstrating PennyLane-like interface
"""

from quantum import device, qnode, H, X, Z, CNOT, expval, probs, state

# Create a device (similar to qml.device('default.qubit', wires=2))
dev = device(wires=2)

# Example 1: Simple Hadamard gate
print("=== Example 1: Hadamard Gate ===")
H(0, dev)
print(f"State after H(0): {state(dev)}")
print(f"Probabilities: {probs(dev)}")
print()

# Reset and try a Bell state
print("=== Example 2: Bell State (|00⟩ + |11⟩)/√2 ===")
dev.reset()
H(0, dev)
CNOT(0, 1, dev)
print(f"State: {state(dev)}")
print(f"Probabilities: {probs(dev)}")
print(f"Expectation of Z on qubit 0: {expval('PauliZ(0)', dev)}")
print()

# Example 3: Using with a function (qnode-like)
print("=== Example 3: Quantum Circuit Function ===")
dev.reset()

@qnode
def bell_circuit():
    """Create a Bell state."""
    H(0)
    CNOT(0, 1)
    return expval('PauliZ(0)')

result = bell_circuit()
print(f"Expectation value: {result}")
print(f"Final state: {state(dev)}")
print()

# Example 4: More complex circuit
print("=== Example 4: GHZ State ===")
dev = device(wires=3)
H(0, dev)
CNOT(0, 1, dev)
# Note: CNOT for >2 qubits needs implementation
print(f"State after H(0) and CNOT(0,1): {state(dev)}")

