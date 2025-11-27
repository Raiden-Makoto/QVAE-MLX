import mlx.core as mx
import numpy as np


def one_hot(indices, depth, axis=-1):
    """
    One-hot encode indices. MLX doesn't have one_hot, so we implement it.
    
    Args:
        indices: MLX array of indices to encode
        depth: Number of classes/depth of one-hot encoding
        axis: Axis along which to insert the one-hot dimension
    
    Returns:
        One-hot encoded array
    """
    indices = mx.array(indices) if not isinstance(indices, mx.array) else indices
    
    # Get shape
    shape = list(indices.shape)
    if axis < 0:
        axis = len(shape) + axis + 1
    shape.insert(axis, depth)
    
    # Convert to numpy for indexing and clamp
    indices_np = np.array(indices)
    indices_np = np.clip(indices_np, 0, depth - 1)  # Clamp in numpy
    output_np = np.zeros(shape, dtype=np.float32)
    
    if indices.ndim == 2:
        # (batch, spatial) -> (batch, depth, spatial) if axis=1
        # or (batch, spatial, depth) if axis=2
        batch_size, spatial = indices.shape
        for b in range(batch_size):
            for s in range(spatial):
                idx = int(indices_np[b, s])
                idx = max(0, min(idx, depth - 1))  # Ensure valid range
                if axis == 1:
                    output_np[b, idx, s] = 1.0
                elif axis == 2:
                    output_np[b, s, idx] = 1.0
                else:
                    idx_tuple = [b, s]
                    idx_tuple.insert(axis - 1, idx)
                    output_np[tuple(idx_tuple)] = 1.0
    elif indices.ndim == 3:
        # (batch, spatial1, spatial2) -> (batch, depth, spatial1, spatial2) if axis=1
        # or (batch, spatial1, depth, spatial2) if axis=2
        batch_size, spatial1, spatial2 = indices.shape
        for b in range(batch_size):
            for s1 in range(spatial1):
                for s2 in range(spatial2):
                    idx = int(indices_np[b, s1, s2])
                    # indices_np is already clamped, but ensure idx is valid
                    idx = max(0, min(idx, depth - 1))
                    if axis == 1:
                        output_np[b, idx, s1, s2] = 1.0
                    elif axis == 2:
                        output_np[b, s1, idx, s2] = 1.0
                    elif axis == 3:
                        output_np[b, s1, s2, idx] = 1.0
                    else:
                        # General case: insert at specified axis
                        idx_tuple = [b, s1, s2]
                        idx_tuple.insert(axis - 1, idx)
                        output_np[tuple(idx_tuple)] = 1.0
    else:
        raise ValueError(f"Unsupported indices ndim: {indices.ndim}")
    
    return mx.array(output_np)

