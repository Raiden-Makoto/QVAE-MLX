import mlx.core as mx
import mlx.nn as nn


class Reshape(nn.Module):
    """
    Reshape layer similar to Keras Reshape.
    
    Args:
        target_shape: Target shape tuple. Can use -1 for one dimension to infer it.
    """
    
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    
    def __call__(self, x):
        """
        Reshape input tensor.
        
        Args:
            x: Input tensor
        
        Returns:
            Reshaped tensor
        """
        # Handle -1 in target_shape (infer dimension)
        if -1 in self.target_shape:
            # Calculate inferred dimension
            total_elements = 1
            for dim in x.shape:
                total_elements *= dim
            
            known_elements = 1
            for dim in self.target_shape:
                if dim != -1:
                    known_elements *= dim
            
            inferred_dim = total_elements // known_elements
            shape = tuple(inferred_dim if d == -1 else d for d in self.target_shape)
        else:
            shape = self.target_shape
        
        return mx.reshape(x, shape)

