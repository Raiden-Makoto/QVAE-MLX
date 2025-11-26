import mlx.core as mx
import mlx.nn as nn

class RelationalGraphConvLayer(nn.Module):
    def __init__(
        self,
        bond_dim,
        atom_dim,
        units=128,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs
    ):
        super().__init__()

        self.units = units
        self.use_bias = use_bias

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        kernel_shape = (bond_dim, atom_dim, units)
        kernel_array = mx.zeros(kernel_shape, dtype=mx.float32)
        
        if kernel_initializer == "glorot_uniform":
            init_fn = nn.init.glorot_uniform()
        elif kernel_initializer == "glorot_normal":
            init_fn = nn.init.glorot_normal()
        elif kernel_initializer == "he_uniform":
            init_fn = nn.init.he_uniform()
        elif kernel_initializer == "he_normal":
            init_fn = nn.init.he_normal()
        else:
            raise ValueError(f"Unsupported kernel_initializer: {kernel_initializer}")
        
        self.kernel = init_fn(kernel_array)

        if self.use_bias:
            bias_shape = (bond_dim, 1, units)
            bias_array = mx.zeros(bias_shape, dtype=mx.float32)
            
            if bias_initializer == "zeros":
                init_fn = nn.init.constant(0.0)
            elif bias_initializer == "ones":
                init_fn = nn.init.constant(1.0)
            else:
                raise ValueError(f"Unsupported bias_initializer: {bias_initializer}")
            
            self.bias = init_fn(bias_array)
        else:
            self.bias = None

    def __call__(self, inputs):
        adjacency, features = inputs
        bond_dim = adjacency.shape[0]
        
        x_list = []
        for i in range(bond_dim):
            x_i = mx.matmul(adjacency[i], features)
            x_list.append(x_i)
        x = mx.stack(x_list, axis=0)
        
        y_list = []
        for i in range(bond_dim):
            y_i = mx.matmul(x[i], self.kernel[i])
            y_list.append(y_i)
        y = mx.stack(y_list, axis=0)
        
        if self.use_bias:
            y = y + self.bias
        
        y_reduced = mx.sum(y, axis=0)
        return self.activation(y_reduced)
