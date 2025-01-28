import torch
import torch.nn as nn

#################################################################################################
# Here we define the neural network classes that are presented in sections 2 and 3 of the paper #
#################################################################################################

# Deep linear network (section 2)
class LinearNN(nn.Module):
    def __init__(self, dim_x, dim_y, hidden_dims=None):
        """
        Deep Linear Neural Network. Uses no activation functions and allows for variable widths in all layers.
        
        Parameters:
        - dim_x (int): Input dimension.
        - dim_y (int): Output dimension.
        - hidden_dims (list of int, optional): Dimensions of hidden layers.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_dims = hidden_dims
        self.H = len(hidden_dims)  

        layers = []
        dims = [dim_x] + hidden_dims + [dim_y]  
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the linear neural network.
        """
        for layer in self.layers:
            x = layer(x)  
        return x

'''
# Deep nonlinear network (section 3)
class NonlinearNN(nn.Module):
    def __init__(self, dim_x, dim_y, *hidden_dims):
        """
        Deep Nonlinear Neural Network. Uses the ReLU activation function and allows for variable widths in all layers.
        
        Parameters:
        - dim_x (int): Input dimension.
        - dim_y (int): Output dimension.
        - hidden_dims (list of int): Dimensions of hidden layers.
        """
        super().__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_dims = hidden_dims
        self.H = len(hidden_dims)  
        #self.q = 1 / (p) mirar la def de p en el paper.

        layers = []
        dims = [dim_x] + list(hidden_dims) + [dim_y]  
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the linear neural network.
        """
        for layer in self.layers:
            x = layer(x) 
            x =  torch.nn.functional.relu(x)
        #x = self.q * x
        return x
'''