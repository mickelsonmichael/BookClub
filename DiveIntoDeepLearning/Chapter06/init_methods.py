from torch import nn


def init_normal(module):
    if type(module) == nn.Linear:
        # Initialize all weights as Gaussian random numbers
        # with a standard deviation of 0.01
        # Bias parameters are set to 0
        nn.init.normal_(module.weight, mean=0, std=0.01)

        nn.init.zeros_(module.bias)


def init_constant(module):
    if type(module) == nn.Linear:
        # Initialize all weights as a constant value (1)
        # Bias parameters are set to 0
        nn.init.constant_(module.weight, 1)

        nn.init.zeros_(module.bias)

def init_xavier(module):
    if type(module) == nn.Linear:
        # Initialize using the Xavier initializer
        # https://cs230.stanford.edu/section/4/
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        # initialize to a constant of 42
        nn.init.constant_(module.weight, 42)

def my_init(module):
    if type(module) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in module.named_parameters()][0]      
        )

        nn.init.uniform_(module.weight, -10, 10)

        module.weight.data *= module.weight.data.abs() >= 5