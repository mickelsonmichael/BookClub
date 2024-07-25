import torch
from torch import nn
from MLP import MLP
from MySequential import MySequential
from FixedHiddenMLP import FixedHiddenMLP
from NestMLP import NestMLP
from init_methods import init_normal, init_constant, init_xavier, init_42, my_init
from gpu_helpers import try_gpu, try_all_gpus

if __name__ == "__main__":
    net = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(10)
    )

    X = torch.rand(2, 20)

    print(net(X).shape)

    # use our hand-written model module
    net = MLP()

    print(net(X).shape)

    # use our hand-written sequential module
    net = MySequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(10)
    )

    print(net(X).shape)

    # use a module with constants
    net = FixedHiddenMLP()

    print(net(X))

    # mix and match multiple modules
    net = nn.Sequential(
        NestMLP(),
        nn.LazyLinear(20),
        FixedHiddenMLP()
    )

    print(net(X))

    # 6.2 Parameter Management
    net = nn.Sequential(
        nn.LazyLinear(8),
        nn.ReLU(),
        nn.LazyLinear(1)
    )

    X = torch.rand(size=(2, 4))

    print(net(X).shape)

    # index the layer by indexing the module
    # use the state_dict() function to inspect the parameters
    print(net[2].state_dict())

    # get the type of the parameters
    print(type(net[2].bias), net[2].bias.data)

    # access the gradient for the layer
    # if backpropogation has not occured, this is equal to None
    print(net[2].weight.grad)

    # you can access all parameters in all layers using named_parameters()
    print(
        [(name, param.shape) for name, param in net.named_parameters()]
    )

    # 6.2.2 Tied Parameters

    shared = nn.LazyLinear(8)  # a named layer
    net = nn.Sequential(
        nn.LazyLinear(8),
        nn.ReLU(),
        shared,  # insert the named layer
        nn.ReLU(),
        shared,  # insert the named layer
        nn.ReLU(),
        nn.LazyLinear(1)
    )

    net(X)  # train

    print(
        # check if the two instances of 'shared' have the same parameters now
        net[2].weight.data[0] == net[4].weight.data[0]
    )

    # set the value of one to some constant to ensure they are the same object in memory
    net[2].weight.data[0, 0] = 100

    print(
        # check again if they have both been updated to the constant value
        net[2].weight.data[0] == net[4].weight.data[0]
    )

    # 6.3 Parameter Initialization

    # PyTorch will initialize with some common conventions
    net = nn.Sequential(
        nn.LazyLinear(8),
        nn.ReLU(),
        nn.LazyLinear(1)
    )

    X = torch.rand((2, 4))

    print(net(X).shape)

    # initialize using a "normal" method
    net.apply(init_normal)

    print(net[0].weight.data[0], net[0].bias.data[0])

    # initialize using constant method
    net.apply(init_constant)
    print(net[0].weight.data[0], net[0].bias.data[0])

    # initialize layers differently
    net[0].apply(init_xavier)
    net[2].apply(init_42)

    print(net[0].weight.data[0])
    print(net[2].weight.data)

    # initialize using a custom initializer
    net.apply(my_init)

    print(net[0].weight[:2])

    # set parameters directly
    net[0].weight.data[:] += 1
    net[0].weight.data[0, 0] = 42

    print(net[0].weight.data[0])

    # 6.4 Lazy Initialization

    net = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(10)
    )

    # Because the model has not seen data yet, it's size is uninitialized
    print(net[0].weight) # <UninitializedParameter>

    X = torch.rand(2, 10)

    net(X) # Initialize the parameters by passing data

    print(net[0].weight.shape) # Should not be initialized as [256, 20]

    # 6.7 GPUs

    X = torch.tensor([1, 2, 3])

    # Tensors are created on the CPU by default
    print(X.device) # device(type='cpu')

    # You can initialize a tensor on the GPU by specifying the device
    X = torch.tensor([1, 2, 3], device=try_gpu())

    print(X.device)

    # Initialize a new Tensor on a different device
    Y = torch.rand(2, 3, device='cpu')

    # Copy the Tensor to the GPU
    # If the Tensor is already on the GPU, the original value is returned to save memory
    Z = Y.cuda(0)

    # Now you can do operations
    print(X + Z)

    # 6.7.3 Neural Networks and GPUs

    # Neural Networks can also be initialized on the GPU
    net = nn.Sequential(nn.LazyLinear(1))
    net = net.to(device=try_gpu())

    