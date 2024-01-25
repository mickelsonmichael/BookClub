import torch
from torch import nn
from corr2d import corr2d
from comp_conv2d import comp_conv2d

if __name__ == "__main__":
    
    # =======================================
    # 7.2.1 - The Cross-Correlation Operation
    # =======================================

    # Create a 3x3 matrix
    X = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ])

    # Create a 2x2 matrix as a kernel
    K = torch.tensor([
        [0.0, 1.0],
        [2.0, 3.0],
    ])

    print(f'7.2.1 -> convoluted X, K: {corr2d(X, K)}')

    # =======================================
    # 7.2.3 - Object Edge Detection in Images
    # =======================================

    # Create a fake 6x8 "image"
    X = torch.ones(size=(6, 8))

    # The middle columns are "black" (0) and the outer columns are "white" (1)
    X[:, 2:6] = 0

    print(f'7.2.3 -> fake image X: {X}')

    # Create a 1x2 kernel
    K = torch.tensor([[1.0, -1.0]])

    # Perform cross-correlation
    # 1 indicates an edge form white to black
    # -1 indicates an edge from black to white
    # 0 indicates no edge
    Y = corr2d(X, K)

    print(f'7.2.3 -> convoluted image Y: {Y}')

    # Transpose X so that the black/white columns are now rows
    X = X.t()

    # Perform corss-correlation on the transposed matrix
    # All entries are now 0 because the kernel (K) only detects horizontal changes
    Y = corr2d(X, K)

    print(f'7.2.3 -> transposed iamge, convoluted to Y: {Y}')

    # =========================
    # 7.2.4 - Learning a Kernel
    # =========================

    c2d = nn.LazyConv2d(
        out_channels=1, # A single output channel
        kernel_size=(1, 2), # 1x2 kernel
        bias=False # ignore bias for simplicity
    )

    # The book omits this, but we need the tensors to be the original shape before transposition
    X = X.t()
    Y = corr2d(X, K)

    X = X.reshape(shape=(1, 1, 6, 8))
    Y = Y.reshape(shape=(1, 1, 6, 7))

    learning_rate = 3e-2
    epochs = 10

    for i in range(epochs):
        Y_hat = c2d(X)

        loss = (Y_hat - Y) ** 2

        c2d.zero_grad()

        loss.sum().backward()

        # Update the kernel
        c2d.weight.data[:] -= learning_rate * c2d.weight.grad

        if (i + 1) % 2 == 0:
            print(f'7.2.4 -> epoch {i + 1}, loss {loss.sum():.3f}')

    # Resulting values should be close to `[1, -1]`
    print(f'7.2.4 -> resulting kernel {c2d.weight.data.reshape((1, 2))}')

    # ===============
    # 7.3.1 - Padding
    # ===============

    c2d = nn.LazyConv2d(
        out_channels=1,
        kernel_size=3, # 3x3 kernel, meaning equal sized padding
        padding=1, # padding on ONE SIDE (so 1 * 2 = total padding for rows/columns)
    )

    X = torch.rand(size=(8,8))
    print(f'7.3.1 -> initial shape of X: {X.shape}')

    # shape should be the same size as the input matrix (8x8)
    print(f'7.3.1 -> shape of convolution with even padding: {comp_conv2d(c2d, X).shape}')

    c2d = nn.LazyConv2d(
        out_channels=1,
        kernel_size=(5,3), # 5x3 kernel
        padding=(2,1), # left/top padding is 2, right/bottom padding is 1
    )

    # shape should be the same size as the input matrix (8x8)
    print(f'7.3.1 -> shape of convolution with uneven padding: {comp_conv2d(c2d, X).shape}')

    # ==============
    # 7.3.2 - Stride
    # ==============

    X = torch.rand(size=(8,8))

    c2d = nn.LazyConv2d(
        out_channels=1,
        kernel_size=3, # 3x3 kernel
        padding=1, # padding of 2 total (1 on each side)
        stride=2 # stride of 2
    )

    print(f'7.3.2 -> shape of convolution with stride: {comp_conv2d(c2d, X).shape}')

    c2d = nn.LazyConv2d(
        out_channels=1,
        kernel_size=(3, 5), # 3x5 kernel
        padding=1, # padding of 2 total (1 on each side)
        stride=(3, 4), # stride rows by 3, columns by 4
    )

    print(f'7.3.2 -> shape of convolution with uneven stride: {comp_conv2d(c2d, X).shape}')
