import torch


def corr2d(X, K):
    """
    Calculates the 2-dimension cross-correlation

    NOTE: I have modified the original code to be more expressive,
        but it is not necessarily optimzed

    :param X: The tensor containing pixel values, but not color channels (e.g., 2-dimension)
    :param K: The kernel
    """

    K_height, K_width = K.shape
    X_height, X_width = X.shape

    Y_height = X_height - K_height + 1
    Y_width = X_width - K_width + 1

    Y = torch.zeros(size=(Y_height, Y_width))

    for i in range(Y_height):
        for j in range(Y_width):
            # Get the subset of X covered by K
            X_subset = X[i:i + K_height, j:j + K_width]

            Y[i, j] = (X_subset * K).sum()

    return Y
