import torch

# given two items to predict

# y is a tensor labels that are the correct answers (i.e., the first item is label 0)
y = torch.tensor([0, 2])

print('y\n\t', y)

# y_hat is a tensor of probabilities, one for each item
# (i.e., the first item is 10% likely to be a '0',
#   30% likely to be a '1', and 60% likely to be a '2')
y_hat = torch.tensor([
    [0.1, 0.3, 0.6],
    [0.3, 0.2, 0.5]
])

print('y_hat\n\t', y_hat)

# we can get the y_hat probability for the correct labels
# from y, get the two items we are interested in (the first and second, i.e., [0, 1])
# then, with those new indices (e.g., [0, 2]) we can retrieve the probabilities
# resulting in [0.1, 0.5]
print('y_hat[[0, 1], y]\n\t', y_hat[[0, 1], y])

# you can also write the above without manually specifying the label indices ([0, 1])
# by grabbing all the indices
# this is analogous to Enumerable.Range(0, y_hat.Length).ToList()
print('list(range(len(y_hat)))\n\t', list(range(len(y_hat))))

# replacing [0, 1] with this new index range will result in a similar effect
print('y_hat[list(range(len(y_hat))), y]\n\t', y_hat[list(range(len(y_hat))), y])

# this logic comes from Numpy
# https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
# and can be rethought of as combining the indeces to map into a matrix
# in the example above, we essentially have two arrays of the same size
# [0, 1] and [0, 2]
# the multi-integer indexing combines those to become unique coordinates into the matrix
# [0, 0] and [1, 2]
# you can also do this with different sized indexes, where broadcasting rules apply
print('y_hat[[0], y]\n\t', y_hat[[0], y])
