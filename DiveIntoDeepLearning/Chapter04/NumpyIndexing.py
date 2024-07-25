import torch
from FashionMNIST import FashionMNIST

# While reading Chapter 4, we encountered some odd Python syntax that we weren't sure how to interpret.
# <https://d2l.ai/chapter_linear-classification/softmax-regression-scratch.html#the-cross-entropy-loss>

# Assume you have 2 items to make predictions for.
# In the FashionMNIST dataset, this would be two pictures of clothing to categorize.
#
# NOTE: This code is demonstrative and is omitted because it is not used and is a slow process
# dataset = FashionMNIST(batch_size=2)
# minibatch_size2 = next(iter(dataset.get_dataloader()))

# Let 'y' be a tensor containing the answer to these pictures, known ahead of time to be true
# In the FashionMNIST dataset, this would mean the first item is a t-shirt and the second is a pullover.
y = torch.tensor([0, 2])

print('y\n\t', y)                                                   # [0, 2]

# Let 'y_hat' be a tensor of probabilities with an entry for each item and an entry for each category.
# For brevity, only three categories are displayed.
# 
# In other words, given the data below:
# the first item has a 30% probability of being type '1' (pants)
# and a 60% probability of being type '2' (pullover)
y_hat = torch.tensor([
    [0.1, 0.3, 0.6],
    [0.3, 0.2, 0.5]
])

print('y_hat\n\t', y_hat)                                           # [[ 0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]

# In order to perform the Cross-Entropy Loss calculation,
# we need to get the probability assigned to the correct answer.
# 
# In the example, we would need to get the probability that the first item is type '0' (t-shirt)
# and the probability that the second item is type '2' (pullover).
#
# We can do that using the tensor[int[], int[]] syntax.
# 
# This syntax will create coordinates into the 'y_hat' matrix by doing a sort of dot product
# with the two arguments.
#
# The 0 from the first list is combined with the first value of 'y' to create [0, 0].
# The 1 from the first list is combined with the second value of 'y' to create [1, 0.5]
#
# The indexer then fetches the values at those coordinates and creates a list

print('y_hat[[0, 1], y]\n\t', y_hat[[0, 1], y])                     # [0.1, 0.5]

# Because [0, 1] is essentially equivalent to the indexes of the items of 'y',
# can generate those indexes automatically using the `range` function.
# 
# The following is equivalent to doing `Enumerable.Range(0, y_hat.Length).ToList()` in C#

print('list(range(len(y_hat)))\n\t', list(range(len(y_hat))))       # [0, 1]

# We can then replace the manually written [0, 1] in the above function
# with the new method of generating the indexes.

print('y_hat[list(range(len(y_hat))), y]\n\t', y_hat[list(range(len(y_hat))), y]) # [0, 1]

# This logic comes from Numpy, and PyTorch has brought it over to their library.
# <https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing>
# 
# You can also do this operation with different sized indexes and allow broadcasting rules to take effect:

print('y_hat[[0], y]\n\t', y_hat[[0], y])                           # [ [0, 0], [0, 2] ]
