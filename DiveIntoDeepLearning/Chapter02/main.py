import torch

# 2.1

## 1.
### Run the code in this section.
### Change the conditional statement X == Y to X < Y or X > Y,
### and then see what kind of tensor you can get.

print('======== EXERCISE 2.1 (1) ========')

X = torch.rand(4)
Y = torch.rand(4)

print(X, '>', Y, '->', X > Y)

## 2.
### Replace the two tensors that operate by element
### in the broadcasting mechanism with other shapes,
### e.g., 3-dimensional tensors.
### Is the result the same as expected?

print('======== EXERCISE 2.1 (2) ========')

a = torch.arange(12).reshape((3, 2, 2))
print('a ->', a)

b = torch.arange(4).reshape((1, 2, 2))
print('b ->', b)

print('a + b ->', a + b)
