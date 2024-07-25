# Chapter 5 - Multilayer Perceptrons

## Terms

- _Affine function_ - the result of a linear regression; a function in which all the variables are given weights, and an additional bias is added
    - An affine function of an affine function results in another affine function
- _Monotonicity_ - When a given feature either always increase or decrease the model's output - not both
- _Multilayer perceptron (MLP)_ - A deep network where the layers are fully connected
- _Activation function_ - Used to modify hidden layers so they are no longer affine functions; determine whether a neuron should be activate or not
    - For a neuron, calculate the weighted sum and add additional bias to it
    - e.g., ReLU (rectified linear unit)
- _Rectified linear unit (ReLU) function_ - An activator function that takes `max(0, x)` to discard all negative elements
    - Comes in a parameterized version (_pReLU_) that allows some information from negative arguments to come through
- _Sigmoid function_ - Transforms inputs so that they are between 0 and 1.
    - Also called the _squashing function_
- _Hyperbolic tangent (tanh) function_ - Similar to sigmoid, transforms inputs so they are between -1 and 1.