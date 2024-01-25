# Chapter 6 - Builders' Guide

- Neural network modules are groups of multiple layers that can be grouped together and considered as a single layer
- `net(X)` is shorthand for `net.__call__(X)`
- Layers can contain _constant parameters_ which don't change with different inputs (see [FixedHiddenMLP.py](./FixedHiddenMLP.py))
- PyTorch tensors have `load` and `save` functions
  - These save the **parameters** and not the model itself
  - To load the parameters, you must recreate the module, then load the parameters from disk
- Torch can target a gpu using `torch.device('cuda')` or `torch.device('cuda:i')` where `i` is the GPU ID (e.g. `torch.device('cuda:0')`)
  - Can query the number of GPUs using `torch.cuda.device_count()`
- When printing or converting a Tensor, the data is copied to the main memory from the device