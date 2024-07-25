# Dive Into Deep Learning

## Developing

To execute the chapter exercises, you must have Python 3 already installed.
You can then create a virtual environment (or not, it's up to you), activate it,
and then install the requirements.

There are two requirements files, [requirements.cpu.txt](./requirements.cpu.txt)
and [requirements.cuda.txt](./requirements.cuda.txt).
The former installs libraries for running PyTorch on your CPU,
while the latter installs libraries for running PyTorch on an Nvidia GPU.
If you do not have an NVidia GPU, select the CPU version of the requirements.

> **Warning**: [`torch` is not available for Python 3.11 yet](https://github.com/pytorch/pytorch/issues/110436).
> You will need to use a lower version of Python (i.e., <=3.11).
> You can have multiple python versions installed on your machine
> and create the virtual environment using the compatible version
> by replacing `python -m venv` below with `python3.11 -m venv` (or your preferred version).

```shell
python -m venv DeepLearning

DeepLearning/scripts/activate

pip install -r requirements.cuda.txt
```

You can then execute any example similar to the following:

```shell
python Chapter02/main.py
```
