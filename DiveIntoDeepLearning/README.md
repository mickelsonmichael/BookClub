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

```shell
python -m venv DeepLearning

DeepLearning/scripts/activate

pip install -r requirements.cuda.txt
```

You can then execute any example similar to the following:

```shell
python Chapter02/main.py
```
