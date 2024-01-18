from torch import device
from torch.cuda import device_count


def try_gpu(i=0):
    if device_count() >= i + 1:
        return device(f'cuda:{i}')
    return device('cpu')


def try_all_gpus():
    return [device(f'cuda:{i}') for i in range(device_count())]
