import time
from FashionMNIST import FashionMNIST

# https://stackoverflow.com/questions/24374288/where-to-put-freeze-support-in-a-python-script
# Must be wrapped in this 'if' statement to support multiple workers
if __name__ == "__main__":
    print('Downloading dataset')

    data = FashionMNIST(resize=(32, 32)) # download the fashion dataset as 32x32 images

    print('dataset length', len(data.train))
    print('validation set length', len(data.val))

    # Prints out '1 x 32 x 32' since the images are stored as a tensor
    # 1 is the number of color channels, 32 are the height and width
    print('data shape:', data.train[0][0].shape)

    dataIter = iter(data.train_dataloader())

    X, y = next(dataIter) # get the first minibatch

    print('X:', X.shape, X.dtype)
    print('y:', y.shape, y.dtype)

    # time the dataset to see how long it takes to read
    # NOTE: this is not the time it takes to train
    # This will be a large amount, but training will take longer
    # so the process is not constrained by I/O speeds
    tic = time.time()
    for X,y in data.train_dataloader():
        continue
    print(f'{time.time() - tic:.2f} seconds to loop through minibatches')

    # visualize the first minibatch
    data.visualize((X, y))