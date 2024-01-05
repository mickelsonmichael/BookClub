import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

class FashionMNIST(d2l.DataModule): #@save
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()

        self.save_hyperparameters()

        transform = transforms.Compose([ 
            transforms.Resize(resize), # resize the images to the specified dimensions
            transforms.ToTensor()  # convert the image pizels to a tensor
        ])

        self.train = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=True, # use the test set
            transform=transform,
            download=True
        )

        self.val = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=False, # use the validation set
            transform=transform,
            download=True
        )

    # Retrieves the text versions of the labels for human reading
    def text_labels(self, indices):
        # each category is pre-assigned an index
        # some names have been modified from the original D2L names
        labels = [
            't-shirt',
            'pants',
            'pullover',
            'dress',
            'coat',
            'sandal',
            'shirt',
            'sneaker',
            'bag',
            'boot'
        ]

        # return all the labels for the provided indices
        return [ labels[int(i)] for i in indices ]

    # Get a dataloader for training purposes
    def get_dataloader(self, train):

        # get either the training set or the validation set
        data = self.train if train else self.val

        return torch.utils.data.DataLoader(
            data,
            self.batch_size,
            shuffle=train,
            num_workers=self.num_workers
        )

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch

        if not labels:
            labels = self.text_labels(y)

        d2l.show_images(
            # Removes all 1-dimensions
            # https://pytorch.org/docs/stable/generated/torch.squeeze.html
            X.squeeze(1),
            nrows,
            ncols,
            titles=labels
        )