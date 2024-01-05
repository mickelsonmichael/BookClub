import torch
from d2l import torch as d2l

class Classifier(d2l.Module):
    def validation_setp(self, batch):
        Y_hat = self(*batch[:-1])

        self.plot(
            'loss',
            self.loss(Y_hat, batch[-1]),
            train=False
        )

        self.plot(
            'acc',
            self.accuracy(Y_hat, batch[-1]),
            train=False
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def accuracy(self, Y_hat, Y, averaged=True):
        # get the last column, since it is assumed to be the predictions
        # https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))

        # find the index of the largest probability in the row
        # https://pytorch.org/docs/stable/generated/torch.argmax.html#torch.argmax
        predictions = Y_hat.argmax(axis=1).type(Y.dtype)

        # a list of predictions (1 for correct, 0 for incorrect)
        comparisons = (predictions == Y.reshape(-1)).type(torch.float32)

        return comparisons.mean() if averaged else comparisons
