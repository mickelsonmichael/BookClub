from d2l import torch as d2l
from MLPScratch import MLPScratch
from MLP import MLP

def train_from_scratch():
    model = MLPScratch(
        num_inputs=784, # flatten the images into a single vector of the pixels
        num_outputs=10, # we want one of the ten categories
        num_hiddens=256,# hidden layer is 256 neurons
        lr=0.1          # learning rate
    )

    # get the dataset in batches of 256 images
    data = d2l.FashionMNIST(batch_size=256)

    # create the trainer
    trainer = d2l.Trainer(max_epochs=10)

    # train the model on the dataset
    trainer.fit(model, data)

def train_with_library():
    model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)

     # get the dataset in batches of 256 images
    data = d2l.FashionMNIST(batch_size=256)

    # create the trainer
    trainer = d2l.Trainer(max_epochs=10)

    trainer.fit(model, data)

if __name__ == "__main__":
    # train_from_scratch()
    train_with_library()
