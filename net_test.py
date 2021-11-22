import math
import random
import pickle
from collections import OrderedDict, Counter
import torch
from torch import nn
import numpy as np

import glob


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

MAX_AA_LEN = 1000
BATCH_SIZE = 17

class_lookup = {
    "AMINOGLYCOSIDE": 0,
    "BETA-LACTAM": 1,
    "FOLATE-SYNTHESIS-INHABITOR": 2,
    "GLYCOPEPTIDE": 3,
    "MACROLIDE": 4,
    "MULTIDRUG": 5,
    "PHENICOL": 6,
    "QUINOLONE": 7,
    "TETRACYCLINE": 8,
    "TRIMETHOPRIM": 9,
}


class TooLong(Exception):
    pass


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class ClassifyNet(torch.nn.Module):
    def __init__(self, pretrained_module):
        super().__init__()
        self.pretrained = pretrained_module

        self.new_layers = nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(200 * 32, 10), torch.nn.Softmax()).to(
            DEVICE
        )

    def forward(self, inputs):
        x = self.pretrained(inputs)
        x = self.new_layers(x)
        return x


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

architecture = torch.nn.Sequential(
    torch.nn.Conv1d(1000, 32, kernel_size=3, padding=1),
    # 32 filters in and out, no max pooling so the shapes can be added
    ResNet(
        torch.nn.Sequential(
            torch.nn.Conv1d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Conv1d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
    ),
    # Another ResNet block, you could make more of them
    # Downsampling using maxpool and others could be done in between etc. etc.
    torch.nn.Linear(1280, 600),
    ResNet(
        torch.nn.Sequential(
            torch.nn.Conv1d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Conv1d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
    ),
    torch.nn.Linear(600, 200),
    # Pool all the 32 filters to 1, you may need to use `torch.squeeze after this layer`
    # torch.nn.AdaptiveAvgPool1d(1),
    # 32 10 classes
    # torch.nn.Linear(32, 10),
).to(DEVICE)


def pad(tensor):
    aa_len, _ = tensor.shape

    if aa_len > MAX_AA_LEN:
        raise TooLong

    aa_len_diff = float(MAX_AA_LEN - aa_len)
    pad_amount = (0, 0, math.ceil(aa_len_diff / 2), math.floor(aa_len_diff / 2))
    return nn.functional.pad(tensor, pad_amount, "constant", 0)


def get_pair_names(old_pairs=None, run_type="test"):
    if old_pairs is None:
        old_pairs = []
    else:
        old_pairs = ["|".join(pair.split("|")[0:-1]).split("/")[1] for pair, _ in old_pairs]
    pairs = []
    total_types = []
    with open("data/esm40.fa", "r") as f:
        for line in f:
            line = line.strip()
            # ten of each
            if line.startswith(">"):
                header = line[1:]
                res = header.split("|")[2]
                if Counter(total_types)[res] < BATCH_SIZE and header not in old_pairs:
                    # pairs.append([pair.split('/')[1] for pair in glob.glob(f"test_embeddings/{header}*")])
                    pairs.append(glob.glob(f"test_embeddings/{header}*"))
                    total_types.append(res)
    # pairs = random.choices(pairs, k=100)
    with open(f"pairs_{run_type}.pkl", "wb") as f:
        pickle.dump(pairs, f)


def contrastive_network(pairs):
    model = architecture
    # Create Tensors to hold input and outputs.

    # model = nn.Sequential(OrderedDict([
    #             ('layer1', torch.nn.Linear(1280, 600)),
    #             ('layer2', torch.nn.Linear(600, 300)),
    #             ('layer3', torch.nn.Linear(300,100))
    #             ])
    #         ).to(DEVICE)

    # 1280+2*0-1*(7-1)-1 + 1

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = nn.MSELoss(reduction="sum")

    learning_rate = 1e-6
    for epoch in range(1000):
        epoch_loss = 0
        batch1 = []  # torch.empty(0).to(DEVICE)
        batch2 = []  # torch.empty(0).to(DEVICE)
        for pair1, pair2 in pairs:
            try:
                x1 = torch.load(pair1)["representations"][34].to(DEVICE)
                x1 = pad(x1)
                x2 = torch.load(pair2)["representations"][34].to(DEVICE)
                x2 = pad(x2)
                # there is probably a more efficient way to do this
                batch1.append(x1)  # = torch.stack((batch1, x1), 0)
                batch2.append(x2)  # = torch.stack((batch2, x2), 0)
            except TooLong:
                continue
        batch1 = torch.stack(batch1, dim=0)
        batch2 = torch.stack(batch2, dim=0)

        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.

        # test_model = torch.nn.Conv1d(1000, 32, kernel_size=7).to(DEVICE)
        # test_pred = test_model(batch1).to(DEVICE)
        y_pred1 = model(batch1).to(DEVICE)
        y_pred2 = model(batch2).to(DEVICE)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred1, y_pred2)
        epoch_loss += loss
        # if epoch % 100 == 99:

        print(epoch, epoch_loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        # You can access the first layer of `model` like accessing the first item of a list

    # For linear layer, its parameters are stored as `weight` and `bias`.
    torch.save(model.state_dict(), "model1.pth")


def classifier(pairs):

    pre_trained_model = torch.load("model1.pth")

    # freeze all but the last layer
    for name, param in pre_trained_model.items():
        if name not in ["4.weight", "4.bias"]:
            param.requires_grad = False

    nn_arch = architecture

    nn_arch.load_state_dict(pre_trained_model)
    model = ClassifyNet(pretrained_module=nn_arch)
    loss_fn = nn.MSELoss(reduction="sum")
    learning_rate = 1e-6
    for epoch in range(200):
        epoch_loss = 0
        y_actual = torch.zeros(len(pairs), 10).to(DEVICE)
        batch1 = []  # torch.empty(0).to(DEVICE)
        for i, (pair1, _) in enumerate(pairs):
            y_actual[i][class_lookup[pair1.split("|")[-2]] - 1] = 1
            try:
                x1 = torch.load(pair1)["representations"][34].to(DEVICE)
                x1 = pad(x1)
                # there is probably a more efficient way to do this
                batch1.append(x1)  # = torch.stack((batch1, x1), 0)
            except TooLong:
                continue
        batch1 = torch.stack(batch1, dim=0)

        y_pred1 = model(batch1).to(DEVICE)
        loss = loss_fn(y_actual, y_pred1)
        epoch_loss += loss
        # if epoch % 100 == 99:

        print(epoch, epoch_loss.item())

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        # You can access the first layer of `model` like accessing the first item of a list

    # For linear layer, its parameters are stored as `weight` and `bias`.
    torch.save(model.state_dict(), "model2.pth")


def get_accuracy(test_pairs):
    nn_arch = architecture

    trained_model = torch.load("model1.pth")
    nn_arch.load_state_dict(trained_model)
    trained_model = ClassifyNet(pretrained_module=nn_arch)

    y_actual = np.zeros((len(test_pairs), 10))
    batch1 = []  # torch.empty(0).to(DEVICE)
    for i, (pair1, _) in enumerate(test_pairs):
        y_actual[i][class_lookup[pair1.split("|")[-2]] - 1] = 1
        try:
            x1 = torch.load(pair1)["representations"][34].to(DEVICE)
            x1 = pad(x1)
            # there is probably a more efficient way to do this
            batch1.append(x1)  # = torch.stack((batch1, x1), 0)
        except TooLong:
            continue
    batch1 = torch.stack(batch1, dim=0)

    y_pred = trained_model(batch1).to(DEVICE)
    preds = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
    actual = np.argmax(y_actual, axis=1)
    print(preds)
    print(actual)
    print(sum([a == b for a, b in zip(preds, actual)]))
    print(len(preds))


def main():
    get_pair_names(run_type="test")

    with open("pairs_test.pkl", "rb") as f:
        train_pairs = pickle.load(f)
    contrastive_network(train_pairs)
    classifier(train_pairs)

    get_pair_names(train_pairs, run_type="train")

    with open("pairs_test.pkl", "rb") as f:
        test_pairs = pickle.load(f)
    get_accuracy(test_pairs)


if __name__ == "__main__":
    main()
