import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sklearn
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=in_features, out_features=out_features)
        # decoder
        self.dec1 = nn.Linear(in_features=out_features, out_features=in_features)
    def forward(self, x):
        h1 = F.sigmoid(self.enc1(x))
        h2 = F.sigmoid(self.dec1(h1))
        return h2


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(F.sigmoid(rho_hat), 1)  # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

# define the sparse loss function
def sparse_loss(rho, data):
    values = data
    loss = 0
    for i in range(len(model_children)):
        values = model_children[i](values)
        loss += kl_divergence(rho, values)
    return loss


def forward(x):
    h1 = F.sigmoid(x@(param[0]).T.detach()+(param[1]).detach())
    return h1
class MyModule(nn.Module):
    def __init__(self, wenc1,wenc2,wenc3):
        super(MyModule, self).__init__()

        self.linear1 = nn.Linear(wenc1.shape[1], wenc1.shape[0], bias=True)
        with torch.no_grad():
            self.linear1.weight.copy_(wenc1)
            self.linear1.bias.copy_(benc1)
        self.linear2 = nn.Linear(wenc2.shape[1], wenc2.shape[0], bias=True)
        with torch.no_grad():
            self.linear2.weight.copy_(wenc2)
            self.linear2.bias.copy_(benc2)
        self.linear3 = nn.Linear(wenc3.shape[1], wenc3.shape[0], bias=True)
        with torch.no_grad():
            self.linear3.weight.copy_(wenc3)
            self.linear3.bias.copy_(benc3)
        self.linear4 = nn.Linear(out_features, 10, bias=True)

    def forward(self, x):
        o1 = F.sigmoid(self.linear1(x))
        # nn.BatchNorm1d(wenc1.shape[0])
        # nn.Dropout(0.5)
        o2 = F.sigmoid(self.linear2(o1))
        nn.Dropout(0.5)
        # nn.BatchNorm1d(wenc2.shape[0])
        o3 = F.sigmoid(self.linear3(o2))
        o4 = F.softmax(self.linear4(o3))
        # nn.Dropout(0.5)
        # nn.BatchNorm1d(wenc3.shape[0])
        return o4

def train(net, d_train, NUM_EPOCHS):
    print('Training')
    net.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(d_train), total=int(len(data_train) / d_train.batch_size)):
        counter += 1
        data
        optimizer.zero_grad()
        outputs = net(data)
        mse_loss = criterion(outputs, data)
        if ADD_SPARSITY == 'yes':
            sparsity = sparse_loss(RHO, data)
            # add the sparsity penalty
            loss = mse_loss + BETA * sparsity
        else:
            loss = mse_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"Train Loss: {epoch_loss:.3f}")

    return epoch_loss

####################################
def training(d_train, d_test, model, n_epochs, optimizer,n_in):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.2, verbose=True)
    x_tr = d_train.dataset.data[:, :n_in]
    y_tr = d_train.dataset.data[:, n_in:]

    x_te = d_test.dataset.data[:, :n_in]
    y_te = d_test.dataset.data[:, n_in:]
    loss_train = []
    loss_test = []
    for epoch in range(n_epochs):
        print('epoch %d/%d' % (epoch + 1, n_epochs))
        train_loss, train_acc = train1(d_train, model, optimizer, n_in)
        train_loss = train_loss/len_train
        loss_train.append(train_loss)
        pred = torch.argmax(model(x_tr), 1)
        trgt = torch.argmax(y_tr, 1)
        train_cnfm = confusion_matrix(trgt.numpy(), pred.detach().numpy())

        test_loss, test_acc = test(d_test, model, n_in)
        test_loss = test_loss / len_test
        loss_test.append(test_loss)
        pred = torch.argmax(model(x_te), 1)
        trgt = torch.argmax(y_te, 1)
        test_cnfm = confusion_matrix(trgt.numpy(), pred.detach().numpy())


        if epoch%50 == 0:
            plt.imshow(train_cnfm, cmap='plasma', interpolation='nearest')
            for i in range(10):
                for j in range(10):
                    text = plt.text(j, i, train_cnfm[i, j],
                                   ha="center", va="center", fontsize='large', color="w")
            plt.colorbar()
            plt.title('Train confusion matrix')
            plt.show()

            plt.imshow(test_cnfm, cmap='plasma', interpolation='gaussian')
            for i in range(10):
                for j in range(10):
                    text = plt.text(j, i, test_cnfm[i, j],
                                    ha="center", va="center", fontsize='large', color="w")
            plt.colorbar()
            plt.title('Test confusion matrix')
            ()
            plt.show()

        scheduler.step(test_loss)

        print('Loss train: %f  | Acc train: %f === Loss test: %f  | Acc test: %f' % (train_loss,train_acc, test_loss, test_acc))
    plt.plot(loss_train,'b', label='loss train')
    plt.plot(loss_test,'r', label='loss test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def train1(d_train, model, optimizer,n_in):
    model.train()
    total_loss = 0
    accu = 0.
    total = 0.
    for data in d_train:
        x = data[:, :n_in]
        y = data[:, n_in:]

        optimizer.zero_grad()
        y_hat = model(x)

        loss = loss_func(y_hat, y)
        loss.backward()
        optimizer.step()

        accu += (y_hat.max(dim=1)[-1] == y.max(dim=1)[-1]).sum().item()
        total += y.shape[0]

        total_loss += loss.item()
    return total_loss, 100*accu/total


def test(d_test, model,n_in):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        accu = 0.
        total = 0.
        for data in d_test:
            x = data[:, :n_in]
            y = data[:, n_in:]

            y_hat = model(x)
            accu += (y_hat.max(dim=1)[-1] == y.max(dim=1)[-1]).sum().item()
            total += y.shape[0]

            loss = loss_func(y_hat, y)
            loss_total +=loss.item()
    return loss_total, 100*accu/total


###############################

if __name__ == '__main__':
##########################################################################################
    # data split
    raw_data = np.load('data_all.npy', allow_pickle=True)
    raw_data = raw_data.item()
    datasets = {'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['1', '2', '3']}
    data = {}
    n_sample = 200
    for k1 in datasets.keys():
        data[k1] = []
        for k2 in raw_data.keys():
            # notnormalized = []
            for load in datasets[k1]:
                rand_sample = np.random.choice(raw_data[k2][load].shape[0] - 24000, n_sample, replace=False)
                for i in rand_sample:
                    data[k1].append(raw_data[k2][load][i: i + 24000])
        data[k1] = np.concatenate(data[k1])

    y = np.hstack([0 * np.ones(2400), 1 * np.ones(2400),
                   2 * np.ones(2400), 3 * np.ones(2400),
                   4 * np.ones(2400), 5 * np.ones(2400),
                   6 * np.ones(2400), 7 * np.ones(2400),
                   8 * np.ones(2400), 9 * np.ones(2400), ]
                  )

    dataset_A = data['A']
    mean=dataset_A.mean()
    std=dataset_A.std()
    dataset_A = (dataset_A-mean)/std
    dataset_B = data['B']
    mean = dataset_B.mean()
    std = dataset_B.std()
    dataset_B = (dataset_B - mean) / std
    dataset_C = data['C']
    mean = dataset_C.mean()
    std = dataset_C.std()
    dataset_C = (dataset_C - mean) / std
    dataset_D = data['D']

    data_A = dataset_A.reshape(24000, 2000)
    data_B = dataset_B.reshape(24000, 2000)
    data_C = dataset_C.reshape(24000, 2000)
    data_D = dataset_D.reshape(24000, 6000)

# constructing argument parsers
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', type=int, default=10,
                    help='number of epochs to train our network for')
    ap.add_argument('-l', '--reg_param', type=float, default=0.2,
                    help='regularization parameter `lambda`')
    ap.add_argument('-sc', '--add_sparse', type=str, default='yes',
                    help='whether to add sparsity contraint or not')
    args = vars(ap.parse_args())
    EPOCHS = args['epochs']
    BETA = args['reg_param']
    ADD_SPARSITY = args['add_sparse']
    RHO = 0.15
    ### parameters
    n_in = 2000
    #model params
    in_dim = n_in
    hid_dim1 = 600
    hid_dim2 = 200
    hid_dim3 = 50
    out_dim = 10

    #train params
    lr = 1e-3
    reg_coef = 1e-5
    momentum = 0.9

    b_size = 64
    test_size = 0.25

    n_epochs = 101

    dtype = torch.float32
    x = data_C
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = test_size, random_state=42)
    #convert data to pytorch
    x_train = torch.from_numpy(x_train).to(dtype)
    y_train = torch.from_numpy(y_train).view(-1,1).to(dtype)
    y_train = torch.nn.functional.one_hot(y_train.to(torch.long)).squeeze().to(dtype)

    data_train = torch.cat([x_train,y_train], dim=1)
    d_train = DataLoader(data_train, batch_size=b_size, shuffle=True, drop_last=True)
    len_train = len(d_train)
    x_test = torch.from_numpy(x_test).to(dtype)
    y_test = torch.from_numpy(y_test).view(-1, 1).to(dtype)
    y_test = torch.nn.functional.one_hot(y_test.to(torch.long)).squeeze().to(dtype)

    data_test = torch.cat([x_test, y_test], dim=1)
    d_test = DataLoader(data_test, batch_size=b_size, shuffle=True, drop_last=True)
    len_test = len(d_test)

    # utility functions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Autoencoder
    in_features = 2000
    out_features = 600
    net = Autoencoder()
    print(net)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    model_children = list(net.children())


    print(device)
    # load the neural network onto the device
    net.to(device)

    # train the network
    data_train0 = x_train
    d_train = DataLoader(data_train0, batch_size=b_size, shuffle=True, drop_last=True)
    train_loss = train(net, d_train, n_epochs)
    param = list(net.parameters())
    wenc1 = (param[0]).detach()
    benc1 = (param[1]).detach()
    h1 = forward(x_train)
    data_train = h1
    d_train = DataLoader(data_train, batch_size=b_size, shuffle=True, drop_last=True)
    in_features = 600
    out_features = 200
    net = Autoencoder()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    model_children = list(net.children())
    train_loss = train(net, d_train, n_epochs)
    param = list(net.parameters())
    wenc2 = (param[0]).detach()
    benc2 = (param[1]).detach()
    h2 = forward(h1)
    data_train = h2
    d_train = DataLoader(data_train, batch_size=b_size, shuffle=True, drop_last=True)
    in_features = 200
    out_features = 100
    net = Autoencoder()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    model_children = list(net.children())
    train_loss = train(net, d_train, n_epochs)

    param = list(net.parameters())
    wenc3 = (param[0]).detach()
    benc3 = (param[1]).detach()
    h3 = forward(h2)
    # data_train = h3
    # d_train = DataLoader(data_train, batch_size=b_size, shuffle=True, drop_last=True)
    # in_features = 100
    # out_features = 10
    # net = Autoencoder()
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    # model_children = list(net.children())
    # train_loss = train(net, d_train, n_epochs)

    param = list(net.parameters())
    wenc4 = (param[0]).detach()
    benc4 = (param[1]).detach()



    #training
    data_train = torch.cat([x_train,y_train], dim=1)
    d_train = DataLoader(data_train, batch_size=b_size, shuffle=True, drop_last=True)
    model = MyModule(wenc1, wenc2, wenc3)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_coef)
    # optimizer = optim.Rprop(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=reg_coef, momentum=0.9)
    loss_func = torch.nn.BCELoss(reduction='mean')
    n_epochs = 101
    training(d_train, d_test, model, n_epochs, optimizer, n_in)