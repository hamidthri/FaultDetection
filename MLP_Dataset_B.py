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
        train_loss, train_acc = train(d_train, model, optimizer, n_in)
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

        print('Loss train: %f  | Acc train: %f === Loss test: %f  | Acc test: %f' % (train_loss, train_acc, test_loss, test_acc))
    plt.plot(loss_train,'b', label='loss train')
    plt.plot(loss_test,'r', label='loss test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train(d_train, model, optimizer,n_in):
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
        # total_loss = total_loss/len
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





if __name__ == '__main__':

    ### parameters
    n_in = 2000
    #model params
    in_dim = n_in
    hid_dim1 = 600
    hid_dim2 = 200
    hid_dim3 = 100
    out_dim = 10

    #train params
    lr = 1e-3
    reg_coef = 1e-5
    momentum = 0.9

    b_size = 64
    test_size = 0.25

    n_epochs = 101

    dtype = torch.float32
    #######
##############################################################################################

    raw_data = np.load('data_all.npy', allow_pickle=True)
    raw_data = raw_data.item()

    datasets = {'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['1', '2', '3']}
    data = {}
    n_sample = 200
    for k1 in datasets.keys():
        data[k1] = []
        for k2 in raw_data.keys():
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
    dataset_D = data['D']
    data_A = dataset_A.reshape(24000, 2000)
    data_B = dataset_B.reshape(24000, 2000)
    data_C = dataset_C.reshape(24000, 2000)
    data_D = dataset_D.reshape(24000, 6000)
    x = data_B
##########################################################################################
    # data split
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


    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hid_dim1, bias=True),
        # torch.nn.BatchNorm1d(hid_dim),
        torch.nn.Tanh(),
        torch.nn.Dropout(.5),
        torch.nn.Linear(hid_dim1, hid_dim2,bias=True),
        torch.nn.Tanh(),
        # torch.nn.Dropout(.5),
        torch.nn.Linear(hid_dim2, hid_dim3,bias=True),
        torch.nn.Tanh(),
        torch.nn.Dropout(.5),
        torch.nn.Linear(hid_dim3, out_dim,bias=True),
        torch.nn.Softmax()
    )




    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_coef)
    # optimizer = optim.Rprop(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=reg_coef, momentum=0.9)

    loss_func = torch.nn.BCELoss(reduction='mean')

    training(d_train, d_test, model, n_epochs, optimizer, n_in)





