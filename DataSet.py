import torch as t
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import h5py
import numpy as np
from torch.nn.functional import one_hot


def load_data(train_data_dir):
    print('loading train data set')
    allData = h5py.File(train_data_dir, 'r')
    x_train = np.transpose(allData['train_x']).astype('float64')
    y_train = np.transpose(allData['train_y']).astype('float64')
    x_valid = np.transpose(allData['valid_x']).astype('float64')
    y_valid = np.transpose(allData['valid_y']).astype('float64')

    n = np.shape(x_train)[0]
    n_v = np.shape(x_valid)[0]
    # 预处理随机打乱
    index = np.arange(n)
    np.random.shuffle(index)
    x_train = x_train[index, :]
    y_train = y_train[index]
    y_train = y_train - 1

    index1 = np.arange(n_v)
    np.random.shuffle(index1)
    x_valid = x_valid[index1, :]
    y_valid = y_valid[index1]
    y_valid = y_valid - 1

    return x_train, y_train, x_valid, y_valid


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs),
    )


# def get_model(options):
# model = CustomizedNet(options.train_net_iter)
# return model, t.optim.SGD(model.parameters(), lr=options.learning_rate, momentum=options.momentum)


def loss_batch(model, loss_f, xb, yb, opt=None):
    loss = loss_f(model(xb), yb)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), len(xb)


def preprocess(x, y):
    return x.to('cuda'), y.to('cuda')


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


def ber(y_pred, y_target):
    return t.mean(t.abs(y_pred - y_target))


def load_train(train_data_dir, batch_size):
    x_train, y_train, x_valid, y_valid = map(t.tensor, (load_data(train_data_dir)))
    # torch one_hot 类型转换
    y_train = t.squeeze(y_train.to(t.int64))
    y_valid = t.squeeze(y_valid.to(t.int64))
    y_train = one_hot(y_train)
    y_valid = one_hot(y_valid)

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)
    print("load train data successfully")
    return train_dl, valid_dl


if __name__ == '__main__':
    load_train(train_data_dir='./data/train/train0.mat', batch_size=128)
