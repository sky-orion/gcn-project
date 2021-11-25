import os
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp

from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_adj(file_path, adjtype):
    df = pd.read_csv(file_path, header=None)
    adj_mx = df.to_numpy().astype(np.float32)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data1 = {}
    seq_length_x, seq_length_y = 120, 120
    raw_data = np.load("/devdata/zhaohaoran/all_day_feature_single_daytime.npy", mmap_mode='r')
    raw_data = raw_data.astype(np.float)
    print(raw_data.shape, raw_data.dtype)
    # print(raw_data.shpae)
    data, scaler = normalize_dataset(raw_data)

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets)
    # per = np.random.permutation(x.shape[0])  # 打乱后的行号
    # x = x[per, :, :,:]  # 获取打乱后的训练数据
    # y = y[per, :, :,:]
    print('\n****************** Data Generator ******************')
    print(f'x shape: {x.shape}, y shape: {y.shape}')

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f'{cat} x: {_x.shape}, y: {_y.shape}')
    #
    data1['train_loader'] = DataLoader(x_train, y_train, batch_size)
    data1['val_loader'] = DataLoader(x_val, y_val, batch_size)
    data1['test_loader'] = DataLoader(x_test, y_test, batch_size)
    data1['scaler'] = scaler
    data1['y_test'] = y_test
    data1['x_test'] = x_test
    # data['train_loader'] = DataLoader(x_train, y_train, batch_size, shuffle=True)
    # data['val_loader'] = DataLoader(x_val, y_val, batch_size, shuffle=False)
    # data['test_loader'] = DataLoader(x_test, y_test, batch_size, shuffle=False)
    # data['scaler'] = scaler
    # for category in ['train', 'val', 'test']:
    #     cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
    #     data['x_' + category] = cat_data['x']
    #     data['y_' + category] = cat_data['y']
    # scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # # Data format
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    # data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    # data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    # data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    # data['scaler'] = scaler
    return data1
def normalize_dataset(data):
    main_scaler = StandardScaler(data[..., -1].mean(), (data[..., -1]).std())
    data[..., -1] = main_scaler.transform(data[..., -1])

    for idx in range(data.shape[-1] - 1):
        feature_scaler = StandardScaler(data[..., idx].mean(), data[..., idx].std())
        data[..., idx] = feature_scaler.transform(data[..., idx])
    return data, main_scaler
def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    # cuda = True if torch.cuda.is_available() else False
    TensorFloat =  torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets, scaler=None):
    """
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes, input_dim = data.shape

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = np.expand_dims(data[t + y_offsets, :, -1], axis=-1)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def metric_acc(pred, true):
    y_truth_np, y_pred_np = true, pred
    y_truth_cls, y_pred_cls = np.zeros(shape=true.size), np.zeros(shape=pred.size)
    high_idx, normal_idx, low_idx = [], [], []
    for i in range(y_truth_cls.size):
        if y_truth_np[i] < 2/3:
            y_truth_cls[i] = 0
            low_idx.append(i)
        elif 2/3 <= y_truth_np[i] < 4/3:
            y_truth_cls[i] = 1
            normal_idx.append(i)
        else:
            y_truth_cls[i] = 2
            high_idx.append(i)
    for i in range(y_pred_cls.size):
        if y_pred_np[i] < 2/3:
            y_pred_cls[i] = 0
        elif 2/3 <= y_pred_np[i] < 4/3:
            y_pred_cls[i] = 1
        else:
            y_pred_cls[i] = 2
    acc = sum(y_truth_cls==y_pred_cls)/(y_truth_cls.size)
    accH = sum(y_truth_cls[high_idx]==y_pred_cls[high_idx])/(y_truth_cls[high_idx].size)
    accN = sum(y_truth_cls[normal_idx]==y_pred_cls[normal_idx])/(y_truth_cls[normal_idx].size)
    accL = sum(y_truth_cls[low_idx]==y_pred_cls[low_idx])/(y_truth_cls[low_idx].size)
    return acc, accH, accN, accL

