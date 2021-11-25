import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'airspace':
        df = pd.read_csv('/devdata/zhaohaoran/airspace_data.csv', header=None)
        data = df.values

        # data_path = os.path.join('/devdata/zhaohaoran/all_day_feature_single_daytime.npy')
        # raw_data = np.load(data_path, mmap_mode='r')
        # data = raw_data.astype(np.float)
    elif dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
