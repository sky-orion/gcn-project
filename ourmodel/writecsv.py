import numpy as np
import pandas as pd

arr = np.load("/devdata/zhaohaoran/all_day_feature_single_daytime.npy", mmap_mode='r')
towtite = arr[:, :, -1]
towtite = towtite.astype(np.float)
towtite = towtite.astype(np.int)
print(towtite)
print(towtite.shape, towtite.dtype)
print(np.max(np.max(towtite,axis=0)))
print(np.max(np.max(towtite,axis=1)))
print(np.min(np.min(towtite,axis=0)))
print(np.min(np.min(towtite,axis=1)))
##写入文件
pd_data = pd.DataFrame(towtite)
print(pd_data)
pd_data.to_csv('/devdata/zhaohaoran/airspace_data.csv',header=None,index=None)
