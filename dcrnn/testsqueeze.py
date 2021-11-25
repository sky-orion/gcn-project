import numpy as np

a = np.zeros((1,5,6))
a = np.array(a)
a_sque = np.squeeze(a)

print(a.shape)
print(a_sque.shape)
