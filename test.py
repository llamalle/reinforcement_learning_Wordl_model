import torch
import numpy as np

a = np.random.rand(3,3)
print(a)
with open('test.npy', 'wb') as f:
    np.save(f, a)

print(np.zeros([3,32]))