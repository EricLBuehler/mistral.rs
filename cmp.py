import numpy as np

a = np.load("test.npy")
b = np.load("out.npy")
print(np.max(a-b))