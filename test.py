import numpy as np

p = np.load("encoder_out.npy")
m = np.load("encoder_out_m.npy")

print(np.allclose(p,m))
print(np.mean(np.abs(p-m)))

print(p)
print(m)
