import numpy as np

p = np.load("prefill.npy")
m = np.load("prefill_m.npy")[None]

print(np.allclose(p, m))
print(np.mean(np.abs(p - m)))

print(p)
print(m)