import numpy as np

p = np.load("encoder_out.npy")
m = np.load("encoder_out_m.npy")

print(np.allclose(p,m))
print(np.mean(np.abs(p-m)))

print(p[1])
print("\n\n")
print(m[0])
print(p.shape,m.shape)

# p = np.load("x_norm.npy")
# m = np.load("x_norm_m.npy")

# print(np.allclose(p,m))
# print(np.mean(np.abs(p-m)))

# p = np.load("ca_out.npy")
# m = np.load("ca_out_m.npy")

# print(np.allclose(p,m))
# print(np.mean(np.abs(p-m)))

# p = np.load("mlp_out.npy")
# m = np.load("mlp_out_m.npy")

# print(np.allclose(p,m))
# print(np.mean(np.abs(p-m)))