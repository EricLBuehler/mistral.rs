import numpy as np

p = np.load("generated_codes.npy")
m = np.load("generated_codes_m.npy")[:860, :]

print(np.allclose(p, m))
print(np.mean(np.abs(p - m)))

print(p)
print(m)
for line in range(860):
    print(line, p[0, line]-m[line])