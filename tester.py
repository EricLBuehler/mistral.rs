import numpy as np

def diff_mask_allclose(array1, array2, rtol=1e-05, atol=1e-08):
    return ~np.isclose(array1, array2, rtol=rtol, atol=atol)

name = "pixel_values_probe_X.npy"
py = np.load(name.replace("X","p"))
mistralrs = np.load("mistral.rs/"+name.replace("X","m"))

print(np.allclose(py, mistralrs))
print(py.shape, mistralrs.shape)
print((py - mistralrs)[:50,:50])

mask = (diff_mask_allclose(py, mistralrs))
print(py[mask]-mistralrs[mask])