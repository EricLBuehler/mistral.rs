import torch
x=torch.randn(1,1,3)
s=torch.full((1,9,1), 2)
print(x)
print(s)
print(x*s)