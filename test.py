import torch

data = torch.randn((3,3))
print(data.max(1)[1])