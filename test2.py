import torch

x = torch.zeros(9, 23, 4096)
y = torch.zeros(9, 4096, 16)

result = x @ y