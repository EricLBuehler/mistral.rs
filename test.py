import torch

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2,3)
        self.b = torch.nn.Linear(2,3)

a=A()
print(list(a.named_parameters()))