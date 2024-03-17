import torch
import time

def bmm(input, a):
    # Perform batch matrix multiplication
    print("inp",input.shape)
    print("a",a.shape)
    result_reshaped = input @ a #torch.matmul(input_reshaped, a_reshaped)

    # Reshape the result back to the original shape
    result = result_reshaped.reshape(input.size(0), input.size(1), a.size(-1))

    return result


first_time = 0.
snd_time = 0.

in_size = 300
out_size = 200
bs = 1
n_adapters = 8
input = torch.randn(n_adapters,bs,5,in_size)
for n in range(50):
    a1 = torch.randn(out_size,in_size)
    a2 = torch.randn(out_size,in_size)
    a3 = torch.randn(out_size,in_size)
    a4 = torch.randn(out_size,in_size)
    a5 = torch.randn(out_size,in_size)
    a6 = torch.randn(out_size,in_size)
    a7 = torch.randn(out_size,in_size)
    a8 = torch.randn(out_size,in_size)

    l = [a1.unsqueeze(0), a2.unsqueeze(0), a3.unsqueeze(0), a4.unsqueeze(0),
         a5.unsqueeze(0), a6.unsqueeze(0), a7.unsqueeze(0), a8.unsqueeze(0)]
    a = torch.cat(l, dim=0)
    a = a.permute(0,2,1)

    st = time.time()

    res = torch.zeros(bs,5,out_size)
    res += input[0] @ a1.T
    res += input[1] @ a2.T
    res += input[2] @ a3.T
    res += input[3] @ a4.T
    res += input[4] @ a5.T
    res += input[5] @ a6.T
    res += input[6] @ a7.T
    res += input[7] @ a8.T
    
    end = time.time()
    first_time += end-st

    st = time.time()

    res2 = bmm(input.reshape(n_adapters, bs*5, in_size), a)
    res2 = res2.reshape(n_adapters, bs, 5, out_size).sum(0)

    end = time.time()
    snd_time += end-st

    print((res - res2).mean())
    #assert torch.allclose(res, res2), (res, res2, res.shape, res2.shape, (res - res2).mean())

print(first_time / 50 * 1000)
print(snd_time / 50 * 1000)