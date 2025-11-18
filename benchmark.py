import time
import torch
from custom_relu import custom_relu

def bench(fn, x, iters=200):
    torch.cuda.synchronize() if x.is_cuda else None
    start = time.time()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize() if x.is_cuda else None
    return (time.time() - start) / iters

def main():
    x = torch.randn(10000, requires_grad=True)

    # warmup
    custom_relu(x); torch.relu(x)

    t1 = bench(custom_relu, x)
    t2 = bench(torch.relu, x)

    print("custom relu:", t1 * 1e6, "us")
    print("torch.relu:", t2 * 1e6, "us")
    print("max error:", (custom_relu(x) - torch.relu(x)).abs().max().item())

if __name__ == "__main__":
    main()
