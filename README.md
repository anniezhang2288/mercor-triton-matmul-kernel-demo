# Custom PyTorch Operator

## Features
- Custom operator (`CustomReLU`)
- Manual forward pass
- Manual backward pass (gradient kernel)

## How to run:
1. Clone the repository
2. Install requirements and run benchmark.py like below
```bash
pip install -r requirements.t
python benchmark.py
```
## Expected Output: 
```bash
custom relu: 8.139610290527344 us
torch.relu: 3.944635391235352 us
max error: 0.0
```
