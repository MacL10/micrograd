import torch

print("PyTorch", torch.__version__)
print("Apple GPU (MPS) available:", torch.backends.mps.is_available())

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print("Gradient at x=3:", x.grad.item(), "(expected 8.0)")
