"""
Simple test of gradient flow through frozen layers.
"""

import torch
import torch.nn as nn

# Test 1: Trainable -> Frozen -> Loss
print("=" * 80)
print("TEST 1: Trainable -> Frozen -> Loss")
print("=" * 80)

trainable = nn.Linear(10, 10, bias=False)
frozen = nn.Linear(10, 10, bias=False)

# Freeze the second layer
for param in frozen.parameters():
    param.requires_grad = False

x = torch.randn(1, 10)
h1 = trainable(x)
h2 = frozen(h1)
loss = h2.sum()

print(f"Trainable param requires_grad: {list(trainable.parameters())[0].requires_grad}")
print(f"Frozen param requires_grad: {list(frozen.parameters())[0].requires_grad}")
print(f"Loss requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("Backward succeeded")
    grad = list(trainable.parameters())[0].grad
    print(f"Trainable has gradient: {grad is not None}")
    if grad is not None:
        print(f"Gradient norm: {grad.norm().item()}")
except Exception as e:
    print(f"Backward failed: {e}")

# Test 2: Embedding -> Frozen -> Loss
print("\n" + "=" * 80)
print("TEST 2: Embedding -> Frozen -> Loss")
print("=" * 80)

emb = nn.Embedding(100, 10)
frozen2 = nn.Linear(10, 10, bias=False)

for param in frozen2.parameters():
    param.requires_grad = False

token_ids = torch.tensor([[5, 10, 15]])
h1 = emb(token_ids)
h2 = frozen2(h1)
loss = h2.sum()

print(f"Embedding param requires_grad: {list(emb.parameters())[0].requires_grad}")
print(f"Frozen param requires_grad: {list(frozen2.parameters())[0].requires_grad}")
print(f"Loss requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("Backward succeeded")
    grad = list(emb.parameters())[0].grad
    print(f"Embedding has gradient: {grad is not None}")
    if grad is not None:
        print(f"Gradient norm: {grad.norm().item()}")
except Exception as e:
    print(f"Backward failed: {e}")
