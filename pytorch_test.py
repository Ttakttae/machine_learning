import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w=torch.tensor([1.0],requires_grad = True)
#아무런 숫자가 들어가도 된다.

# model의 forward pass 구현 부분
def forward(x):
    return x * w

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# Before training
print("predict (before training)",  4, forward(4))

# Training loop
print("predict (before training)" , 4, forward(4))
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data): #1 
        l = loss(x_val, y_val) #2
        l.backward() #3
        print("grad: ", x_val, y_val,w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data #4
        # Manually zero the gradients after updating weights
        w.grad.data.zero_() #5

    print("progress:", epoch, l.data[0])
print("predict (after training)" , 4, forward(4))