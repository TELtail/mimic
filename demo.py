import torch
import torch.nn as nn
import numpy as np



a = torch.tensor(np.random.randn(2,12,3).astype(np.float32))
print(a)

conv = nn.Conv1d(12, 32, kernel_size=3, stride=1)
b = conv(a)
print(b.shape)

norm = nn.BatchNorm1d(32)
c = norm(b)
print(c.shape)


pool = nn.MaxPool1d(kernel_size=4,stride=2)
c = c.view(-1,32)
d = pool(c)
print(d.shape)

