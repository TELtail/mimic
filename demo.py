import torch
import torch.nn as nn
import numpy as np

conv = nn.Conv1d(12, 8,kernel_size=3, stride=1)

a = torch.tensor(np.random.randn(12,3).astype(np.float32))
a = a.reshape(-1,12,3)
print(a)

b = conv(a)