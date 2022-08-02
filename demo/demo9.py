import torch
import torch.nn as nn


num_axis = 1
hidden_dim = 64

conv = nn.Conv1d(num_axis,hidden_dim,kernel_size=3,stride=1)
print("weight",conv.weight.shape)

batch_size = 8
L_in = 100

inputs = torch.randn([batch_size,num_axis,L_in])
print("inputs",inputs.shape)
y = conv(inputs)
#print(y)
print("outputs",y.shape)