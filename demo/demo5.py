import torch
import torch.nn as nn
import numpy as np





data_num = 20

data_x = torch.ones((data_num,100,3))
data_t = torch.ones((data_num,1))


net = nn.Linear(300,1)
loss_fn = nn.MSELoss()

print("-----------バッチサイズ:1")
batch_size = 1
for epoch in range(int(data_num/batch_size)):
    inputs = data_x[epoch*batch_size:(epoch+1)*batch_size].view(batch_size,300)
    outputs = net(inputs)
    loss = loss_fn(outputs,data_t[epoch*batch_size:(epoch+1)*batch_size])
    print(loss.item())


print("------------バッチサイズ:4")
batch_size = 4
for epoch in range(int(data_num/batch_size)):
    inputs = data_x[epoch*batch_size:(epoch+1)*batch_size].view(batch_size,300)
    outputs = net(inputs)
    loss = loss_fn(outputs,data_t[epoch*batch_size:(epoch+1)*batch_size])
    print(loss.item())