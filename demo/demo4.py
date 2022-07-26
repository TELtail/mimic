import torch
import torch.nn as nn
import numpy as np


data_x = np.random.randn(100,1000,3)
data_t = np.random.randint(0,2,(100,1))
data_x = torch.tensor(data_x).float()
data_t = torch.tensor(data_t)
print(data_x.shape,data_t.shape)
dataset = torch.utils.data.TensorDataset(data_x,data_t)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=25)

loss_fn = nn.CrossEntropyLoss()
fc = nn.Linear(3*1000,2)
softmax = nn.Softmax(dim=1)
correct = 0
size = len(dataloader.dataset)
for i, (inputs,labels) in enumerate(dataloader):
    outputs = softmax(fc(inputs.view(-1,3*1000)))
    labels = torch.flatten(labels)
    loss = loss_fn(outputs,labels.long())
    correct += (outputs.argmax(1)==labels).sum().item()
    print(labels.shape,outputs.shape)
print(correct / size)

for label,out in zip(labels,outputs):
    print(label,out)