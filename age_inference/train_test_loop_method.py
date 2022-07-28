import torch
from common_utils import log_start
import numpy as np


#regression
def train_regression_method(trainloader,net,optimizer,loss_fn,device,batch_size):
    logger = log_start()
    running_loss = 0
    size = len(trainloader.dataset)
    for i,(inputs,labels) in enumerate(trainloader):
        optimizer.zero_grad() #勾配初期化
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs,labels.float())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            logger.info(f" {i}/{int(size/batch_size)} loss:{loss}")
    
    running_loss /= (i+1)
    logger.info(f"train_loss:{running_loss}")

    return running_loss

def test_regression_method(testloader,net,loss_fn,device,print_result_flag):
    logger = log_start()
    running_loss = 0
    predicted_for_plot = []
    for i,(inputs,labels) in enumerate(testloader):
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        if print_result_flag:
            logger.info(outputs)
        loss = loss_fn(outputs,labels.float())
        outputs_np = outputs.to('cpu').detach().numpy().copy().flatten()[0] #プロット用に、ndarray → 一次元化
        labels_np = labels.to('cpu').detach().numpy().copy().flatten()[0] #プロット用に、ndarray → 一次元化
        predicted_for_plot.append([outputs_np,labels_np])
        running_loss += loss.item()
    
    running_loss /= (i+1)
    logger.info(f"test_loss:{running_loss}")

    return running_loss,np.array(predicted_for_plot)



#classification
def train_classification_method(trainloader,net,optimizer,loss_fn,device,batch_size):
    logger = log_start()
    running_loss = 0
    correct = 0
    size = len(trainloader.dataset)
    for i,(inputs,labels) in enumerate(trainloader):
        optimizer.zero_grad() #勾配初期化
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        labels = torch.flatten(labels)
        outputs = torch.flatten(outputs)
        loss = loss_fn(outputs,labels)
        running_loss += loss.item()
        predicted = torch.where(outputs<0.5,0,1)
        correct += (predicted==labels).sum().item()
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            logger.info(f" {i}/{int(size/batch_size)} loss:{loss}")
    
    correct /= size
    running_loss /= (i+1)
    logger.info(f"train_loss:{running_loss}")
    logger.info(f"train_accuracy:{correct*100:>5f}")

    return running_loss,correct

def test_classification_method(testloader,net,loss_fn,device,print_result_flag):
    logger = log_start()
    running_loss = 0
    correct = 0
    size = len(testloader.dataset)
    for i,(inputs,labels) in enumerate(testloader):
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        labels = torch.flatten(labels)
        outputs = torch.flatten(outputs)
        loss = loss_fn(outputs,labels)
        running_loss += loss.item()
        predicted = torch.where(outputs<0.5,0,1)
        correct += (predicted==labels).sum().item()
    correct /= size
    running_loss /= (i+1)
    logger.info(f"test_loss:{running_loss}")
    logger.info(f"test_accuracy:{correct*100:>5f}")

    return running_loss,correct