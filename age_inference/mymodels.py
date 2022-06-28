import torch.nn as nn
import torch.nn.functional as F

class Lstm_regression_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers,out_dim):
        super(Lstm_regression_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_axis,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,out_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        _,x = self.lstm(x)
        x = x[0][-1].view(-1, self.hidden_dim)
        x = self.fc(x)
        return x
class Lstm_classification_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers,out_dim):
        super(Lstm_classification_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_axis,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,out_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        _,x = self.lstm(x)
        x = x[0][-1].view(-1, self.hidden_dim)
        x = self.fc(x)
        return x

class Conv1D_regression_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,sig_length,out_dim):
        super(Conv1D_regression_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(sig_length,hidden_dim,kernel_size=num_axis,stride=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim,out_dim)
        self.dropout = nn.Dropout(0.25)

        
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = x.view(-1,self.hidden_dim)
        x = self.fc(x)
        return x

class Conv1D_classification_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,sig_length,out_dim):
        super(Conv1D_classification_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(sig_length,hidden_dim,kernel_size=num_axis,stride=1)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_dim,out_dim)
        self.dropout = nn.Dropout(0.25)

        
    def forward(self,x):
        x = self.sigmoid(self.conv1(x))
        x = x.view(-1,self.hidden_dim)
        x = self.fc(x)
        return x

class Linear_regression_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers,sig_length,out_dim):
        super(Linear_regression_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_axis = num_axis
        self.sig_length = sig_length
        self.fc_start = nn.Linear(num_axis*sig_length,hidden_dim) 
        self.fc_bet = nn.Linear(hidden_dim,hidden_dim)
        self.fc_end = nn.Linear(hidden_dim,out_dim)

        self.dropout = nn.Dropout(0.25)
    
    def forward(self,x):
        x = x.view(-1,self.num_axis*self.sig_length)
        x = F.relu(self.fc_start(x))
        for i in range(self.num_layers-2):
            x = F.relu(self.fc_bet(x))
        x = self.fc_end(x)
        return x

class Linear_classification_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers,sig_length,out_dim):
        super(Linear_classification_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_axis = num_axis
        self.sig_length = sig_length
        self.fc_start = nn.Linear(num_axis*sig_length,hidden_dim) 
        self.fc_bet = nn.Linear(hidden_dim,hidden_dim)
        self.fc_end = nn.Linear(hidden_dim,out_dim)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.25)
    
    def forward(self,x):
        x = x.view(-1,self.num_axis*self.sig_length)
        x = self.sigmoid(self.fc_start(x))
        for i in range(self.num_layers-2):
            x = F.relu(self.fc_bet(x))
        x = self.fc_end(x)
        return x
