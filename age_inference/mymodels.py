import torch.nn as nn
import torch.nn.functional as F

class Lstm_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers,model_type):
        super(Lstm_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.lstm = nn.LSTM(num_axis,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,1)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self,x):
        _,x = self.lstm(x)
        x = x[0][-1].view(-1, self.hidden_dim)
        x = self.fc(x)
        if self.model_type == "classification":
            x = self.softmax(x)
        
        return x

class Conv1D_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,sig_length,model_type):
        super(Conv1D_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.conv1 = nn.Conv1d(sig_length,hidden_dim,kernel_size=num_axis,stride=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim,1)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = x.view(-1,self.hidden_dim)
        x = self.fc(x)
        if self.model_type == "classification":
            x = self.softmax(x)
        

        
        return x


class Linear_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers,sig_length,model_type):
        super(Linear_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.num_layers = num_layers
        self.num_axis = num_axis
        self.sig_length = sig_length
        self.fc_start = nn.Linear(num_axis*sig_length,hidden_dim) 
        self.fc_bet = nn.Linear(hidden_dim,hidden_dim)
        self.fc_end = nn.Linear(hidden_dim,1)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.25)
    
    def forward(self,x):
        x = x.view(-1,self.num_axis*self.sig_length)
        self.dropout(x)
        x = F.relu(self.fc_start(x))
        for i in range(self.num_layers-2):
            x = F.relu(self.fc_bet(x))
            self.dropout(x)
        x = self.fc_end(x)
        if self.model_type == "classification":
            x = self.softmax(x)

        return x
