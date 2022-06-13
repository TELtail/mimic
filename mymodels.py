import torch.nn as nn
import torch.nn.functional as F

class Lstm_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers):
        super(Lstm_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_axis,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,1)

        
    def forward(self,x):
        _,x = self.lstm(x)
        x = x[0][-1].view(-1, self.hidden_dim)
        x = self.fc(x)
        
        return x

class Linear_net(nn.Module):
    def __init__(self,num_axis,hidden_dim,num_layers,data_len):
        super(Linear_net,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_axis = num_axis
        self.data_len = data_len
        self.fc_start = nn.Linear(num_axis*data_len,hidden_dim) 
        self.fc_bet = nn.Linear(hidden_dim,hidden_dim)
        self.fc_end = nn.Linear(hidden_dim,1)

        self.dropout = nn.Dropout(0.25)
    
    def forward(self,x):
        x = x.view(-1,self.num_axis*self.data_len)
        self.dropout(x)
        x = F.relu(self.fc_start(x))
        for i in range(self.num_layers-2):
            x = F.relu(self.fc_bet(x))
            self.dropout(x)
        x = self.fc_end(x)

        return x
