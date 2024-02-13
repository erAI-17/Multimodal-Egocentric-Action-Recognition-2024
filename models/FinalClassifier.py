import torch
from torch import nn
import torch.nn.functional as F
from utils.args import args
import utils

input_size = 1024
hidden_size = 512

class MLP(nn.Module):
    def __init__(self):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        print("NUMCLASSES:",num_classes)
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.dropout= nn.Dropout(args.models.RGB.dropout)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        
    def forward(self, x):
        print("input SHAPE IS:",x.shape)
        if args.feat_avg:   #*Feature Averaging
            x = self.avg_pool(x.permute(0, 2, 1))  
            x = x.permute(0, 2, 1)
            print("input SHAPE if feat averaging IS:",x.shape)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            logits = self.fc3(x)
            print("logits SHAPE if feat averaging IS:",x.shape)
        else:              #*Logits Averaging
            
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            logits = self.fc3(x)
            logits = self.avg_pool(logits.permute(0, 2, 1)) 
            logits = logits.permute(0, 2, 1)
            print("logits SHAPE if NO feat averaging IS:",x.shape)
        features = {"output features": x}  # Create a dictionary of features from last layer
        return logits, features


class LSTM(nn.Module):
    def __init__(self, num_layers=1):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(args.models.RGB.dropout)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out[:, -1, :]) # extract last output of the sequence (the one obtained after all the timesteps)
        return out, {}
   
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout_rate):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size),
            num_layers
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        encoded = self.encoder(embedded)
        encoded = encoded.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        logits = self.fc(self.dropout(encoded))
        return F.log_softmax(logits, dim=2)

###############
#ACTION-NET MODELS
##############

class MLP_EMG(nn.Module):
    def __init__(self):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(args.models.RGB.dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(args.models.RGB.dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        
    def forward(self, x):
        if args.feat_avg:   #*Feature Averaging
            x = self.avg_pool(x.permute(0, 2, 1))  
            x = x.permute(0, 2, 1)
            
        x = torch.relu(self.fc1(x))
        x = torch.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = torch.dropout2(x)
        logits = self.classifier(x)
        
        if args.feat_avg==False:   #*Logits Averaging
            logits = self.avg_pool(logits.permute(0, 2, 1)) 
            logits = logits.permute(0, 2, 1)

        features = {"output features": x}  # Create a dictionary of features from last layer
        return logits, features

class LSTM_EMG(nn.Module):
    def __init__(self, num_layers=1):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(args.models.RGB.dropout)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out[:, -1, :]) # extract last output of the sequence (the one obtained after all the timesteps)
        return out, {}
   