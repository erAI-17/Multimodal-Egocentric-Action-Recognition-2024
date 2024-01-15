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
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.classifier(x)
        features = {"output features": x}  
        return logits, features


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
   
    
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
