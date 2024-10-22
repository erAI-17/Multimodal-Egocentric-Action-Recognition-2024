import torch
from torch import nn
import torch.nn.functional as F
from utils.args import args
import utils
import torchaudio.transforms as T
import numpy as np

input_size = 1024
hidden_size = 512

class MLP(nn.Module):
    def __init__(self):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.dropout= nn.Dropout(args.models.RGB.dropout)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        
    def forward(self, x):
        if args.feat_avg:   #*Feature Averaging
            x = self.avg_pool(x.permute(0, 2, 1))  
            x = x.permute(0, 2, 1)
            x = x.squeeze(dim=1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            logits = self.fc3((x))
        else:              #*Logits Averaging
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            feat = x
            logits = self.fc3(x)
            logits = self.avg_pool(logits.permute(0, 2, 1)) 
            logits = logits.permute(0, 2, 1)
            logits = logits.squeeze(dim=1)
       
        return logits, {'feat':feat}


class LSTM(nn.Module):
    def __init__(self, num_layers=1):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(args.models.RGB.dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        #out = self.relu(out)
        out = self.fc(out[:, -1, :])
        return out, {}
   
class TRN(torch.nn.Module):
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    def __init__(self, img_feature_dim=1024, num_frames=args.train.num_frames_per_clip, num_class=valid_labels, dropout=0.5):
        super(TRN, self).__init__()
        self.subsample_num = 3
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] 

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) 

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() 
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]

        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, x):
        act_all = x[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = x[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = self.classifier_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all, {'features': x}

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


#!##############
#!ACTION-NET MODELS
#!#############
class LSTM_EMG(nn.Module):
    def __init__(self, num_layers=1):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(LSTM_EMG, self).__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=100, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(args.models.EMG.dropout) #0.2
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.float() #It receives float64 but can work only on float32
        out, _ = self.lstm(x)
        out = self.dropout(out)
        feat = out[:, -1, :]
        out = self.fc(out[:, -1, :]) # extract last output of the sequence (the one obtained after all the timesteps)
        return out, {'feat':feat}
    
    
class MLP_EMG(nn.Module):
    def __init__(self):
        input_size = 16
        hidden_size = 512
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(MLP_EMG, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.dropout= nn.Dropout(args.models.RGB.dropout)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        
    def forward(self, x):
        x = x.float()
        if args.feat_avg:   #*Feature Averaging
            x = self.avg_pool(x.permute(0, 2, 1))  
            x = x.permute(0, 2, 1)
            x = x.squeeze(dim=1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            logits = self.fc3((x))
        else:              #*Logits Averaging
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            feat = x
            logits = self.fc3(x)
            logits = self.avg_pool(logits.permute(0, 2, 1)) 
            logits = logits.permute(0, 2, 1)
            logits = logits.squeeze(dim=1)
       
        return logits, {'feat':feat}



class CNN_EMG(nn.Module):
    def __init__(self):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(CNN_EMG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256*1*3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        n_fft = 32
        win_length = None
        hop_length = 16
        
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=True
        )
        
        x = torch.stack([spectrogram(x[:,:, i]) for i in range(16)], dim=1).float()
       
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 256*1*3 )
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x, {}
    

class FUSION_net(nn.Module):
    def __init__(self):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(FUSION_net, self).__init__()
        self.rgb_model = MLP() 
        self.emg_model = MLP_EMG() 
        self.fc1 = nn.Linear(512, 256)   
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, num_classes)  

    def forward(self, x):
        rgb_output, rgb_feat  = self.rgb_model(x['RGB']) #[32, 10, 512]
        emg_output, emg_feat = self.emg_model(x['EMG']) #[32, 100,512]
        
        # Aggregate features to match dimensions
        rgb_feat_pooled = rgb_feat['feat'].mean(dim=1)  # [batch_size, 512]
        emg_feat_pooled = emg_feat['feat'].mean(dim=1)  # [batch_size, 512]
        combined = rgb_feat_pooled + emg_feat_pooled

        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x, {}
    
  
class basic_attention(nn.Module):
    def __init__(self, feat_dim, reduction_ratio=8):
        super(basic_attention, self).__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim//reduction_ratio)
        self.fc2 = nn.Linear(feat_dim//reduction_ratio, feat_dim)

    def forward(self, feats):
        attention_weights = torch.sigmoid(self.fc2(F.relu(self.fc1(feats))))
        
        return attention_weights
    
class basic_att_fusion(nn.Module):
    def __init__(self):
        num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
        super(basic_att_fusion, self).__init__()

        self.rgb_model = MLP() 
        self.emg_model = MLP_EMG() 
        
        self.attention_rgb = basic_attention(512, reduction_ratio=8) 
        self.attention_emg = basic_attention(512, reduction_ratio=8) 
        
        self.fc1 = nn.Linear(512*2, 256)   
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, num_classes)  
            
        
    def forward(self, x):
        rgb_output, rgb_feat  = self.rgb_model(x['RGB']) #[32, 10, 512]
        emg_output, emg_feat = self.emg_model(x['EMG']) #[32, 100,512]
        
        # Apply attention to each feature
        rgb_attention = self.attention_rgb(rgb_feat['feat'])  # [batch_size, 10, 512]
        emg_attention = self.attention_emg(emg_feat['feat'])  # [batch_size, 100, 512]

        # Re-weight the features
        attended_rgb_feats = rgb_attention * rgb_feat['feat']  # [batch_size, 10, 512]
        attended_emg_feats = emg_attention * emg_feat['feat']  # [batch_size, 100, 512]
        
        # Aggregate the features
        aggregated_rgb_feats = attended_rgb_feats.mean(dim=1)  # [batch_size, 512]
        aggregated_emg_feats = attended_emg_feats.mean(dim=1)  # [batch_size, 512]
        
        # Concatenate the aggregated features
        combined_feats = torch.cat((aggregated_rgb_feats, aggregated_emg_feats), dim=1)  # [batch_size, 1024]
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(combined_feats))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x, {}
    