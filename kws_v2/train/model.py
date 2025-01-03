import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=True)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=True)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.conv1(x))
        x = F.relu_(self.conv2(x))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ExtractFeature(nn.Module):
    def __init__(self, sample_rate, num_mel=40, window='hann', center=True, pad_mode='reflect',
                ref= 1.0, n_fft=1024, amin=1e-6, top_db = None, hop_length=128, fmin=0):
        super(ExtractFeature, self).__init__()
        n_mels = num_mel
        win_length = n_fft
        fmax = sample_rate // 2
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True,is_log=True)

    def forward(self, inputs):
        x = self.spectrogram_extractor(inputs)
        # print('outspec:',x[0,0,0,:10])
        logmel_spec = self.logmel_extractor(x)
        # print('specshape:',logmel_spec.shape)
        return logmel_spec
    

class Cnn6(nn.Module):
    def __init__(self, classes_num, sample_rate, num_mel=40, window='hann', center=True, pad_mode='reflect',
                ref= 1.0, n_fft=1024, amin=1e-6, top_db = None, hop_length=128, fmin=0):
        
        super(Cnn6, self).__init__()
        n_mels = num_mel
        win_length = n_fft
        fmax = sample_rate // 2
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True,is_log=True)

        self.bn0 = nn.BatchNorm2d(40)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=32)
        self.conv_block2 = ConvBlock(in_channels=32, out_channels=64)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4 = ConvBlock(in_channels=128, out_channels=128)

        self.fc1 = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, x):
        """
        Input: (batch_size, data_length)"""
        
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        # bs,1,126,40

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        
        # clipwise_output = torch.sigmoid(self.fc_audioset(x))
        


        return x