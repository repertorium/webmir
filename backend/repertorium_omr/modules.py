import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F

from repertorium_omr.preprocessing import NUM_CHANNELS, IMG_HEIGHT

BN_IDS = [1, 5, 9, 13]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.config = {
            "filters": [NUM_CHANNELS, 64, 64, 128, 128],
            "kernel": [5, 5, 3, 3],
            "pool": [[2, 2], [2, 1], [2, 1], [2, 1]],
            "leaky_relu": 0.2,
        }
        self.bn_ids = []

        layers = []
        for i in range(len(self.config["filters"]) - 1):
            layers.append(
                nn.Conv2d(
                    self.config["filters"][i],
                    self.config["filters"][i + 1],
                    self.config["kernel"][i],
                    padding="same",
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(self.config["filters"][i + 1]))
            layers.append(nn.LeakyReLU(self.config["leaky_relu"], inplace=True))
            layers.append(nn.MaxPool2d(self.config["pool"][i]))
            # Save BN ids
            self.bn_ids.append(len(layers) - 3)  # [1, 5, 9, 13]
        assert BN_IDS == self.bn_ids, "BN ids are not the same!"

        self.backbone = nn.Sequential(*layers)
        self.height_reduction, self.width_reduction = np.prod(
            self.config["pool"], axis=0
        )
        self.out_channels = self.config["filters"][-1]

    def forward(self, x):
        x = self.backbone(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.blstm = nn.LSTM(
            input_size,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256 * 2, output_size)

    def forward(self, x):
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class CRNN(nn.Module):
    def __init__(self, output_size):
        super(CRNN, self).__init__()
        # CNN
        self.cnn = CNN()
        # RNN
        self.rnn_input_size = self.cnn.out_channels * (
            IMG_HEIGHT // self.cnn.height_reduction
        )
        self.rnn = RNN(input_size=self.rnn_input_size, output_size=output_size)

    def forward(self, x):
        # CNN
        x = self.cnn(x)
        # Prepare for RNN
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.rnn_input_size)
        # RNN
        x = self.rnn(x)
        return x

    def da_forward(self, x, bn_ids):
        # CNN
        bn = []
        for i in range(len(self.cnn.backbone)):
            if i in bn_ids:
                bn.append(x.clone())
            x = self.cnn.backbone[i](x)
        # Prepare for RNN
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.rnn_input_size)
        # RNN
        x = self.rnn(x)
        return x, bn
    
class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1,1), dilation=(1,1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None
        
        if padding:
            if padding is True:
                padding = [int((k-1)/2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1,1))
        self.activation = activation

    def forward(self, inputs):
        x = self.depth_conv(inputs)
        if self.padding:
            x = F.pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        
        x = self.point_conv(x)

        return x
    
class MixDropout(nn.Module):
    def __init__(self, dropout_prob=0.4, dropout_2d_prob=0.2):
        super(MixDropout, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)
    
    def forward(self, inputs):
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2D(inputs)
    
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1), kernel=3, activation=nn.ReLU, dropout=0.5):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3,3), padding=(1,1), stride=stride)
        self.normLayer = nn.InstanceNorm2d(num_features=out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, inputs):
        pos = random.randint(1,3)

        x = self.conv1(inputs)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x

class DSCBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=(2, 1), activation=nn.ReLU, dropout=0.5):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        #x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x
    
class Encoder(nn.Module):

    def __init__(self, in_channels, dropout=0.5):
        super(Encoder, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=32, stride=(1,1), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=128, out_c=256, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=256, out_c=512, stride=(2,1), dropout=dropout),
            ConvBlock(in_c=512, out_c=512, stride=(2,1), dropout=dropout)
        ])

        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1,1), dropout = dropout)
        ])
    
    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt

        return x
    
class RecurrentDecoder(nn.Module):

    def __init__(self, out_cats):
        super(RecurrentDecoder, self).__init__()
        self.dec_lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True)
        self.out_dense = nn.Linear(in_features=512, out_features=out_cats)
    
    def forward(self, inputs):
        x = inputs
        b, c, h, w = x.size()
        x = x.reshape(b, c, h*w)
        x = x.permute(0,2,1)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        x = x.permute(1,0,2)
        return F.log_softmax(x, dim=2)
    
class E2EScore_CRNN(nn.Module):
    def __init__(self, in_channels, out_cats):
        super(E2EScore_CRNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = RecurrentDecoder(out_cats=out_cats)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x
