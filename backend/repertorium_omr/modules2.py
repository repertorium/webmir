import torch
import torch.nn as nn

NUM_CHANNELS = 1
IMG_HEIGHT = 128


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: tuple):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, padding="same", bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.pool(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            ConvolutionalBlock(NUM_CHANNELS, 64, 5, (2, 2)),
            ConvolutionalBlock(64, 64, 5, (2, 1)),
            ConvolutionalBlock(64, 128, 3, (2, 1)),
            ConvolutionalBlock(128, 128, 3, (2, 1)),
        )

        # calculate the width reduction of the image taking into account all the pooling
        self.width_reduction = 2**1  # number of poolings in second dimension
        self.height_reduction = 2**4  # number of poolings in first dimension
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RNN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.blstm = nn.LSTM(
            input_size,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )
        # (batch_size, seq_len, 2 * 256)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256 * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (hidden_state, cell_state) = self.blstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class CRNN2(nn.Module):
    def __init__(self, output_size: int, freeze_cnn: bool = False, model_loaded=None):
        super().__init__()
        # CNN
        if freeze_cnn:
            self.cnn = model_loaded.model.cnn
            for param in self.cnn.parameters():
                param.requires_grad = False
            print("CNN freezed")
        else:
            self.cnn = CNN()
        # RNN
        # self.num_frame_repeats = self.cnn.width_reduction * frame_multiplier_factor
        self.rnn_input_size = self.cnn.out_channels * (
            IMG_HEIGHT // self.cnn.height_reduction
        )
        self.rnn = RNN(input_size=self.rnn_input_size, output_size=output_size)

    def forward(self, x):
        # CNN
        # x: [b, NUM_CHANNELS, IMG_HEIGHT, w]

        x = self.cnn(x.float())
        # x: [b, self.cnn.out_channels, nh = IMG_HEIGHT // self.height_reduction, nw = w // self.width_reduction]
        # Prepare for RNN
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        # x: [b, w, c, h]

        x = x.reshape(b, w, self.rnn_input_size)
        # x: [b, w, self.rnn_input_size]

        # RNN
        x = self.rnn(x)
        return x
