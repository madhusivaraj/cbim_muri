import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1  = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2  = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                weight_norm(self.conv1),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        )
        self.layers.append(
            nn.Sequential(
                weight_norm(self.conv2),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        )
        self.downsample = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu       = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x
        for l in self.layers:
            x = l(x)
        if self.downsample:
            residual = self.downsample(residual)
        return self.relu(residual + x)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, num_channels=[64, 64, 128], kernel_size=3, aggregation='average', num_classes=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        
        self.conv   = nn.Conv1d(in_channels=in_channels, out_channels=num_channels[0], 
                                    kernel_size=14, stride=7, padding=7 )
        
        self.layers = nn.ModuleList([
                nn.Sequential(
                    weight_norm(self.conv),
                    Chomp1d(chomp_size=kernel_size-1),
                    nn.ReLU()
                )
        ])
        
        num_levels  = len(num_channels)
        for i in range(1, num_levels):
            dilation_size = 2 ** i
            self.layers.append( TemporalBlock( in_channels=num_channels[i-1], out_channels=num_channels[i], kernel_size=kernel_size,
                                                stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout ) 
            )
        if aggregation == 'average':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif aggregation == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc      = nn.Linear(in_features=num_channels[-1], out_features=num_classes)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = self.pooling(x) # [B, Feat, 1]
        x = self.fc(x.squeeze())
        return x


if __name__ == "__main__":
    model      = TemporalConvNet(in_channels=17)
    from torchsummary import summary
    print(model)
    summary(model.to('cuda:0'), input_size=(17, 150))
