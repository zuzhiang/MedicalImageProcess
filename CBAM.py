import torch
import torch.nn as nn


class CBAM_Module(nn.Module):
    def __init__(self, dim, in_channels, ratio, kernel_size):
        super(CBAM_Module, self).__init__()
        self.avg_pool = getattr(nn, "AdaptiveAvgPool{0}d".format(dim))(1)
        self.max_pool = getattr(nn, "AdaptiveMaxPool{0}d".format(dim))(1)
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        self.fc1 = conv_fn(in_channels, in_channels // ratio, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = conv_fn(in_channels // ratio, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = conv_fn(2, 1, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        print("CBAM")
        # Channel attention module:（Mc(f) = σ(MLP(AvgPool(f)) + MLP(MaxPool(f)))）
        module_input = x
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = self.sigmoid(avg + mx)
        x = module_input * x
        # Spatial attention module:Ms (f) = σ( f7×7( AvgPool(f) ; MaxPool(F)] )))
        module_input = x
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((avg, mx), dim=1)
        x = self.sigmoid(self.conv(x))
        x = module_input * x
        return x