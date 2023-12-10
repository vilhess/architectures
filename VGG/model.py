import torch
import torch.nn as nn


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, repeat=False):
        super(Conv_block, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.repeat = repeat
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv_1(x)))
        x = self.relu(self.bn(self.conv_2(x)))
        if self.repeat:
            x = self.relu(self.bn(self.conv_2(x)))
        return self.maxpool(x)


class VGG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG, self).__init__()

        self.block_1 = Conv_block(
            in_channels=in_channels, out_channels=64, kernel_size=3)
        self.block_2 = Conv_block(
            in_channels=64, out_channels=128, kernel_size=3)
        self.block_3 = Conv_block(
            in_channels=128, out_channels=256, kernel_size=3, repeat=True)
        self.block_4 = Conv_block(
            in_channels=256, out_channels=512, kernel_size=3, repeat=True)
        self.block_5 = Conv_block(
            in_channels=512, out_channels=512, kernel_size=3, repeat=True)

        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        self.dp = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dp(self.relu(self.fc1(x)))
        x = self.dp(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def test():
    x = torch.randn(10, 3, 224, 224).to('mps')
    model = VGG(in_channels=3, num_classes=1000).to('mps')
    print(model(x).shape)


if __name__ == '__main__':
    test()
