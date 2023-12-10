import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()

        self.conv1_1 = ConvBlock(3, 32, 3, 2, 0)
        self.conv1_2 = ConvBlock(32, 32, 3, 1, 0)
        self.conv1_3 = ConvBlock(32, 64, 3, 1, 1)

        self.conv2_1 = ConvBlock(64, 96, 3, 2, 0)

        self.conv3_1 = ConvBlock(160, 64, 1, 1, 0)
        self.conv3_2 = ConvBlock(64, 64, (1, 7), 1, (0, 3))
        self.conv3_3 = ConvBlock(64, 64, (7, 1), 1, (3, 0))
        self.conv3_4 = ConvBlock(64, 96, 3, 1, 0)

        self.conv4_1 = ConvBlock(192, 192, 3, 2, 0)

        self.conv3b_1 = ConvBlock(160, 64, 1, 1, 0)
        self.conv3b_2 = ConvBlock(64, 96, 3, 1, 0)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1_3(self.conv1_2(self.conv1_1(x)))

        x1 = self.conv2_1(x)
        x2 = self.maxpool(x)
        x = torch.cat([x1, x2], 1)

        x1 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        x2 = self.conv3b_2(self.conv3b_1(x))

        x = torch.cat([x1, x2], 1)

        x1 = self.conv4_1(x)
        x2 = self.maxpool(x)

        x = torch.cat([x1, x2], 1)

        return x


class InceptionBlockA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlockA, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=96,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=96, out_channels=96,
                      kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=96,
                      kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=96,
                      kernel_size=1, stride=1, padding=0)
        )

        self.branch4 = ConvBlock(
            in_channels=in_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], 1)


class ReductionBlockA(nn.Module):
    def __init__(self, in_channels):
        super(ReductionBlockA, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=224,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=224, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
        )

        self.branch2 = ConvBlock(
            in_channels=in_channels, out_channels=384, kernel_size=3, stride=2, padding=0)

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        return torch.cat([x1, x2, x3], 1)


class InceptionBlockB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlockB, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=192,
                      kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=192, out_channels=224,
                      kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=224, out_channels=224,
                      kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=224, out_channels=256,
                      kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=224,
                      kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=224, out_channels=256,
                      kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=128,
                      kernel_size=1, stride=1, padding=0)
        )

        self.branch4 = ConvBlock(
            in_channels=in_channels, out_channels=384, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], 1)


class ReductionBlockB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionBlockB, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=256,
                      kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=256, out_channels=320,
                      kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=320, out_channels=320,
                      kernel_size=3, stride=2, padding=0),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=192,
                      kernel_size=3, stride=2, padding=0)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        return torch.cat([x1, x2, x3], 1)


class InceptionBlockC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlockC, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=384,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=384, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
        )

        self.branch1_1 = ConvBlock(in_channels=512, out_channels=256, kernel_size=(
            1, 3), stride=1, padding=(0, 1))

        self.branch1_2 = ConvBlock(in_channels=512, out_channels=256, kernel_size=(
            3, 1), stride=1, padding=(1, 0))

        self.branch2 = ConvBlock(
            in_channels=in_channels, out_channels=384, kernel_size=1, stride=1, padding=0)

        self.branch2_1 = ConvBlock(in_channels=384, out_channels=256, kernel_size=(
            1, 3), stride=1, padding=(0, 1))

        self.branch2_2 = ConvBlock(in_channels=384, out_channels=256, kernel_size=(
            3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=256,
                      kernel_size=3, stride=1, padding=1)
        )

        self.branch4 = ConvBlock(
            in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x11 = self.branch1_1(self.branch1(x))
        x12 = self.branch1_2(self.branch1(x))
        x21 = self.branch2_1(self.branch2(x))
        x22 = self.branch2_2(self.branch2(x))
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x11, x12, x21, x22, x3, x4], 1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels):
        super(AuxiliaryClassifier, self).__init__()

        self.conv = ConvBlock(
            in_channels=in_channels, out_channels=1536, kernel_size=8, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=1536, out_features=1536)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1536, 1000)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return self.softmax(x)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.stem = StemBlock()
        self.block_a = InceptionBlockA(in_channels=384)
        self.red_a = ReductionBlockA(384)
        self.block_b = InceptionBlockB(in_channels=1024)
        self.red_b = ReductionBlockB(1024)
        self.block_c = InceptionBlockC(1536)
        self.clf = AuxiliaryClassifier(1536)

    def forward(self, x):

        x = self.stem(x)

        x = self.block_a(x)
        x = self.block_a(x)
        x = self.block_a(x)

        x = self.red_a(x)

        x = self.block_b(x)
        x = self.block_b(x)
        x = self.block_b(x)
        x = self.block_b(x)
        x = self.block_b(x)
        x = self.block_b(x)
        x = self.block_b(x)

        x = self.red_b(x)

        x = self.block_c(x)
        x = self.block_c(x)
        x = self.block_c(x)

        x = self.clf(x)

        return x


if __name__ == '__main__':

    x = torch.rand((10, 3, 299, 299)).to('mps')
    model = Inception().to('mps')
    print(model(x).shape)
