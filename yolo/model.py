import torch
import torch.nn as nn

architectures = [
    [3, 64, 7, 2, 3, True],
    [64, 192, 3, 1, 1, True],
    [192, 128, 1, 1, 0],
    [128, 256, 3, 1, 1],
    [256, 256, 1, 1, 0],
    [256, 512, 3, 1, 1, True],
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 256, 1, 1, 0],
    [256, 512, 3, 1, 1],
    [512, 512, 1, 1, 0],
    [512, 1024, 3, 1, 1, True],
    [1024, 512, 1, 1, 0],
    [512, 1024, 3, 1, 1],
    [1024, 512, 1, 1, 0],
    [512, 1024, 3, 1, 1],
    [1024, 1024, 3, 1, 1],
    [1024, 1024, 3, 2, 1],
    [1024, 1024, 3, 1, 1],
    [1024, 1024, 3, 1, 1]
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, ispool = False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()
        self.ispool = ispool
        if ispool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.maxpool = nn.Identity()

    def forward(self, x):
        return self.maxpool(self.act(self.bn(self.conv(x))))
    

class Yolo(nn.Module):

    def __init__(self):
        super(Yolo, self).__init__()
        self.architectures = architectures
        self.base_network = self.get_convs()
        self.final_layers = self.fcs(7, 2, 20)

    def get_convs(self):
        layers = []
        for layer in self.architectures:
            in_c = layer[0]
            out_c = layer[1]
            k = layer[2]
            s = layer[3]
            p = layer[4]
            pool = layer[5] if len(layer)==6 else False
            layers.append(
                ConvBlock(in_channels=in_c, out_channels=out_c, 
                                    kernel_size=k, stride=s, 
                                    padding=p, ispool=pool)
                        )
        return nn.Sequential(*layers)
    
    def fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(C+B*5))
        )
    
    def forward(self, x):
        x = self.base_network(x)
        out = self.final_layers(torch.flatten(x, start_dim=1))
        return out

if __name__=='__main__':

    x = torch.rand((10, 3, 448, 448))
    model = Yolo()

    print(x.shape)
    print(model(x).shape)

