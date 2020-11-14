import torch
from torch import nn
from torchvision.models import resnet34,resnet18


class YOLOv1(nn.Module):
    def __init__(self, B, nb_classes):
        super(YOLOv1, self).__init__()
        self.B = B
        self.nb_classes = nb_classes
        self.resnet = resnet18()
        # print(self.resnet.fc.in_features)
        # print(*list(self.resnet.children())[-2:])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*512, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 7*7*(self.B*5+self.nb_classes)),
            nn.Sigmoid() # must add 0~1
        )

    def forward(self, x):
        out = self.resnet(x)
        out = self.conv_layers(out)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, 7, 7, (self.B*5+self.nb_classes))
        return out
