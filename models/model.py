import torch
from torch import nn
from torchvision.models import resnet34, resnet18, resnet50


class YOLOv1(nn.Module):
    """YOLOv1 model structure
    yolov1 = resnet(backbon) + conv + fc
    """

    def __init__(self, B, nb_classes):
        super(YOLOv1, self).__init__()
        self.B = B
        self.nb_classes = nb_classes
        self.resnet = resnet18()  # you can replace to other models you like
        # print(self.resnet.fc.in_features)
        # print(*list(self.resnet.children())[-2:]) # show last two layers

        # backbone part, (cut resnet's last two layers)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])

        # conv part
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
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        # full connection part
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7*7*(self.B*5+self.nb_classes)),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv_layers(out)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, 7, 7, self.B*5+self.nb_classes)
        # out = out.reshape(-1, self.B*5+self.nb_classes, 7, 7) # must reshape to 30*7*7, because label is
        return out


def build_model(weight_path, cfg):
    model = YOLOv1(int(cfg['B']), int(cfg['nb_class']))

    # load pretrained model
    if weight_path and weight_path != '':
        model.load_state_dict(torch.load(weight_path))
    return model
