import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ELayer(nn.Module):
    def __init__(self, fc_input, pool_kernel, n_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(fc_input, n_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return F.sigmoid(self.fc(x))


class Encoder(nn.Module):
    def __init__(self, block, layers, num_classes=32):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dilation=2)
        self.layer4 = self._make_layer(
            block, num_classes, layers[3], dilation=4)

        self.es1 = ELayer(256, 7, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_classes * block.expansion, 256)
        self.fc_out = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          dilation=dilation),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        self.se2 = self.es1(x)

        x = self.layer4(x)
        self.feature_map = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.fc_out(x)
        self.se1 = F.sigmoid(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.cnv1 = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, padding=1)
        # self.cnv2 = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, padding=1)
        # self.cnv3 = nn.ConvTranspose2d(num_classes, num_classes, 3, 2)

        self.conv1 = nn.Conv2d(
            num_classes, num_classes*4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            num_classes, num_classes*4, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(
            num_classes, num_classes*4, kernel_size=3, padding=(0, 1), bias=False)

        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)
        self.ps3 = nn.PixelShuffle(2)

        self.bn1 = nn.BatchNorm2d(num_classes)
        self.bn2 = nn.BatchNorm2d(num_classes)

    def forward(self, input):
        h = self.ps1(F.selu(self.conv1(input)))
        # h = F.selu(self.cnv1(input))

        h = self.ps2(F.selu(self.conv2(h)))
        # h = F.selu(self.cnv2(h))

        h = self.ps3(self.conv3(h))
        # h = self.cnv3(h)
        return h


class Net(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.encoder = Encoder(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.decoder = Decoder(num_classes)

    def forward(self, input):
        h = self.encoder(input)
        decoder_in =\
            self.encoder.feature_map\
            * h.repeat(self.encoder.feature_map.shape[2:])\
               .reshape(self.encoder.feature_map.shape)
        out = self.decoder(decoder_in)
        return out, self.encoder.se2, self.encoder.se1


def enc_net(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Net(num_classes, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
