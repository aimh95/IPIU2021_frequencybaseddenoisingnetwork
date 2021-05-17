import torch.nn as nn
import torch.nn.init as init
import math
import torch
from typing import Union, List, Dict,Any, cast
import torch.nn.functional as F

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class Model(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(Model, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)



class LowFreq(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(LowFreq, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class EDSRResBlock(nn.Module):
    def __init__(self, nchannel=64):
        super(EDSRResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(nchannel, eps = 0.0001, momentum = 0.95)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        identity_data = input
        output = self.relu(self.conv1(input))
        # output *= 0.1
        output = torch.add(output*0.1, identity_data)
        return output

class NoBatchNorm(nn.Module):
    def __init__(self, nchannel=64, image_channels=1, n_block=32):
        super(NoBatchNorm, self).__init__()
        self.n_block = n_block

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv1 = nn.Conv2d(image_channels, nchannel, 3, stride=1, padding=1)

        self.residual = self.make_layer(ResBlock, n_block)

        self.conv2 = nn. Conv2d(nchannel, nchannel, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(nchannel, 1, 3, stride=1, padding=1)
        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        y = input
        # output = self.sub_mean(input)
        output = self.conv1(input)
        residual = output
        output = self.conv2(self.residual(output))
        output = torch.add(output, residual)
        output = self.conv3(output)
        output = self.conv4(output)
        # output = self.add_mean(output)
        return y-output


class Encoder1(nn.Module):
    def __init__(self, nchannel=64, image_channels=1, n_block=18):
        super(Encoder1, self).__init__()
        self.n_block = n_block

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv1 = nn.Conv2d(image_channels, nchannel, 3, stride=1, padding=1)

        self.residual = self.make_layer(EDSRResBlock, n_block)

        self.conv2 = nn. Conv2d(nchannel, nchannel, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(nchannel, 1, 3, stride=1, padding=1)
        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        y = input
        # output = self.sub_mean(input)
        output = self.conv1(input)
        residual = output
        output = self.conv2(self.residual(output))
        output = torch.add(output, residual)
        output = self.conv3(output)
        output = self.conv4(output)
        # output = self.add_mean(output)
        return y-output

class Encoder2(nn.Module):
    def __init__(self, nchannel=64, image_channels=1, n_block=18):
        super(Encoder2, self).__init__()
        self.n_block = n_block

        self.conv1 = nn.Conv2d(image_channels, nchannel, 3, stride=1, padding=1)

        self.residual = self.make_layer(EDSRResBlock, n_block)

        self.conv2 = nn. Conv2d(nchannel, nchannel, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(nchannel, 1, 3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv1(input)
        residual = output
        output = self.conv2(self.residual(output))
        output = torch.add(output, residual)
        output = self.conv3(output)
        output = self.conv4(output)
        return output

class ConcatLayer(nn.Module):
    def __init__(self, skip_input, output_features):
        super(ConcatLayer, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = self.leakyreluA(self.convA( torch.cat([x, concat_with], dim=1)))
        return self.leakyreluB( self.convB(up_x ))


class Decoder(nn.Module):
    def __init__(self, features_num=128, decoder_params=0.5):
    # def __init__(self, nchannel=64, image_channels=1, n_block=16):
        super(Decoder, self).__init__()
        features = int(features_num * decoder_params)  # 64

        self.conv1 = nn.Conv2d(features_num, features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

        self.upsample1 = ConcatLayer(features + 128, features)  # 256  features[8]
        self.upsample2 = ConcatLayer(features + 128, features)  # 128  features[6]

        self.residual = self.make_layer(EDSRResBlock, 16)

        self.conv3 = nn.Conv2d(features, 1, kernel_size=3, stride=1, padding=1)

    def make_layer(selfself, block, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, denoise, edge):
        denoise0, denoise1, denoise2 = denoise[0], denoise[1], denoise[2]
        edge0, edge1, edge2 = edge[0], edge[1], edge[2]

        x_block0, x_block1, x_block2 = torch.cat([denoise0,edge0], dim=1), torch.cat([denoise1,edge1], dim=1), torch.cat([denoise2,edge1], dim=1)

        x_d0 = self.conv2(self.conv1(x_block0))
        x_d1 = self.upsample1(x_d0, x_block1)
        x_d2 = self.upsample2(x_d1, x_block2)
        return self.conv3(x_d2)


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Sigmoid()
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)

class ResBlock(nn.Module):
    def __init__(self, nchannel=64):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(nchannel, eps = 0.0001, momentum = 0.95)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        identity_data = input
        output = self.relu(self.conv1(input))
        # output *= 0.1
        output = torch.add(output*0.1, identity_data)
        return output

class DecomposeNet(nn.Module):
    def __init__(self, nchannel=64, image_channels=2, n_block=32):
        super(DecomposeNet, self).__init__()
        self.n_block = n_block

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv1 = nn.Conv2d(image_channels, nchannel, 3, stride=1, padding=1)

        self.residual = self.make_layer(ResBlock, n_block)

        self.conv2 = nn. Conv2d(nchannel, nchannel, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(nchannel, image_channels, 3, stride=1, padding=1)
        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        y = input
        # output = self.sub_mean(input)
        output = self.conv1(input)
        residual = output
        output = self.conv2(self.residual(output))
        output = torch.add(output, residual)
        output = self.conv3(output)
        output = self.conv4(output)
        # output = self.add_mean(output)
        return y-output

class DecomposeNetLowFreq(nn.Module):
    def __init__(self, nchannel=64, image_channels=2, n_block=32):
        super(DecomposeNetLowFreq, self).__init__()
        self.n_block = n_block

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv1 = nn.Conv2d(image_channels, nchannel, 3, stride=1, padding=1)

        self.residual = self.make_layer(ResBlock, n_block)

        self.conv2 = nn. Conv2d(nchannel, nchannel, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(nchannel, image_channels, 3, stride=1, padding=1)
        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        # output = self.sub_mean(input)
        output = self.conv1(input)
        residual = output
        output = self.conv2(self.residual(output))
        output = torch.add(output, residual)
        output = self.conv3(output)
        output = self.conv4(output)
        # output = self.add_mean(output)
        return output


class ComposeNet(nn.Module):
    def __init__(self, nchannel=64, input_channel=2, output_channel=1, n_block=16):
        super(ComposeNet, self).__init__()
        self.n_block = n_block

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv1 = nn.Conv2d(input_channel, nchannel, 3, stride=1, padding=1)

        self.residual = self.make_layer(ResBlock, n_block)

        self.conv2 = nn. Conv2d(nchannel, nchannel, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(nchannel, output_channel, 3, stride=1, padding=1)
        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv1(input)
        residual = output
        output = self.conv2(self.residual(output))
        output = torch.add(output, residual)
        # output = self.conv3(output)
        output = self.conv4(output)
        return output

class UpSampling(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampling, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.subpixel = nn.PixelShuffle(2)

    def forward(self, x, concat_with):
        out = self.relu(self.conv(x))
        out = self.subpixel(out)
        return out

class UNet(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNet, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.relu = nn.PReLU()
        self.downsampling = nn.MaxPool2d(2,2)

        self.input_conv = nn.Conv2d(input_channels, 32, 3, stride=1, padding=1)

        self.encode1_conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.encode1_conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.encode1_conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.encode1_conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.encode1_bn = nn.BatchNorm2d(32, eps=0.0001, momentum=0.95)

        self.encode2_conv1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encode2_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.encode2_conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.encode2_conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.encode2_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.95)

        self.encode3_conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encode3_conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.encode3_conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.encode3_conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.encode3_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.95)

        self.encode4_conv1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encode4_conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.encode4_conv3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.encode4_conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.encode4_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.95)

        self.upsample1 = UpSampling(384, 128)
        self.decode1_conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.decode1_conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.decode1_conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.decode1_conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.decode1_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.95)

        self.upsample2 = UpSampling(192, 64)
        self.decode2_conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.decode2_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.decode2_conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.decode2_conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.decode2_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.95)

        self.upsample3 = UpSampling(96, 32)
        self.decode3_conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.decode3_conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.decode3_conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.decode3_conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.decode3_bn = nn.BatchNorm2d(32, eps=0.0001, momentum=0.95)

        self.recon = nn.Conv2d(32, image_channels, 3, stride=1, padding=1)

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, input_img, input_noise):
        # y = input[:,0:2]
        input1 = input_img
        input2 = input_noise
        y = torch.cat([input1, input2], dim=1)

        # output = self.sub_mean(input)
        output = self.relu(self.input_conv(y))

        output = self.encode1_bn(self.relu(self.encode1_conv1(output)))
        output = self.encode1_bn(self.relu(self.encode1_conv2(output)))
        output = self.encode1_bn(self.relu(self.encode1_conv3(output)))
        output = self.encode1_bn(self.relu(self.encode1_conv4(output)))
        encode1 = output
        output = self.downsampling(output)

        output = self.encode2_bn(self.relu(self.encode2_conv1(output)))
        output = self.encode2_bn(self.relu(self.encode2_conv2(output)))
        output = self.encode2_bn(self.relu(self.encode2_conv3(output)))
        output = self.encode2_bn(self.relu(self.encode2_conv4(output)))
        encode2 = output
        output = self.downsampling(output)

        output = self.encode3_bn(self.relu(self.encode3_conv1(output)))
        output = self.encode3_bn(self.relu(self.encode3_conv2(output)))
        output = self.encode3_bn(self.relu(self.encode3_conv3(output)))
        output = self.encode3_bn(self.relu(self.encode3_conv4(output)))
        encode3 = output
        output = self.downsampling(output)

        output = self.encode4_bn(self.relu(self.encode4_conv1(output)))
        output = self.encode4_bn(self.relu(self.encode4_conv2(output)))
        output = self.encode4_bn(self.relu(self.encode4_conv3(output)))
        output = self.encode4_bn(self.relu(self.encode4_conv4(output)))
        output = self.downsampling(output)

        output = self.upsample1(output, encode3)
        output = self.decode1_bn(self.relu(self.decode1_conv1(output)))
        output = self.decode1_bn(self.relu(self.decode1_conv2(output)))
        output = self.decode1_bn(self.relu(self.decode1_conv3(output)))

        output = self.upsample2(output, encode2)
        output = self.decode2_bn(self.relu(self.decode2_conv1(output)))
        output = self.decode2_bn(self.relu(self.decode2_conv2(output)))
        output = self.decode2_bn(self.relu(self.decode2_conv3(output)))

        output = self.upsample3(output, encode1)
        output = self.decode3_bn(self.relu(self.decode3_conv1(output)))
        output = self.decode3_bn(self.relu(self.decode3_conv2(output)))
        output = self.decode3_bn(self.relu(self.decode3_conv3(output)))

        residual = self.recon(output)



        return input1-residual

class ResBlock_noBN(nn.Module):
    def __init__(self, nchannel=64):
        super(ResBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nchannel, nchannel, 3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(nchannel, eps = 0.0001, momentum = 0.95)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        identity_data = input
        output = self.relu(self.conv1(input))
        # output *= 0.1
        output = torch.add(output*0.1, identity_data)
        return output

class UNetDense(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDense, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.relu = nn.PReLU()
        self.downsampling = nn.MaxPool2d(2,2)

        self.input_conv = nn.Conv2d(input_channels, 32, 3, stride=1, padding=1)

        self.encode1_conv1 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_conv2 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_conv3 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_conv4 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_bn = nn.BatchNorm2d(32, eps=0.0001, momentum=0.95)

        self.encode2_conv1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encode2_conv2 = self.make_layer(ResBlock_noBN,64, 4)
        self.encode2_conv3 = self.make_layer(ResBlock_noBN,64, 4)
        self.encode2_conv4 = self.make_layer(ResBlock_noBN,64, 4)
        self.encode2_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.95)

        self.encode3_conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encode3_conv2 = self.make_layer(ResBlock_noBN,128, 4)
        self.encode3_conv3 = self.make_layer(ResBlock_noBN,128, 4)
        self.encode3_conv4 = self.make_layer(ResBlock_noBN,128, 4)
        self.encode3_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.95)

        self.encode4_conv1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encode4_conv2 = self.make_layer(ResBlock_noBN,256, 4)
        self.encode4_conv3 = self.make_layer(ResBlock_noBN,256, 4)
        self.encode4_conv4 = self.make_layer(ResBlock_noBN,256, 4)
        self.encode4_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.95)

        self.upsample1 = UpSampling(384, 128)
        self.decode1_conv1 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_conv2 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_conv3 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_conv4 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.95)

        self.upsample2 = UpSampling(192, 64)
        self.decode2_conv1 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_conv2 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_conv3 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_conv4 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.95)

        self.upsample3 = UpSampling(96, 32)
        self.decode3_conv1 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_conv2 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_conv3 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_conv4 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_bn = nn.BatchNorm2d(32, eps=0.0001, momentum=0.95)

        self.recon = nn.Conv2d(32, image_channels, 3, stride=1, padding=1)

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, nchannel, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img, input_noise):
        # y = input[:,0:2]
        input1 = input_img
        input2 = input_noise
        y = torch.cat([input1, input2], dim=1)

        # output = self.sub_mean(input)
        output = self.relu(self.input_conv(y))

        output = (self.relu(self.encode1_conv1(output)))
        output = (self.relu(self.encode1_conv2(output)))
        output = (self.relu(self.encode1_conv3(output)))
        output = (self.relu(self.encode1_conv4(output)))
        encode1 = output
        output = self.downsampling(output)

        output = (self.relu(self.encode2_conv1(output)))
        output = (self.relu(self.encode2_conv2(output)))
        output = (self.relu(self.encode2_conv3(output)))
        output = (self.relu(self.encode2_conv4(output)))
        encode2 = output
        output = self.downsampling(output)

        output = (self.relu(self.encode3_conv1(output)))
        output = (self.relu(self.encode3_conv2(output)))
        output = (self.relu(self.encode3_conv3(output)))
        output = (self.relu(self.encode3_conv4(output)))
        encode3 = output
        output = self.downsampling(output)

        output = (self.relu(self.encode4_conv1(output)))
        output = (self.relu(self.encode4_conv2(output)))
        output = (self.relu(self.encode4_conv3(output)))
        output = (self.relu(self.encode4_conv4(output)))
        output = self.downsampling(output)

        output = self.upsample1(output, encode3)
        output = (self.relu(self.decode1_conv1(output)))
        output = (self.relu(self.decode1_conv2(output)))
        output = (self.relu(self.decode1_conv3(output)))

        output = self.upsample2(output, encode2)
        output = (self.relu(self.decode2_conv1(output)))
        output = (self.relu(self.decode2_conv2(output)))
        output = (self.relu(self.decode2_conv3(output)))

        output = self.upsample3(output, encode1)
        output = (self.relu(self.decode3_conv1(output)))
        output = (self.relu(self.decode3_conv2(output)))
        output = (self.relu(self.decode3_conv3(output)))

        # output = self.encode1_bn(self.relu(self.encode1_conv1(output)))
        # output = self.encode1_bn(self.relu(self.encode1_conv2(output)))
        # output = self.encode1_bn(self.relu(self.encode1_conv3(output)))
        # output = self.encode1_bn(self.relu(self.encode1_conv4(output)))
        # encode1 = output
        # output = self.downsampling(output)
        #
        # output = self.encode2_bn(self.relu(self.encode2_conv1(output)))
        # output = self.encode2_bn(self.relu(self.encode2_conv2(output)))
        # output = self.encode2_bn(self.relu(self.encode2_conv3(output)))
        # output = self.encode2_bn(self.relu(self.encode2_conv4(output)))
        # encode2 = output
        # output = self.downsampling(output)
        #
        # output = self.encode3_bn(self.relu(self.encode3_conv1(output)))
        # output = self.encode3_bn(self.relu(self.encode3_conv2(output)))
        # output = self.encode3_bn(self.relu(self.encode3_conv3(output)))
        # output = self.encode3_bn(self.relu(self.encode3_conv4(output)))
        # encode3 = output
        # output = self.downsampling(output)
        #
        # output = self.encode4_bn(self.relu(self.encode4_conv1(output)))
        # output = self.encode4_bn(self.relu(self.encode4_conv2(output)))
        # output = self.encode4_bn(self.relu(self.encode4_conv3(output)))
        # output = self.encode4_bn(self.relu(self.encode4_conv4(output)))
        # output = self.downsampling(output)
        #
        # output = self.upsample1(output, encode3)
        # output = self.decode1_bn(self.relu(self.decode1_conv1(output)))
        # output = self.decode1_bn(self.relu(self.decode1_conv2(output)))
        # output = self.decode1_bn(self.relu(self.decode1_conv3(output)))
        #
        # output = self.upsample2(output, encode2)
        # output = self.decode2_bn(self.relu(self.decode2_conv1(output)))
        # output = self.decode2_bn(self.relu(self.decode2_conv2(output)))
        # output = self.decode2_bn(self.relu(self.decode2_conv3(output)))
        #
        # output = self.upsample3(output, encode1)
        # output = self.decode3_bn(self.relu(self.decode3_conv1(output)))
        # output = self.decode3_bn(self.relu(self.decode3_conv2(output)))
        # output = self.decode3_bn(self.relu(self.decode3_conv3(output)))

        residual = self.recon(output)



        return residual




class _down(nn.Module):
    def __init__(self, nchannel):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=2*nchannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)

        out = self.relu(self.conv(out))

        return out

class _up(nn.Module):
    def __init__(self, nchannel):
        super(_up, self).__init__()
        self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=nchannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.subpixel(out)
        return out
class DCR_Block(nn.Module):
    def __init__(self, nchannel=64):
        super(DCR_Block, self).__init__()
        self.conv1 = nn.Conv2d(nchannel, nchannel//2, 3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(nchannel*3//2, nchannel//2, 3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(nchannel*2, nchannel, 3, stride=1, padding=1)
        self.relu3 = nn.PReLU()
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv3(conc))
        out = torch.add(out, residual)
        return out

class ResBlockFinal(nn.Module):
    def __init__(self, nchannel=64):
        super(ResBlockFinal, self).__init__()
        self.conv1 = nn.Conv2d(nchannel, nchannel//2, 3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(nchannel*3//2, nchannel // 2, 3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(nchannel * 2, nchannel, 3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

    def forward(self, input):
        out = self.relu1(self.conv1(input))
        conc = torch.cat([input, out], 1)
        out = self.relu2(self.conv2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv3(conc))
        return out


class UBlock(nn.Module):
    def __init__(self, nchannel):
        super(UBlock, self).__init__()
        self.relu1 = nn.PReLU()
        self.encode1_conv1 = self.make_layer(ResBlockFinal, 128)
        self.encode1_conv2 = self.make_layer(ResBlockFinal, 128)
        self.down1 = self.make_layer(_down, 128)

        self.encode2_conv1 = self.make_layer(ResBlockFinal, 256)
        self.encode2_conv2 = self.make_layer(ResBlockFinal, 256)
        self.down2 = self.make_layer(_down, 256)

        self.encode3_conv1 = self.make_layer(ResBlockFinal, 512)
        self.encode3_conv2 = self.make_layer(ResBlockFinal, 512)
        self.down3 = self.make_layer(_down, 512)

        self.encode4_conv1 = self.make_layer(ResBlockFinal, 1024)
        self.encode4_conv2 = self.make_layer(ResBlockFinal, 1024)
        # self.down4 = self.make_layer(_down, 1024)

        self.up1 = self.make_layer(_up, 2048)
        self.decode1_conv1 = self.make_layer(ResBlockFinal, 1024)
        self.decode1_conv2 = self.make_layer(ResBlockFinal, 1024)

        self.up2 = self.make_layer(_up, 1024)
        self.decode2_conv1 = self.make_layer(ResBlockFinal, 512)
        self.decode2_conv2 = self.make_layer(ResBlockFinal, 512)

        self.up3 = self.make_layer(_up, 512)
        self.decode3_conv1 = self.make_layer(ResBlockFinal, 256)
        self.decode3_conv2 = self.make_layer(ResBlockFinal, 256)
        self.relu2 = nn.PReLU()

        self.convf = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)


    def forward(self, x):
        residual = x
        # out = self.relu1(self.conv_i(x))
        out = self.encode1_conv1(x)
        conc1 = self.encode1_conv2(out)
        out = self.down1(conc1)
        out = self.encode2_conv1(out)
        conc2 = self.encode2_conv2(out)
        out = self.down2(conc2)
        out = self.encode3_conv1(out)
        conc3 = self.encode3_conv2(out)
        conc4 = self.down3(conc3)
        out = self.encode4_conv1(conc4)
        out = self.encode4_conv2(out)
        out = torch.cat([conc4, out], 1)
        out = self.up1(out)
        out = torch.cat([conc3, out], 1)
        out = self.decode1_conv1(out)
        out = self.decode1_conv2(out)
        out = self.up2(out)
        out = torch.cat([conc2, out], 1)
        out = self.decode2_conv1(out)
        out = self.decode2_conv2(out)
        out = self.up3(out)
        out = torch.cat([conc1, out], 1)
        out = self.decode3_conv1(out)
        out = self.decode3_conv2(out)
        out = self.relu2(self.convf(out))
        out = torch.add(residual, out)
        return out

class ReconBlock(nn.Module):
    def __init__(self, nchannel):
        super(ReconBlock, self).__init__()

        self.conv1 = nn.Conv2d(nchannel, 256, 3, padding=1, stride=1)
        self.relu1 = nn.PReLU()

        self.conv2 = self.make_layer(DCR_Block, 256)
        self.conv3 = self.make_layer(DCR_Block, 256)
        self.conv4 = self.make_layer(DCR_Block, 256)
        self.conv5 = self.make_layer(DCR_Block, 256)
        self.conv6 = self.make_layer(DCR_Block, 256)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.relu1(self.conv1(x))

        res2 = output
        output =self.conv2(output)
        output = torch.add(res2,output)

        res3 = output
        output = self.conv3(output)
        output = torch.add(res3,output)

        res4 = output
        output =self.conv4(output)
        output = torch.add(res4, output)

        res5 = output
        output = self.conv5(output)
        output = torch.add(res5, output)

        output = self.conv6(output)
        output = torch.add(output, res2)

        return output


class UNetDenseFrequencyVer2(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequencyVer2, self).__init__()

        self.relu = nn.PReLU()
        self.downsampling = nn.MaxPool2d(2,2)

        self.input_conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.feature1 = self.make_layer(UBlock, 128)


        self.input_conv2 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.feature2 = self.make_layer(UBlock, 128)

        self.input_conv3 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.feature3 = self.make_layer(UBlock, 128)

        self.conv = nn.Conv2d(384, 256, 1, stride=1, padding=0)
        self.relu1 = nn.PReLU()

        self.recon_block = self.make_layer(ReconBlock, 384)

        self.recon = nn.Conv2d(256, image_channels, 3, stride=1, padding=1)

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img, input_high, input_low):
        # y = input[:,0:2]
        input1 = self.feature1(self.input_conv1(input_img))
        input2 = self.feature2(self.input_conv2(input_high))
        input3 = self.feature3(self.input_conv3(input_low))

        over_input = torch.cat([input1, input2, input3], dim=1)

        res = self.relu1(self.conv(over_input))

        output = self.recon_block(over_input)
        output = torch.add(output, res)

        output = self.recon(output)


        return output

class UNetDenseFrequencyVer6(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequencyVer6, self).__init__()

        self.input_conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu1 = nn.PReLU()
        self.feature1 = self.make_layer(UBlock, 128)

        self.input_conv2 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu2 = nn.PReLU()
        self.feature2 = self.make_layer(UBlock, 128)

        self.recon1 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self.recon2 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self._initialize_weights()

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img, input_low):
        # y = input[:,0:2]
        input1 = self.feature1(self.input_relu1(self.input_conv1(input_img)))
        input2 = self.feature2(self.input_relu2(self.input_conv2(input_img/3 + input_low*2/3)))

        output1 = self.recon1(input1)
        output2 = self.recon2(input2)

        output = (output1+output2)/2
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class UNetDenseFrequencyVer41(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequencyVer41, self).__init__()

        self.input_conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu1 = nn.PReLU()
        self.feature1 = self.make_layer(UBlock, 128)

        self.input_conv2 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu2 = nn.PReLU()
        self.feature2 = self.make_layer(UBlock, 128)

        self.recon1 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self.recon2 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self._initialize_weights()

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img, input_low):
        # y = input[:,0:2]
        input1 = self.feature1(self.input_relu1(self.input_conv1(input_img)))
        input2 = self.feature2(self.input_relu2(self.input_conv2(input_img)))

        out = self.recon1(input1)
        output1 = input_img-out
        output2 = self.recon2(input2)

        output = (output1+output2)/2
        return output
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)




class UNetDenseFrequencyVer4(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequencyVer4, self).__init__()

        self.input_conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu1 = nn.PReLU()
        self.feature1 = self.make_layer(UBlock, 128)

        self.input_conv2 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu2 = nn.PReLU()
        self.feature2 = self.make_layer(UBlock, 128)

        self.recon1 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self.recon2 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self._initialize_weights()

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img, input_low):
        # y = input[:,0:2]
        input1 = self.feature1(self.input_relu1(self.input_conv1(input_img)))
        input2 = self.feature2(self.input_relu2(self.input_conv2(input_img/3 + input_low*2/3)))

        out = self.recon1(input1)
        output1 = input_img-out
        output2 = self.recon2(input2)

        output = (output1+output2)/2
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)



class UNetDenseFrequencyVer42(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequencyVer42, self).__init__()

        self.input_conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu1 = nn.PReLU()
        self.feature1 = self.make_layer(UBlock, 128)

        self.input_conv2 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu2 = nn.PReLU()
        self.feature2 = self.make_layer(UBlock, 128)

        self.recon1 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self.recon2 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)
        self._initialize_weights()

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img, input_low, input_high):
        # y = input[:,0:2]
        input1 = self.feature1(self.input_relu1(self.input_conv1(input_img/3 + input_high*2/3)))
        input2 = self.feature2(self.input_relu2(self.input_conv2(input_img/3 + input_low*2/3)))

        out = self.recon1(input1)
        output1 = input_img-out
        output2 = self.recon2(input2)

        output = (output1+output2)/2
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class UNetDenseFrequencyVer3(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequencyVer3, self).__init__()

        self.relu = nn.PReLU()
        self.downsampling = nn.MaxPool2d(2,2)

        self.input_conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu1 = nn.PReLU()
        self.feature1 = self.make_layer(UBlock, 128)

        self.recon_block1 = self.make_layer(ReconBlock, 128)
        self.recon1 = nn.Conv2d(256, image_channels, 3, stride=1, padding=1)

        self.recon_block2 = self.make_layer(ReconBlock, 128)
        self.recon2 = nn.Conv2d(256, image_channels, 3, stride=1, padding=1)

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img):
        # y = input[:,0:2]
        input1 = self.feature1(self.input_relu1(self.input_conv1(input_img)))

        output1 = self.recon1(self.recon_block1(input1))

        output2 = self.recon2(self.recon_block2(input1))
        out2 = input_img-output2

        return (output1+out2)/2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class UNetDenseFrequencyVer5(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequencyVer5, self).__init__()

        self.relu = nn.PReLU()
        self.downsampling = nn.MaxPool2d(2,2)

        self.input_conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.input_relu1 = nn.PReLU()
        self.feature1 = self.make_layer(UBlock, 128)

        self.recon1 = nn.Conv2d(128, image_channels, 3, stride=1, padding=1)

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img):
        # y = input[:,0:2]
        input1 = self.feature1(self.input_relu1(self.input_conv1(input_img)))

        output1 = self.recon1(input1)

        return input_img-output1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class UNetDenseFrequency(nn.Module):
    def __init__(self, input_channels=1, image_channels=1):
        super(UNetDenseFrequency, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.relu = nn.PReLU()
        self.downsampling = nn.MaxPool2d(2,2)

        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            self.make_layer(ResBlock_noBN, 32, 8),
            nn.Conv2d(32, 32, 3, stride=1, padding=1)
        )

        kernel_size = 3
        padding = 1
        layers = []
        depth = 17
        n_channels = 64


        layers.append(
            nn.Conv2d(in_channels=96, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()



        self.input_conv = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        self.input_conv2 = nn.Conv2d(96, 32, 3, stride=1, padding=1)

        self.encode1_conv1 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_conv2 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_conv3 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_conv4 = self.make_layer(ResBlock_noBN,32, 3)
        self.encode1_bn = nn.BatchNorm2d(32, eps=0.0001, momentum=0.95)

        self.encode2_conv1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encode2_conv2 = self.make_layer(ResBlock_noBN,64, 4)
        self.encode2_conv3 = self.make_layer(ResBlock_noBN,64, 4)
        self.encode2_conv4 = self.make_layer(ResBlock_noBN,64, 4)
        self.encode2_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.95)

        self.encode3_conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encode3_conv2 = self.make_layer(ResBlock_noBN,128, 4)
        self.encode3_conv3 = self.make_layer(ResBlock_noBN,128, 4)
        self.encode3_conv4 = self.make_layer(ResBlock_noBN,128, 4)
        self.encode3_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.95)

        self.encode4_conv1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encode4_conv2 = self.make_layer(ResBlock_noBN,256, 4)
        self.encode4_conv3 = self.make_layer(ResBlock_noBN,256, 4)
        self.encode4_conv4 = self.make_layer(ResBlock_noBN,256, 4)
        self.encode4_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.95)

        self.upsample1 = UpSampling(384, 128)
        self.decode1_conv1 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_conv2 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_conv3 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_conv4 = self.make_layer(ResBlock_noBN,128, 4)
        self.decode1_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.95)

        self.upsample2 = UpSampling(192, 64)
        self.decode2_conv1 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_conv2 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_conv3 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_conv4 = self.make_layer(ResBlock_noBN,64, 4)
        self.decode2_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.95)

        self.upsample3 = UpSampling(96, 32)
        self.decode3_conv1 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_conv2 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_conv3 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_conv4 = self.make_layer(ResBlock_noBN,32, 4)
        self.decode3_bn = nn.BatchNorm2d(32, eps=0.0001, momentum=0.95)

        self.recon = nn.Conv2d(32, image_channels, 3, stride=1, padding=1)

        # self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(selfself, block, nchannel, n_layer):
        layers = []
        for _ in range(n_layer):
            layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, input_img, input_high, input_low):
        # y = input[:,0:2]
        input1 = input_img
        high = input_high
        low = input_low

        high = self.feature(high)
        low = self.feature(low)

        # output = self.sub_mean(input)
        output = self.relu(self.input_conv(input1))

        y = torch.cat([output, high, low], dim=1)

        output = self.relu(self.input_conv2(y))
        output = (self.relu(self.encode1_conv1(output)))
        output = (self.relu(self.encode1_conv2(output)))
        output = (self.relu(self.encode1_conv3(output)))
        output = (self.relu(self.encode1_conv4(output)))
        encode1 = output
        output = self.downsampling(output)

        output = (self.relu(self.encode2_conv1(output)))
        output = (self.relu(self.encode2_conv2(output)))
        output = (self.relu(self.encode2_conv3(output)))
        output = (self.relu(self.encode2_conv4(output)))
        encode2 = output
        output = self.downsampling(output)

        output = (self.relu(self.encode3_conv1(output)))
        output = (self.relu(self.encode3_conv2(output)))
        output = (self.relu(self.encode3_conv3(output)))
        output = (self.relu(self.encode3_conv4(output)))
        encode3 = output
        output = self.downsampling(output)

        output = (self.relu(self.encode4_conv1(output)))
        output = (self.relu(self.encode4_conv2(output)))
        output = (self.relu(self.encode4_conv3(output)))
        output = (self.relu(self.encode4_conv4(output)))
        output = self.downsampling(output)

        output = self.upsample1(output, encode3)
        output = (self.relu(self.decode1_conv1(output)))
        output = (self.relu(self.decode1_conv2(output)))
        output = (self.relu(self.decode1_conv3(output)))

        output = self.upsample2(output, encode2)
        output = (self.relu(self.decode2_conv1(output)))
        output = (self.relu(self.decode2_conv2(output)))
        output = (self.relu(self.decode2_conv3(output)))

        output = self.upsample3(output, encode1)
        output = (self.relu(self.decode3_conv1(output)))
        output = (self.relu(self.decode3_conv2(output)))
        output = (self.relu(self.decode3_conv3(output)))

        output = self.recon(output)

        output2 = self.dncnn(x)
        residual = input_img-output2

        return (output+residual)/2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
