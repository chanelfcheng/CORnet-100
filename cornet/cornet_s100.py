import math
from collections import OrderedDict
from torch import nn


HASH = '1d3f7975'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1, V4_to_V1=False):
        super().__init__()

        self.times = times
        self.V4_to_V1 = V4_to_V1

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0 and not self.V4_to_V1:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            # # if the spatial dimensions of x and skip do not match, pool skip
            # if x.shape[2:] != skip.shape[2:]:
            #     adapter = nn.AdaptiveAvgPool2d(x.shape[2:])
            #     skip = adapter(skip)
            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnetSModel(nn.Module):
    def __init__(self, V4_to_V1_times=1):
        super().__init__()

        # V1 block
        self.V1_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.V1_norm1 = nn.BatchNorm2d(64)
        self.V1_nonlin1 = nn.ReLU(inplace=True)
        self.V1_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.V1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.V1_norm2 = nn.BatchNorm2d(64)
        self.V1_nonlin2 = nn.ReLU(inplace=True)
        self.V1_output = Identity()  # Assuming Identity is defined elsewhere

        # V2, V4, IT blocks (assuming CORblock_S is defined elsewhere)
        self.V2 = CORblock_S(64, 128, times=2)
        self.V4 = CORblock_S(128, 256, times=4)
        self.IT = CORblock_S(256, 512, times=2)

        # Decoder block
        self.decoder_avgpool = nn.AdaptiveAvgPool2d(1)
        self.decoder_flatten = Flatten()  # Assuming Flatten is defined elsewhere
        self.decoder_linear = nn.Linear(512, 100)
        self.decoder_output = Identity()  # Assuming Identity is defined elsewhere
        
        # FEEDBACK
        self.V4_to_V1 = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=1, bias=False)
        self.V4_to_V1_times = V4_to_V1_times

    def forward(self, x):
        
        # Recurrent connection from V4 to V1
        for t in range(self.V4_to_V1_times):
            if t == 0:
                self.V1_conv1.stride = (2, 2)
                self.V1_pool.stride = (2, 2)
                
                self.V2.skip.stride = (2, 2)
                self.V2.conv2.stride = (2, 2)
                self.V2.V4_to_V1 = False
                
                self.V4.skip.stride = (2, 2)
                self.V4.conv2.stride = (2, 2)
                self.V4.V4_to_V1 = False
            else:
                self.V1_conv1.stride = (1, 1)
                self.V1_pool.stride = (1, 1)
                
                self.V2.skip.stride = (1, 1)
                self.V2.conv2.stride = (1, 1)
                self.V2.V4_to_V1 = True
                
                self.V4.skip.stride = (1, 1)
                self.V4.conv2.stride = (1, 1)
                self.V4.V4_to_V1 = True
                
            # V1 block
            # print("Before V1:", x.shape)
            # ensure that x has 3 channels only before passing it to V1
            if x.shape[1] > 3:
                x = self.V4_to_V1(x)
            x = self.V1_conv1(x)
            x = self.V1_norm1(x)
            x = self.V1_nonlin1(x)
            x = self.V1_pool(x)
            x = self.V1_conv2(x)
            x = self.V1_norm2(x)
            x = self.V1_nonlin2(x)
            x = self.V1_output(x)

            # V2 block
            # print("Before V2:", x.shape)
            x = self.V2(x)

            # V4 block
            # print("Before V4:", x.shape)
            x = self.V4(x)

        # IT block
        # print("Before IT:", x.shape)
        x = self.IT(x)

        # Decoder block
        x = self.decoder_avgpool(x)
        x = self.decoder_flatten(x)
        x = self.decoder_linear(x)
        x = self.decoder_output(x)

        return x


def CORnet_S100(V4_to_V1_times=1):
    print("V4_to_V1_times:", V4_to_V1_times)
    model = CORnetSModel(V4_to_V1_times=V4_to_V1_times)

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot 
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model