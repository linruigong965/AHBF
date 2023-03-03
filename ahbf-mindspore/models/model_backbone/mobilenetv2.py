import numpy as np
import mindspore
import mindspore.nn as nn
import math

__all__ = ['mobilenetv2_T_w', 'mobile_half','mobile']

BN = None


def conv_bn(inp, oup, stride):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 3, stride, pad_mode='pad', padding=1, has_bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 1, 1, pad_mode='pad', padding=0, has_bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.SequentialCell(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, pad_mode='pad', padding=1, group=inp * expand_ratio, has_bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(oup)
        )

    def construct(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Cell):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.CellList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.SequentialCell(layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        # building classifier
        # self.classifier = nn.SequentialCell(
        #     # nn.Dropout(0.5),
        #     nn.Dense(self.last_channel, feature_dim)
        # )
        self.classifier = nn.Dense(self.last_channel, feature_dim)

        H = input_size // (32//2)
        self.avgpool = nn.AvgPool2d(H)
        self._initialize_weights()

    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.CellList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def construct(self, x, is_feat=False, preact=False):

        out = self.conv1(x)
        f0 = out

        out = self.blocks[0](out)
        out = self.blocks[1](out)
        f1 = out
        out = self.blocks[2](out)
        f2 = out
        out = self.blocks[3](out)
        out = self.blocks[4](out)
        f3 = out
        out = self.blocks[5](out)
        out = self.blocks[6](out)
        f4 = out

        out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        f5 = out
        print(out.shape)
        out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out
        else:
            return out

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(mindspore.Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        mindspore.numpy.zeros(m.bias.data.shape, dtype="float32"))


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def mobile_half(num_classes):
    return mobilenetv2_T_w(6, 0.5, num_classes)
def mobile(num_classes):
    return mobilenetv2_T_w(6, 1, num_classes)
    