import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from collections import OrderedDict

__all__ = ['DenseNet', 'densenetd40k12', 'densenetd100k12', 'densenetd100k40', 'densenetd190k12']


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = ops.Concat(axis=1)(inputs)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Cell):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.insert_child_to_cell('norm1', nn.BatchNorm2d(num_input_features))
        self.insert_child_to_cell('relu1', nn.ReLU())
        self.insert_child_to_cell('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, pad_mode='valid', weight_init=HeNormal()))
        self.insert_child_to_cell('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.insert_child_to_cell('relu2', nn.ReLU())
        self.insert_child_to_cell('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, pad_mode='pad', padding=1, weight_init=HeNormal()))
        self.dropout = nn.Dropout(1 - drop_rate)
        self.drop_rate = drop_rate
        self.efficient = efficient

    def construct(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            pass
            # bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)
        return new_features


class _Transition(nn.SequentialCell):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.insert_child_to_cell('norm', nn.BatchNorm2d(num_input_features))
        self.insert_child_to_cell('relu', nn.ReLU())
        self.insert_child_to_cell('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, pad_mode='valid', weight_init=HeNormal()))
        self.insert_child_to_cell('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Cell):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.insert_child_to_cell('denselayer%d' % (i + 1), layer)

    def construct(self, init_features):
        features = [init_features]
        for name in self.name_cells().keys():
            new_features = self[name](*features)
            features.append(new_features)
        return ops.Concat(axis = 1)(features)


class DenseNet(nn.Cell):
    def __init__(self, growth_rate=12, block_config=[16, 16, 16], compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False, KD=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.KD = KD
        # First convolution
        if small_inputs:
            self.features = nn.SequentialCell(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, pad_mode='pad', padding=1, weight_init=HeNormal())),
            ]))
        else:
            self.features = nn.SequentialCell(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, pad_mode='pad', padding=3, weight_init=HeNormal())),
            ]))
            self.features.insert_child_to_cell('norm0', nn.BatchNorm2d(num_init_features))
            self.features.insert_child_to_cell('relu0', nn.ReLU())
            self.features.insert_child_to_cell('pool0', nn.MaxPool2d(kernel_size=3, stride=))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.insert_child_to_cell('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.insert_child_to_cell('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.insert_child_to_cell('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Dense(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        # D40K12    B x 132 x 8 x 8
        # D100K12   B x 342 x 8 x 8
        # D100K40   B x 1126 x 8 x 8
        x = ops.ReLU()(features)
        x_f = ops.AvgPool(kernel_size=self.avgpool_size)(x).view(features.size(0), -1)  # B x 132
        x = self.classifier(x_f)
        if self.KD == True:
            return x_f, x
        else:
            return x


def densenetd40k12(pretrained=False, path=None, **kwargs):
    model = DenseNet(growth_rate=12, block_config=[6, 6, 6], **kwargs)
    if pretrained:
        mindspore.load_param_into_net(model, mindspore.load_checkpoint(path))
    return model


def densenetd100k12(pretrained=False, path=None, **kwargs):
    model = DenseNet(growth_rate=12, block_config=[16, 16, 16], **kwargs)
    if pretrained:
        mindspore.load_param_into_net(model, mindspore.load_checkpoint(path))
    return model


def densenetd190k12(pretrained=False, path=None, **kwargs):
    model = DenseNet(growth_rate=12, block_config=[31, 31, 31], **kwargs)
    if pretrained:
        mindspore.load_param_into_net(model, mindspore.load_checkpoint(path))
    return model


def densenetd100k40(pretrained=False, path=None, **kwargs):
    model = DenseNet(growth_rate=40, block_config=[16, 16, 16], **kwargs)
    if pretrained:
        mindspore.load_param_into_net(model, mindspore.load_checkpoint(path))
    return model
