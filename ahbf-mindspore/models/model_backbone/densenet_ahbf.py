import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import torch.utils.checkpoint as cp
from collections import OrderedDict

__all__ = ['DenseNet', 'densenetd40k12', 'densenetd100k12', 'densenetd100k40']


class AHBF(nn.Cell):
    def __init__(self, inchannel, aux, grow_rate, r=2, L=64):
        super(AHBF, self).__init__()
        d = max(int(inchannel * r), L)
        self.inchannel1 = inchannel
        self.inchannel2 = inchannel - aux * grow_rate
        self.real_in = self.inchannel1 + self.inchannel2

        self.conv1 = nn.Conv2d(self.real_in, self.inchannel1, kernel_size=1, pad_mode='valid', has_bias=True, weight_init=HeNormal(mode='fan_out', nonlinearity='relu'))
        self.bn1 = nn.BatchNorm2d(self.inchannel1)

        self.control_v1 = nn.Dense(self.inchannel1, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
        self.pool = ops.AdaptiveAvgPool2D(output_size=1)
        # self.classifier=nn.Linear(64 , 100)

    def construct(self, x, y, logitx, logity):
        feasc = ops.Concat(axis=1)([x, y])
        feasc = self.conv1(feasc)
        feasc = self.bn1(feasc)  ###8/5   1:02 add this line

        feas = self.pool(feasc)
        feas = feas.view(feas.shape(0), -1)

        feas = self.control_v1(feas)
        feas = self.bn_v1(feas)
        feas = self.relu(feas)
        feas = self.softmax(feas)

        # 待修改
        # x_c_1 = feas[:, 0].repeat(logitx.size()[1], 1).transpose(0, 1)
        # logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx

        # x_c_2 = feas[:, 1].repeat(logitx.size()[1], 1).transpose(0, 1)
        # logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit = x_c_1 * logitx + x_c_2 * logity

        return feasc, logit


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


# class ILR(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input, num_branches):
#         ctx.num_branches = num_branches
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         num_branches = ctx.num_branches
#         return grad_output / num_branches, None


class DenseNet(nn.Cell):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), num_branches=3, aux=0, bpscale=False, avg=False,
                 compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False, ind=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.num_branches = num_branches
        self.avg = avg
        self.ind = ind
        self.bpscale = bpscale



        ###### 
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i != len(block_config) - 1:
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            else:
                fix_layer = num_layers
                fix_feature = num_features
                for i in range(self.num_branches):
                    num_layers = num_layers + i * aux
                    block = _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        efficient=efficient,
                    )
                    setattr(self, 'layer3_' + str(i), block)
                    num_features = num_features + num_layers * growth_rate
                    setattr(self, 'norm_final_' + str(i), nn.BatchNorm2d(num_features))
                    setattr(self, 'relu_final_' + str(i), nn.ReLU(inplace=True))
                    # Linear layer
                    setattr(self, 'classifier3_' + str(i), nn.Linear(num_features, num_classes))
                    num_layers = fix_layer
                    num_features = fix_feature

        for i in range(1, self.num_branches):
            num_layers = num_layers + i * aux
            num_features = num_features + num_layers * growth_rate
            setattr(self, 'afm_' + str(i), AHBF(num_features, aux, growth_rate))
            num_layers = fix_layer
            num_features = fix_feature

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        if self.bpscale:
            self.layer_ILR = ILR.apply

    def forward(self, x):
        # For depth 40 growth_rate 1      B x 3 x 32 x 32
        featurelist = []
        logitlist = []
        x = self.features(x)  # B x 60 x 8 x 8
        x_3 = getattr(self, 'layer3_0')(x)  # B x 132 x 8 x 8
        x_3 = getattr(self, 'norm_final_0')(x_3)
        x_3 = getattr(self, 'relu_final_0')(x_3)
        featurelist.append(x_3)
        x_3 = self.avgpool(x_3).view(x_3.size(0), -1)  # B x 132
        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        logitlist.append(x_3_1)
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_' + str(i))(x)
            temp = getattr(self, 'norm_final_' + str(i))(temp)
            temp = getattr(self, 'relu_final_' + str(i))(temp)
            featurelist.append(temp)

            temp = self.avgpool(temp).view(temp.size(0), -1)  # B x 132
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)  # B x num_classes
            logitlist.append(temp_1)

        #  B x num_classes
        ensem_logits = []
        ensem_fea = []
        for i in range(0, self.num_branches - 1):
            if i == 0:
                ensembleff, logit = getattr(self, 'afm_' + str(i + 1))(featurelist[i], featurelist[i + 1], logitlist[i],
                                                                       logitlist[i + 1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff, logit = getattr(self, 'afm_' + str(i + 1))(ensem_fea[i - 1], featurelist[i + 1],
                                                                       ensem_logits[i - 1], logitlist[
                                                                           i + 1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

        return logitlist, ensem_logits


def densenetd40k12(pretrained=False, path=None, **kwargs):
    model = DenseNet(growth_rate=12, block_config=[6, 6, 6], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def densenetd100k12(pretrained=False, path=None, **kwargs):
    model = DenseNet(growth_rate=12, block_config=[16, 16, 16], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def densenetd100k40(pretrained=False, path=None, **kwargs):
    model = DenseNet(growth_rate=40, block_config=[16, 16, 16], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
