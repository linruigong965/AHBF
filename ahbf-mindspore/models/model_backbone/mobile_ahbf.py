import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import math

__all__ = ['mobilenetv2_T_w', 'mobile_half','mobile_half_reduce']

BN = None

class AHBF(nn.Cell):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,pad_mode='pad', padding=0, stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Dense(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(axis=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def construct(self, x,y,logitx,logity):

        feasc = ops.Concat(axis=1)([x, y])
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line

        feas = self.pool(feasc)
        feas = feas.view(feas.shape[0], -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=self.relu(feas)
        feas = self.softmax(feas)
        # 待修改
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity
        return feasc,logit

def conv_bn(inp, oup, stride):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 3, stride, pad_mode='pad', padding=1, has_bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
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
            nn.BatchNorm2d(oup),
        )

    def construct(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 branches=4,
                 aux=2,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg
        self.aux=aux
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
        ]
        self.num_branches=branches
        # building first layer
        assert input_size % 32 == 0
        self.input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, self.input_channel, 2)

        # building inverted residual blocks
        self.blocks = nnCellList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(self.input_channel, output_channel, stride, t)
                )
                self.input_channel = output_channel
            self.blocks.append(nn.SequentialCell(layers))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.fixchannel=self.input_channel
        for i in range(self.num_branches):
            aux_c = i // 3  ###add auxiliary layers monotonously stage by stage   for optimization stability add lower layer first

            if i % 3 == 0:
                aux_a = i // 3
                aux_b = i // 3  ###add auxiliary layers monotonously stage by stage   for optimization stability add lower layer first
            else:
                aux_a = i // 3 + 1
                if i % 3 == 1:
                    aux_b = i // 3  ###add auxiliary layers monotonously stage by stage
                else:
                    aux_b = i // 3 + 1  ###add auxiliary layers monotonously stage by stage

            setattr(self, 'block4_' + str(i), self._make_layer_5([[T, 96, 3+aux_a*self.aux, 1]],width_mult))

            setattr(self, 'block5_' + str(i), self._make_layer_5([[T, 160, 3+aux_b*self.aux, 2]],width_mult))

            setattr(self, 'block6_' + str(i), self._make_layer_6([[T, 320, 1+aux_c*self.aux, 1]],width_mult))
            setattr(self, 'classifier_' +str(i), nn.SequentialCell(nn.Dense(self.last_channel, feature_dim),))
            self.input_channel=self.fixchannel
        for i in range(branches-1):
            setattr(self, 'afm_' + str(i), AHBF(self.last_channel))
        # building classifier
        H = input_size // (32//2)
        self.avgpool = nn.AvgPool2d(H)
        self._initialize_weights()

    def _make_layer_5(self,interverted_residual_setting,width_mult):
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(self.input_channel, output_channel, stride, t)
                )
                self.input_channel = output_channel
        return nn.SequentialCell(layers)
    def _make_layer_6(self,interverted_residual_setting,width_mult):
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(self.input_channel, output_channel, stride, t)
                )
                self.input_channel = output_channel
            last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
            conv2 = conv_1x1_bn(self.input_channel, last_channel)
            layers.append(conv2)
        return nn.SequentialCell(layers)

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

        fealist=[]
        logitlist=[]
        out = self.conv1(x)
        out = self.blocks[0](out)
        out = self.blocks[1](out)
        out = self.blocks[2](out)
        out = self.blocks[3](out)
        out_t = getattr(self, 'block4_' + str(0))(out)
        out_t = getattr(self, 'block5_' + str(0))(out_t)
        # print(out_t.shape)
        out_t = getattr(self, 'block6_' + str(0))(out_t)
        fealist.append(out_t)
        if not self.remove_avg:
            out_t = self.avgpool(out_t)
        out_t = out_t.view(out_t.shape[0], -1)
        out_t = getattr(self, 'classifier_' + str(0))(out_t)
        logitlist.append(out_t)
        for i in range(1, self.num_branches):
            temp = getattr(self, 'block4_' + str(0))(out)
            temp = getattr(self, 'block5_' + str(0))(temp)
            temp = getattr(self, 'block6_'+str(i))(temp)
            fealist.append(temp)
            temp = self.avgpool(temp)
            temp = temp.view(temp.shape[0], -1)
            temp_out = getattr(self, 'classifier_' + str(i ))(temp)
            logitlist.append(temp_out)

        ensem_fea = []
        ensem_logits = []

        for i in range(0,self.num_branches-1):
            if i==0:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(fealist[i],fealist[i+1],logitlist[i],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(ensem_fea[i-1],fealist[i+1],ensem_logits[i-1],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

        # if is_feat:
        #     return [f0, f1, f2, f3, f4, f5], out
        # else:
        return logitlist, ensem_logits

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(mindspore.Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        mindspore.numpy.zeros(m.bias.data.shape, dtype="float32"))


class MobileNetV2_d_2(nn.Cell):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 branches=4,
                 aux=1,
                 remove_avg=False):
        super(MobileNetV2_d_2, self).__init__()
        self.remove_avg = remove_avg
        self.aux=aux
        assert branches<6,'in this version branches cannot larger than 5'
        assert aux>1,'in this version aux cannot larger than 1'
        # print(self.aux)
        # assert aux%2==0 ,'aux must be oven'

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
        ]
        self.num_branches=branches
        # building first layer
        assert input_size % 32 == 0
        self.input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, self.input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.CellList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(self.input_channel, output_channel, stride, t)
                )
                self.input_channel = output_channel
            self.blocks.append(nn.SequentialCell(layers))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.fixchannel=self.input_channel
        for i in range(branches):
            if i%2==0:
                aux_a=i//2
                aux_b=i//2
            else:
                aux_a=i//2
                aux_b = i // 2+1
            setattr(self, 'block4_' + str(i), self._make_layer_pre([[T, 96, 3-aux*aux_a, 1]],width_mult))
            setattr(self, 'block5_' + str(i), self._make_layer_pre([[T, 160, 3-aux*aux_b, 2]],width_mult))

            setattr(self, 'block6_' + str(i), self._make_layer_6([[T, 320, 1, 1]],width_mult))
            setattr(self, 'classifier_' +str(i), nn.SequentialCell(nn.Dense(self.last_channel, feature_dim),))
            self.input_channel=self.fixchannel


        for i in range(branches-1):
            setattr(self, 'afm_' + str(i), AHBF(self.last_channel))
        # building classifier
        H = input_size // (32//2)
        self.avgpool = nn.AvgPool2d(H)

        self._initialize_weights()

    def _make_layer_pre(self,interverted_residual_setting,width_mult):
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(self.input_channel, output_channel, stride, t)
                )
                self.input_channel = output_channel
        return nn.SequentialCell(layers)

    def _make_layer_6(self,interverted_residual_setting,width_mult):
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(self.input_channel, output_channel, stride, t)
                )
                self.input_channel = output_channel
            last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
            conv2 = conv_1x1_bn(self.input_channel, last_channel)
            layers.append(conv2)
        return nn.SequentialCell(layers)

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

        fealist=[]
        logitlist=[]
        out = self.conv1(x)
        out = self.blocks[0](out)
        out = self.blocks[1](out)
        out = self.blocks[2](out)
        out = self.blocks[3](out)
        out_t = getattr(self, 'block4_' + str(0))(out)
        out_t = getattr(self, 'block5_' + str(0))(out_t)
        # print(out_t.shape)
        out_t = getattr(self, 'block6_' + str(0))(out_t)
        fealist.append(out_t)
        if not self.remove_avg:
            out_t = self.avgpool(out_t)
        out_t = out_t.view(out_t.shape[0], -1)
        out_t = getattr(self, 'classifier_' + str(0))(out_t)
        logitlist.append(out_t)
        for i in range(1, self.num_branches):
            temp = getattr(self, 'block4_' + str(0))(out)
            temp = getattr(self, 'block5_' + str(0))(temp)
            temp = getattr(self, 'block6_' + str(i))(temp)
            fealist.append(temp)
            temp = self.avgpool(temp)
            temp = temp.view(temp.shape[0], -1)
            temp_out = getattr(self, 'classifier_' + str(i))(temp)
            logitlist.append(temp_out)
        logitlist=logitlist[::-1]
        fealist=fealist[::-1]
        ensem_fea = []
        ensem_logits = []

        for i in range(0,self.num_branches-1):
            if i==0:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(fealist[i],fealist[i+1],logitlist[i],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(ensem_fea[i-1],fealist[i+1],ensem_logits[i-1],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

        # if is_feat:
        #     return [f0, f1, f2, f3, f4, f5], out
        # else:
        return logitlist, ensem_logits

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(mindspore.Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        mindspore.numpy.zeros(m.bias.data.shape, dtype="float32"))


def mobilenetv2_T_w(T, W,branch,aux, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W,branches=branch,aux=aux)
    return model

def mobilenetv2_T_w_d(T, W,branch,aux, feature_dim=100):
    model = MobileNetV2_d_2(T=T, feature_dim=feature_dim, width_mult=W,branches=branch,aux=aux)
    return model

def mobile_half(branch,aux,num_classes):
    return mobilenetv2_T_w(6, 0.5,branch,aux, num_classes)

def mobile_half_reduce(branch,aux,num_classes):
    return mobilenetv2_T_w_d(6, 0.5,branch,aux, num_classes)