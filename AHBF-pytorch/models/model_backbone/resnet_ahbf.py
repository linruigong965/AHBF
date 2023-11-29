

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet','ResNet_v2','resnet18', 'resnet32','resnet32_d','resnet32_p','resnet34', 'resnet50','resnet110']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class NonLocalBlockND(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlockND, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class AHBF(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
    def forward(self, x,y,logitx,logity):

        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line

        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit
class AHBF_cbam(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF_cbam, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)


        self.cbam=CBAM(64)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
    def forward(self, x,y,logitx,logity):

        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line
        feasc=self.cbam(feasc)
        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit
class AHBF_non(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF_non, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)


        self.non = NonLocalBlockND(64)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
    def forward(self, x,y,logitx,logity):

        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line
        feasc=self.non(feasc)
        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit
class AHBF_se(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF_se, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)


        self.se=SE_Block(64)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
    def forward(self, x,y,logitx,logity):

        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line
        feasc=self.se(feasc)
        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches=3, aux=0, type='conv',zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_branches = num_branches

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        fix_inplanes = self.inplanes  # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), \
            self._make_layer(block, 64, layers[2] + i * aux, stride=2))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier3_' + str(i), \
                    nn.Linear(64 * block.expansion, num_classes))
        if type == 'conv':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF_f(64 * block.expansion))
        elif type == 'se':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF_se(64 * block.expansion))
        elif type == 'nonlocal':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF_non(64 * block.expansion))
        elif type == 'cbam':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF_cbam(64 * block.expansion))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('*'*50)
        # score_list=[]
        featurelist = []
        featurelist1 = []
        logitlist = []
        # print(x.shape,'xxxx')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x_3 = getattr(self, 'layer3_0')(x)  # B x 64 x 8 x 8
        # print(x_3.shape,'x_3')
        featurelist.append(x_3)
        x_3 = self.avgpool(x_3)  # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)  # B x 64
        # featurelist1.append(x_3)

        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        logitlist.append(x_3_1)

        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_' + str(i))(x)
            # print(temp.shape,'temp')
            featurelist.append(temp)

            temp = self.avgpool(temp)  # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            # featurelist1.append(temp)
            temp_out = getattr(self, 'classifier3_' + str(i))(temp)
            logitlist.append(temp_out)

        ensem_fea = []
        ensem_logits = []

        for i in range(0, self.num_branches - 1):
            if i == 0:
                ensembleff, logit = getattr(self, 'afm_' + str(i))(featurelist[i], featurelist[i + 1], logitlist[i],
                                                                   logitlist[i + 1],i+2)
                # score_list.append(sco_list)
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff, logit = getattr(self, 'afm_' + str(i))(ensem_fea[i - 1], featurelist[i + 1],
                                                                   ensem_logits[i - 1], logitlist[
                                                                       i + 1],i+2)
                # score_list.append(sco_list)
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

        return logitlist, ensem_logits

        # return logitlist, ensem_logits, featurelist



class ResNet_ceonly(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches=3, aux=0, type='conv', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet_ceonly, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_branches = num_branches

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        fix_inplanes = self.inplanes  # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2] + i * aux, stride=2))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier3_' + str(i), nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        featurelist = []
        featurelist1 = []
        logitlist = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x_3 = getattr(self, 'layer3_0')(x)  # B x 64 x 8 x 8
        featurelist.append(x_3)
        x_3 = self.avgpool(x_3)  # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)  # B x 64
        featurelist1.append(x_3)

        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        logitlist.append(x_3_1)

        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_' + str(i))(x)
            featurelist.append(temp)

            temp = self.avgpool(temp)  # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            featurelist1.append(temp)
            temp_out = getattr(self, 'classifier3_' + str(i))(temp)
            logitlist.append(temp_out)


        return logitlist


class ResNet_reduce(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches = 3,aux=0,   zero_init_residual=False,
        groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet_reduce, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_branches = num_branches

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        fix_inplanes=self.inplanes    # 32
        for i in range(num_branches):
            if i!=num_branches-1:
                setattr(self, 'layer2_' + str(i), self._make_layer(block, 32, layers[1]-aux*(i>1), stride=2))
                setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2]-aux*(i>0), stride=2))
                self.inplanes = fix_inplanes  ##reuse self.inplanes
                setattr(self, 'classifier3_' +str(i), nn.Linear(64 * block.expansion, num_classes))
            else:
                setattr(self, 'layer2_' + str(i), self._make_layer(block, 32, layers[1]-aux, stride=2))
                setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2]-aux*2, stride=2))
                self.inplanes = fix_inplanes  ##reuse self.inplanes
                setattr(self, 'classifier3_' +str(i), nn.Linear(64 * block.expansion, num_classes))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(num_branches-1):
            setattr(self, 'afm_' + str(i), AHBF(64*block.expansion))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        featurelist=[]
        logitlist=[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)            # B x 16 x 32 x 32

        x = self.layer1(x)          # B x 16 x 32 x 32
        x_2 = getattr(self,'layer2_0')(x)   # B x 64 x 8 x 8
        x_3 = getattr(self,'layer3_0')(x_2)   # B x 64 x 8 x 8
        featurelist.append(x_3)
        x_3 = self.avgpool(x_3)             # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        logitlist.append(x_3_1)
        for i in range(1, self.num_branches ):
            temp = getattr(self, 'layer2_'+str(i))(x)
            temp = getattr(self, 'layer3_'+str(i))(temp)
            featurelist.append(temp)

            temp = self.avgpool(temp)       # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            temp_out = getattr(self, 'classifier3_' + str(i ))(temp)
            logitlist.append(temp_out)
        logitlist=logitlist[::-1]
        featurelist=featurelist[::-1]
        ensem_fea = []
        ensem_logits = []

        for i in range(0,self.num_branches-1):
            if i==0:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(featurelist[i],featurelist[i+1],logitlist[i],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(ensem_fea[i-1],featurelist[i+1],ensem_logits[i-1],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)



        return logitlist, ensem_logits

class ResNet_v2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_branches = 3,aux=0,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, KD = False):
        super(ResNet_v2, self).__init__()
        self.aux=aux
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_branches=num_branches
        self.KD = KD
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        fix_inplanes=self.inplanes    # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(num_branches):
            if i%2==0:
                aux_a=i//2
                aux_b=i//2  ###add auxiliary layers monotonously stage by stage   for optimization stability add lower layer first

            else:
                aux_a=i//2+1
                aux_b=i//2  ###add auxiliary layers monotonously stage by stage


            setattr(self, 'layer3_' + str(i), self._make_layer(block, 256, layers[2]+self.aux*(aux_a), stride=2,
                                       dilate=replace_stride_with_dilation[1]))
            setattr(self, 'layer4_' + str(i), self._make_layer(block, 512, layers[3]+self.aux*(aux_b), stride=2,
                                       dilate=replace_stride_with_dilation[2]))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier4_' +str(i), nn.Linear(512 * block.expansion, num_classes))

        for i in range(num_branches-1):
            setattr(self, 'afm_' + str(i), AHBF(512*block.expansion))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        featurelist=[]
        logitlist=[]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
        x_4 = getattr(self,'layer4_0')(x_3)
        featurelist.append(x_4)
        x_4 = self.avgpool(x_4)             # B x 64 x 1 x 1
        x_4 = x_4.view(x_4.size(0), -1)     # B x 64
        x_4_1 = getattr(self, 'classifier4_0')(x_4)     # B x num_classes
        logitlist.append(x_4_1)
        for i in range(1, self.num_branches ):
            x_3 = getattr(self, 'layer3_'+str(i))(x)  # B x 64 x 8 x 8
            x_4 = getattr(self, 'layer4_'+str(i))(x_3)
            featurelist.append(x_4)
            x_4 = self.avgpool(x_4)  # B x 64 x 1 x 1
            x_4 = x_4.view(x_4.size(0), -1)  # B x 64
            x_4_1 = getattr(self, 'classifier4_'+str(i))(x_4)  # B x num_classes
            logitlist.append(x_4_1)


        ensem_fea = []
        ensem_logits = []

        for i in range(0,self.num_branches-1):
            if i==0:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(featurelist[i],featurelist[i+1],logitlist[i],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(ensem_fea[i-1],featurelist[i+1],ensem_logits[i-1],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

        return logitlist, ensem_logits

def resnet32(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def resnet32_p(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet_ceonly(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
def resnet32_d(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet_reduce(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def resnet110(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(Bottleneck, [12, 12, 12], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model



def resnet18(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet_v2(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def resnet34(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet_v2(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def resnet50(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet_v2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model





