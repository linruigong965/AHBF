

import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}
class AHBF(nn.Module):
    def __init__(self, inchannel, r=2,  L=64):
        super(AHBF, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel
        self.truinchannle=2*inchannel
        self.conv1 = nn.Conv2d(self.truinchannle, self.inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(self.inchannel)

        self.control_v1 = nn.Linear(self.inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
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
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1)
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1)
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit

class VGG(nn.Module):
    def __init__(self, num_classes=10, num_branches=3, aux=3, depth=16, dropout = 0.5):
        super(VGG, self).__init__()
        self.inplances = 64
        self.aux=aux
        self.num_branches = num_branches
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(inplace=True)            
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.fixplanes=self.inplances

        for i in range(num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layers(512, num_layer+i*self.aux))
            self.inplances= self.fixplanes

            setattr(self, 'classifier3_'+str(i), nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
            ))
        for i in range(num_branches-1):
            setattr(self, 'afm_' + str(i), AHBF(512))


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
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        featurelist=[]
        logitlist=[]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_3 = getattr(self,'layer3_0')(x)   # B x 512 x 1 x 1
        featurelist.append(x_3)

        x_3 = x_3.view(x_3.size(0), -1)     # B x 512
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        logitlist.append(x_3_1)
        for i in range(1, self.num_branches ):
            temp = getattr(self, 'layer3_'+str(i))(x)
            featurelist.append(temp)

            temp = temp.view(temp.size(0), -1)
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
            logitlist.append(temp_1)
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

def vgg16(pretrained=False, path=None, **kwargs):
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    
def vgg19(pretrained=False, path=None, **kwargs):
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
