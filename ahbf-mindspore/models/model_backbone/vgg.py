import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import HeNormal

__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

class VGG(nn.Cell):
    def __init__(self, num_classes=10, depth=16, dropout = 0.0, KD= False):
        super(VGG, self).__init__()
        self.KD = KD
        self.inplances = 64
        # self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, pad_mode='pad', padding=1, has_bias=True, weight_init=HeNormal(mode='fan_out', nonlinearity='relu'))
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        # self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, pad_mode='pad', padding=1, has_bias=True, weight_init=HeNormal(mode='fan_out', nonlinearity='relu'))
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.layer4 = self._make_layers(512, num_layer)
        
        self.classifier = nn.SequentialCell(
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dropout(1 - dropout),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dropout(1 - dropout),
            nn.Dense(512, num_classes),
        )
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            # conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, pad_mode='pad', padding=1, has_bias=True, weight_init=HeNormal(mode='fan_out', nonlinearity='relu'))
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU()]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")]
        return nn.SequentialCell(*layers)
    
    def construct(self, x):
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
        x = self.layer4(x)
        
        x_f = x.view(x.shape[0], -1)
        x = self.classifier(x_f)
        if self.KD:
            return x_f, x
        else:
            return x
    
def vgg16(pretrained=False, path=None, **kwargs):
    model = VGG(depth=16, **kwargs)
    if pretrained:
        mindspore.load_param_into_net(model, mindspore.load_checkpoint(path))
    return model
    
def vgg19(pretrained=False, path=None, **kwargs):
    model = VGG(depth=19, **kwargs)
    if pretrained:
        mindspore.load_param_into_net(model, mindspore.load_checkpoint(path))
    return model
    