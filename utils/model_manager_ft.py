import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import timm
# from utils import load_state_dict_mute
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = 0.1
        m.inplace = True

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

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn == True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


# Define the swin_base_patch4_window7_224 Model
# pytorch > 1.6
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate=0.2)
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential()  # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)
        print(
            'Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')

    def forward(self, x):
        x = self.model.forward_features(x)
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        x = self.avgpool(x.permute((0, 2, 1)))
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


class ft_net_swinv2(nn.Module):

    def __init__(self, class_num, input_size=(256, 128), droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swinv2, self).__init__()
        model_ft = timm.create_model('swinv2_base_window8_256', pretrained=False, img_size=input_size,
                                     drop_path_rate=0.2)
        # model_full = timm.create_model('swinv2_base_window8_256', pretrained=True)
        # load_state_dict_mute(model_ft, model_full.state_dict(), strict=False)
        # model_ft = timm.create_model('swinv2_cr_small_224', pretrained=True, img_size = input_size, drop_path_rate = 0.2)
        # avg pooling to global pooling
        model_ft.head = nn.Sequential()  # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)
        print(
            'Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x.permute((0, 2, 1)))  # B * 1024 * WinNum
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


class ft_net_convnext(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_convnext, self).__init__()
        model_ft = timm.create_model('convnext_base', pretrained=True, drop_path_rate=0.2)
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential()  # save memory
        self.model = model_ft
        # self.model.apply(activate_drop)
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


# Define the HRNet18-based Model
class ft_net_hr(nn.Module):
    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w18', pretrained=True)
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential()  # save memory
        self.model = model_ft
        self.features = self.model.forward_features
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.features(x)  # [64,2048,8,4]
        x = self.avgpool(x)  # [64,2048,1,1]
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)  # [64,2048]
        prob = self.classifier(f)  # [64,2048]
        if not self.training:
            return f

        return prob, x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        self.model = model_ft
        self.circle = circle
        # For DenseNet, the feature dim is 1024
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


# Define the Efficient-b4-based Model
class ft_net_efficient(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        # model_ft = timm.create_model('tf_efficientnet_b4', pretrained=True)
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError:
            print('Please pip install efficientnet_pytorch')
        model_ft = EfficientNet.from_pretrained('efficientnet-b4')
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential()  # save memory
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.classifier = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        # For EfficientNet, the feature dim is not fixed
        # for efficientnet_b2 1408
        # for efficientnet_b4 1792
        self.bottleneck = nn.BatchNorm1d(1792)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(1792, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        # x = self.model.forward_features(x)
        x = self.model.extract_features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, linear_num=512):
        super().__init__()
        model_name = 'nasnetalarge'
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.bottleneck = nn.BatchNorm1d(4032)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(4032, class_num, droprate, linear=linear_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num=751, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        if not self.training:
            x = x.unsqueeze(0)

        f = self.bottleneck(x)
        prob = self.classifier(f)

        if not self.training:
            return f

        return prob, x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, linear=256, relu=False, bnorm=True))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i].view(x.size(0), x.size(1))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class ResNet50(nn.Module):
    def __init__(self, class_num = 9449):
        super(ResNet50, self).__init__()


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


__factory = {
    'resnet50':ft_net,
    'swin':ft_net_swin,
    'swinv2':ft_net_swinv2,
    'convnext':ft_net_convnext,
    'hr':ft_net_hr,
    'dense':ft_net_dense,
    'efficient':ft_net_efficient,
    'NAS':ft_net_NAS,
    'middle':ft_net_middle,
    'pcb':PCB,
    'resnet':ResNet50,
}

def init_model(name,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Invalid model, expected to be one of {}".format(__factory.keys()))
    return __factory[name](**kwargs)