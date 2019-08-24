import torch
from torch import nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a

class Encoder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Encoder, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        elif pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return global_feat, cls_score
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                cls_score = self.classifier(feat)
                return feat, cls_score
            else:
                # print("Test with feature before BN")
                cls_score = self.classifier(feat)
                return global_feat, cls_score

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class DFDGenerator(nn.Module):
    def __init__(self, model_path, pretrain_choice, output_nc=3, domain_feature_size=512, reid_feature_size=2048, noise_size=512, dropout=0.0):
        super(DFDGenerator, self).__init__()
        self.dropout = dropout
        self.noise_size = noise_size

        # N*(domain_feature_size+reid_feature_size+noise_size)*1*1
        self.relu1 = nn.ReLU(True)
        self.trans_conv1 = nn.ConvTranspose2d(domain_feature_size + reid_feature_size + noise_size, 1024,
                                              kernel_size=(8, 4))
        # N*1024*8*4
        self.bn1 = nn.BatchNorm2d(1024)
        self.dropout1 = nn.Dropout(self.dropout)
        self.layer1 = self._make_layer(1024, 512)
        # N*512*16*8
        self.layer2 = self._make_layer(512, 256)
        # N*256*32*16
        self.layer3 = self._make_layer(256, 128)
        # N*128*64*32
        self.layer4 = self._make_layer(128, 64)
        # N*64*128*64
        self.relu2 = nn.ReLU(True)
        self.trans_conv2 = nn.ConvTranspose2d(64, output_nc, kernel_size=4, stride=2, padding=1)
        self.tanh1 = nn.Tanh()
        # N*3*256*128

        if pretrain_choice == 'self':
            self.load_param(model_path)

    def _make_layer(self, input_nc, output_nc):
        block = [nn.ReLU(True),
                 nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),
                 nn.BatchNorm2d(output_nc),
                 nn.Dropout(self.dropout)]
        return nn.Sequential(*block)

    def forward(self, reid_feat, domain_feat):
        noise = torch.randn(reid_feat.size(0), self.noise_size).cuda()
        feat = torch.cat((reid_feat, domain_feat, noise), dim=1)
        x = feat.view(feat.size(0), feat.size(1), 1, 1)
        x = self.relu1(x)
        x = self.trans_conv1(x)  # 1024,8,4
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.layer1(x)  # 512,16,8
        x = self.layer2(x)  # 256,32,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu2(x)
        x = self.trans_conv2(x)
        fake_img = self.tanh1(x)

        return fake_img

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])

class Discriminator(nn.Module):
    def __init__(self, last_stride, model_path, neck, model_name, pretrain_choice, dist_func='square'):
        super(Discriminator, self).__init__()
        in_planes = 2048
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet18':
            in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])

        self.dist_func = dist_func
        self.neck = neck
        if dist_func not in ['square', 'abs']:
            raise KeyError("Unknown dist_func:", dist_func)
        self.gap = nn.AdaptiveAvgPool2d(1)
        if self.neck == 'no':
            self.classifier = nn.Linear(in_planes, 1)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(in_planes, 1, bias=False)
            self.bottleneck.apply(weights_init_kaiming)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        elif pretrain_choice == 'self':
            self.load_param(model_path)

        self.classifier.apply(weights_init_classifier)

    def forward(self, x1, x2):
        x1_feature, x2_feature = self.gap(self.base(x1)), self.gap(self.base(x2))
        x1_feature = x1_feature.view(x1_feature.shape[0], -1)  # flatten to (bs, 2048)
        x2_feature = x2_feature.view(x2_feature.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck == 'bnneck':
            x1_feature, x2_feature  = self.bottleneck(x1_feature), self.bottleneck(x2_feature)
        x = x1_feature - x2_feature
        if self.dist_func == 'square':
            x = x.pow(2)
        elif self.dist_func == 'abs':
            x = x.abs()
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
# class IdDiscriminator(nn.Module):
#     def __init__(self, last_stride, model_path, neck, model_name, pretrain_choice, dist_func='square'):
#         super(IdDiscriminator, self).__init__()
#         in_planes = 2048
#         if model_name == 'resnet50':
#             self.base = ResNet(last_stride=last_stride,
#                                block=Bottleneck,
#                                layers=[3, 4, 6, 3])
#
#         self.dist_func = dist_func
#         self.neck = neck
#         if dist_func not in ['square', 'abs']:
#             raise KeyError("Unknown dist_func:", dist_func)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         if self.neck == 'no':
#             self.classifier = nn.Linear(in_planes, 1)
#         elif self.neck == 'bnneck':
#             self.bottleneck = nn.BatchNorm1d(in_planes)
#             self.bottleneck.bias.requires_grad_(False)  # no shift
#             self.classifier = nn.Linear(in_planes, 1, bias=False)
#             self.bottleneck.apply(weights_init_kaiming)
#
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......')
#         elif pretrain_choice == 'self':
#             self.load_param(model_path)
#
#         self.classifier.apply(weights_init_classifier)
#
#     def forward(self, x1, x2):
#         x1_feature, x2_feature = self.gap(self.base(x1)), self.gap(self.base(x2))
#         if self.neck == 'bnneck':
#             x1_feature, x2_feature  = self.bottleneck(x1_feature), self.bottleneck(x2_feature)
#         x = x1_feature - x2_feature
#         if self.dist_func == 'square':
#             x = x.pow(2)
#         elif self.dist_func == 'abs':
#             x = x.abs()
#         x = self.classifier(x)
#         x = torch.sigmoid(x)
#         return x
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             if 'classifier' in i:
#                 continue
#             self.state_dict()[i].copy_(param_dict[i])
#
# class DomainDiscriminator(nn.Module):
#     def __init__(self, last_stride, model_path, neck, model_name, pretrain_choice, dist_func='square'):
#         super(DomainDiscriminator, self).__init__()
#         in_planes = 2048
#         if model_name == 'resnet18':
#             in_planes = 512
#             self.base = ResNet(last_stride=last_stride,
#                                block=BasicBlock,
#                                layers=[2, 2, 2, 2])
#
#         self.dist_func = dist_func
#         self.neck = neck
#         if dist_func not in ['square', 'abs']:
#             raise KeyError("Unknown dist_func:", dist_func)
#
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         if self.neck == 'no':
#             self.classifier = nn.Linear(in_planes, 1)
#         elif self.neck == 'bnneck':
#             self.bottleneck = nn.BatchNorm1d(in_planes)
#             self.bottleneck.bias.requires_grad_(False)  # no shift
#             self.classifier = nn.Linear(in_planes, 1, bias=False)
#             self.bottleneck.apply(weights_init_kaiming)
#
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......')
#         elif pretrain_choice == 'self':
#             self.load_param(model_path)
#         self.classifier.apply(weights_init_classifier)
#
#
#     def forward(self, x1, x2):
#         x1_feature, x2_feature = self.gap(self.base(x1)), self.gap(self.base(x2))
#         if self.neck == 'bnneck':
#             x1_feature, x2_feature = self.bottleneck(x1_feature), self.bottleneck(x2_feature)
#         x = x1_feature - x2_feature
#         if self.dist_func == 'square':
#             x = x.pow(2)
#         elif self.dist_func == 'abs':
#             x = x.abs()
#         x = self.classifier(x)
#         x = torch.sigmoid(x)
#         return x
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             if 'classifier' in i:
#                 continue
#             self.state_dict()[i].copy_(param_dict[i])
#

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
