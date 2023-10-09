import sys
import torchvision.models as model
import torch.nn as nn
from timm.models.ghostnet import ghostnet_050
from timm.models.ghostnet import ghostnet_100
from timm.models.ghostnet import ghostnet_130
# from mnasnet import mnasnet_a1

def modify_output(args, net):
    """替换传统分类网络的分类头用于分心驾驶监测

    Args:
        args (_type_): _description_
        net (_type_): _description_
    """
    classifier_zoo = ['dense121', 'dense161', 'dense201', 'ghost1_0']
    vgg_zoo = ['vgg19bn','vgg16bn']
    Inception_zoo = ['inceptionv3', 'inceptionv4']
    mobile_zoo = [ 'mobilenet_v2', 'mobilenetv3_large', 'mobilenetv3_small','efficientnet_b0']
    squeeze_zoo = ['squeezenet1_0', 'squeezenet1_1']
    if args.net in classifier_zoo:
        channel_in = net.classifier.in_features
        net.classifier = nn.Linear(channel_in, args.num_class)
    elif args.net in mobile_zoo:
#        print('0000000000')
#        print(net.classifier[-1])
        channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(channel, args.num_class)
    elif args.net in Inception_zoo :
        channel_in = net.linear.in_features
        net.fc = nn.Linear(channel_in, args.num_class)
    elif args.net in vgg_zoo:
        net.classifier[-1] = nn.Linear(4096, args.num_class)
    elif args.net in squeeze_zoo:
        net.classifier[1] = nn.Conv2d(512, args.num_class, kernel_size=1)
    else:
        channel_in = net.fc.in_features
        net.fc = nn.Linear(channel_in, args.num_class)
    return net


def MobileNetV3_Small_Pretrain(opt):
    net = model.mobilenet_v3_small(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(channel, opt.classes)
    net.classifier.requires_grad_(True)
    return net

def MobileNetV3_Small_Pretrain_ScoreLoss(opt, level=5):
    net = model.mobilenet_v3_small(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(channel, opt.classes*level)
    net.classifier.requires_grad_(True)
    return net, level

def MobileNetV3_Large_Pretrain_ScoreLoss(opt, level=5):
    net = model.mobilenet_v3_large(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(channel, opt.classes*level)
    net.classifier.requires_grad_(True)
    return net, level

def Resnet18_Pretrain(opt):
    net = model.resnet18(pretrained=True)
    net.requires_grad_(False)
    channel = net.fc.in_features
    net.fc = nn.Linear(channel, opt.classes)
    net.fc.requires_grad_(True)
    return net

def Resnet18_Pretrain_ScoreLoss(opt, level=5):
    net = model.resnet18(pretrained=True)
    net.requires_grad_(False)
    channel = net.fc.in_features
    net.fc = nn.Linear(channel, opt.classes*level)
    net.fc.requires_grad_(True)
    return net, level

def Resnet50_Pretrain_ScoreLoss(opt, level=5):
    net = model.resnet50(pretrained=True)
    net.requires_grad_(False)
    channel = net.fc.in_features
    net.fc = nn.Linear(channel, opt.classes*level)
    net.fc.requires_grad_(True)
    return net, level

def ShuffleNetV2_Pretrain_ScoreLoss(opt, level=5):
    net = model.shufflenet_v2_x1_0(pretrained=True)
    net.requires_grad_(False)
    channel = net.fc.in_features
    net.fc = nn.Linear(channel, opt.classes*level)
    net.fc.requires_grad_(True)
    return net, level

def SqueezeNet_Pretrain_ScoreLoss(opt, level=5):
    net = model.squeezenet1_0(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier[-3].in_channels
    net.classifier[-3].out_channels = opt.classes*level
    net.classifier[-3] = nn.Conv2d(in_channels=channel, out_channels=opt.classes*level, kernel_size=(1, 1), stride=(1, 1))
    net.classifier[-3].requires_grad_(True)
    return net, level

def EfficientNetB0_Pretrain_ScoreLoss(opt, level=5):
    net = model.efficientnet_b0(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(channel, opt.classes*level)
    net.classifier[-1].requires_grad_(True)
    return net, level

def GhostNet_050_Pretrain(opt):
    net = ghostnet_050(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier.in_features
    net.classifier = nn.Linear(channel, opt.classes)
    net.classifier.requires_grad_(True)
    return net

def GhostNet_050_Pretrain_ScoreLoss(opt, level=5):
    net = ghostnet_050(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier.in_features
    net.classifier = nn.Linear(channel, opt.classes*level)
    net.classifier.requires_grad_(True)
    return net, level

def GhostNet_100_Pretrain(opt):
    net = ghostnet_100(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier.in_features
    net.classifier = nn.Linear(channel, opt.classes)
    net.classifier.requires_grad_(True)
    return net

def GhostNet_100_Pretrain_ScoreLoss(opt, level=5):
    net = ghostnet_100(pretrained=True)
    net.requires_grad_(False)
    channel = net.classifier.in_features
    net.classifier = nn.Linear(channel, opt.classes*level)
    net.classifier.requires_grad_(True)
    return net, level

# def MnasNet_A1_Pretrain(opt):
#     net = mnasnet_a1(pretrained=True)
#     net.requires_grad_(False)
#     channel = net.output.in_features
#     net.output = nn.Linear(channel, opt.classes)
#     net.output.requires_grad_(True)
#     return net

# def MnasNet_A1_Pretrain_ScoreLoss(opt, level=5):
#     net = mnasnet_a1(pretrained=True)
#     net.requires_grad_(False)
#     channel = net.output.in_features
#     net.output = nn.Linear(channel, opt.classes*level)
#     net.output.requires_grad_(True)
#     return net, level

if __name__ == '__main__':
    import argparse
    import torch
    import copy
    from thop import profile, clever_format_MB
    import csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=1000, help='the classes ')
    parser.add_argument('--size', type=tuple, default=(224, 224), help='the size of net input image (w, h) ')
    parser.add_argument('--batch_size', type=int, default=1, help='size of one batch')
    parser.add_argument('--model_group', nargs='+', default=["mobileVGG"], help='the name of model')
    parser.add_argument('--inchannel', type=int, default=3, help='the inchannl ')
    parser.add_argument('--device', type=str, default="cpu", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dtype', type=torch.dtype, default=torch.float32, help='dtype of tensor')
    opt = parser.parse_args()
    
    levels = [1]+[i*5 for i in range(1, 6+1)]
    classesList = [j*10 for j in range(1, 100+1)]
    
    writers = []
    for ids in ['macs', 'params']:
        csvfile = open(f'.{ids}.csv', mode='w', newline='')
        # 标题列表
        fieldnames = list(map(str, levels))
        # 创建 DictWriter 对象
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入表头
        write.writeheader()
        writers.append(write)

    for classes in classesList:
        opt.classes = classes
        infos_macs = {}
        infos_params = {}
        for _level in levels:      
            net, level = Resnet18_Pretrain_ScoreLoss(opt, level=_level)
            # net = MobileNetV3_Small_Pretrain(opt)
            # print(net)
            inputs = torch.randn(1, opt.inchannel, opt.size[0], opt.size[1])
            test_model = copy.deepcopy(net).to('cpu')
            macs, params = profile(test_model, inputs=(inputs,), )
            macs, params = clever_format_MB([macs, params], "%.6f")
            print(level, macs, params)
            
            
            infos_macs.setdefault(f'{_level}', float(macs[:-1]))
            infos_params.setdefault(f'{_level}', float(params[:-1]))
            
        writers[0].writerow(infos_macs)
        writers[1].writerow(infos_params)
    