# _*_coding:utf-8_*_
# __author:    duancong
# __date:      4/20/23 1:35 PM
# __filename:  train_config.py.py
from pythonUtils import *
sys.path.append('labelfolder')


#优化器选择S
# SGD = {"name": "SGD", "lr": 0.0001, "weight_decay": 0.001, "momentum": 0.9}
# Adam = {"name": "Adam", "lr": 0.001, "weight_decay": 0.001}
#数据集选择k
import datasetDriver100
Dataset = datasetDriver100.Driver100_Cross_Camera_Setting_D1
# 训练参数部分
Epochs = 50  #训练总纪元数
BatchSize = 64 #批大小
TestPercent = 0 #训练集中按比例划分一部分出来作为测试集，用于检测训练状态（非评估集）[0, 1]
TestRate = 5 #1epoch 测试次数比例
Loss = None #损失函数名称 "cross_entropy_loss" or "mse_loss_plus"
MSEmode = "" #若损失函数采用均方误差，则选择模式，为损失函数各类权重模式
Lr_Decay_Milestones = [40, 45, 50] #阶梯式更改学习率更改的epoch点
Lr_Decay_Gamma = 0.1 #阶梯式更改学习率的衰减系数
Lr_decay_Lambda = None #函数式更改学习率时的lamda值
LR = 0.001
# Optimizer = Adam
Device = "cuda" #训练设备选择，"cuda" or "cpu"
Pretrain = False #是否选用预训练模型
Pretrain_pth = "./pretrainFactory/SFD(trainsub)&AUCv1/epoch_100.pth" #预训练文件路径，继续训练时使用,应配合Pretrain使用

Model_Group = ['MobileNetV3_Small_Pretrain_ScoreLoss', 'Resnet18_Pretrain_ScoreLoss', '#MnasNet_A1_Pretrain_ScoreLoss', '#GhostNet_100_Pretrain_ScoreLoss'][:2]
Model_Group = [i for i in Model_Group if "#" not in i]
Dataset_Mode = ["RGB"] #训练数据模式，"RGB" or "IR" or "Depth" or "Depth-IR"
InChannel = 3 #输入图从channel数
Size = (224, 224) #输入图像尺寸(w, h)
Resize = True
Mul_Gpu_Train = False #是否采用多GPU训练
ParamInit = True#是否进行参数初始化


# 数据增强部分
AugmentGaussianBlur = False #高斯模糊
AugmentGaussion_noise = False #高斯噪声
AugmentSharpen = False #锐化
AugmentContrastNormalization = False #对比度归一�?
AugmentAffineScale = False #缩放
AugmentAffineTranslate = False #平移
AugmentAffineRotate = False #旋转
AugmentAffineShear = False #透视变换
AugmentPiecewiseAffine = False #控制点的方式随机形变
AugmentFliplr = False #左右翻转
AugmentFlipud = False #上下翻转
AugmentMultiply = False #每个像素随机乘一个数
AugmentDropout = False #随机丢弃像素
AugmentBrightness = False
AugmentSaturate = False