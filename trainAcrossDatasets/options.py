# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/16/23 9:52 AM
# __filename:  options.py
from pythonUtils import *
import train_config as config
import torch
class Options(object):
    def __init__(self, Config: config):
        # tensorboard监控
        self.write = True
        # 数据集准备相关配置
        self.num_workers = 6
        self.dataset_mode = Config.Dataset_Mode
        self.size = Config.Size
        self.resize = Config.Resize
        # self.datasetsList = Config.DatasetsList
        self.dataset = Config.Dataset
                # self.dataset_concat = Config.Dataset_Concat
                # self.train_combiles = Config.Train_combiles
                # self.train_combile = self.dataset_concat
        self.classes = None
        self.data_augment = {"GaussianBlur": Config.AugmentGaussianBlur,
                             "Gaussion_noise": Config.AugmentGaussion_noise,
                             "Sharpen": Config.AugmentSharpen,
                             "ContrastNormalization": Config.AugmentContrastNormalization,
                             "AffineScale": Config.AugmentAffineScale,
                             "AffineTranslate": Config.AugmentAffineTranslate,
                             "AffineRotate": Config.AugmentAffineRotate,
                             "AffineShear": Config.AugmentAffineShear,
                             "PiecewiseAffine": Config.AugmentPiecewiseAffine,
                             "Fliplr": Config.AugmentFliplr,
                             "Flipud": Config.AugmentFlipud,
                             "Multiply": Config.AugmentMultiply,
                             "Dropout": Config.AugmentDropout,
                             }

        # 模型相关
        self.model_group = Config.Model_Group
        self.pretrain = Config.Pretrain
        self.pretrain_pth = Config.Pretrain_pth
        # self.model = Config.Model

        # 优化器相关
        self.lr_decay_milestones = Config.Lr_Decay_Milestones
        self.lr_decay_gamma = Config.Lr_Decay_Gamma
        self.lr_decay_lambda = Config.Lr_decay_Lambda
                # self.multistep = Config.Multistep
                # self.optimizer = Config.Optimizer
        self.lr = Config.LR



        self.epochs = Config.Epochs
        self.batch_size = Config.BatchSize
        self.test_percent = Config.TestPercent
        self.test_rate = Config.TestRate
        self.loss = Config.Loss
        self.MSEmode = Config.MSEmode
        
                # self.weight_decay = Config.Optimizer["weight_decay"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.inchannel = Config.InChannel
        self.mul_gpu_train = Config.Mul_Gpu_Train
        self.paraminit = Config.ParamInit


        self.dtype = torch.float32

    def printInfo(self, file=None):
        for key, value in self.__dict__.items():
            key = f"{str(key).capitalize()}:".center(30, " ")
            printPlus(key + str(value), _file=file)


# opt = Options(config)