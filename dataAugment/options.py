# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/16/23 9:52 AM
# __filename:  options.py
from pythonUtils import *
import config as config
import torch
class Options(object):
    def __init__(self, Config):
        # tensorboard监控
        self.write = True
        # 数据集准备相关配置
        self.num_workers = 6
        self.dataset_mode = Config.Dataset_Mode
        self.size = Config.Size
        self.resize = Config.Resize
        self.datasetsList = Config.DatasetsList
        self.dataset = Config.Dataset
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
                             "Dropout": Config.AugmentDropout,
                             "Multiply": Config.AugmentMultiply,
                             "Brightness": Config.AugmentBrightness,
                             "Saturate": Config.AugmentSaturate,
                             }

        self.inchannel = Config.InChannel

        self.dtype = torch.float32

    def printInfo(self, file=None):
        for key, value in self.__dict__.items():
            key = (str(key).capitalize() + ":").center(30, " ")
            printPlus(key + str(value), _file=file)


# opt = Options(config)