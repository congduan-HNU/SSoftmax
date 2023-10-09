# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/16/23 10:39 AM
# __filename:  dataset.py
from pythonUtils import *
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa

class MyCoTransform_numpy(object):
    def __init__(self, opt, DoAugment=False): 
        self.size = opt.size
        if not hasattr(opt, "resize"):
            self.resize = None
        self.resize = opt.resize
        self.gaussian_blur = opt.data_augment["GaussianBlur"] if DoAugment else False
        self.gaussion_noise = opt.data_augment["Gaussion_noise"] if DoAugment else False
        self.Sharpen = opt.data_augment["Sharpen"] if DoAugment else False
        self.ContrastNormalization = opt.data_augment["ContrastNormalization"] if DoAugment else False
        self.AffineScale = opt.data_augment["AffineScale"] if DoAugment else False
        self.AffineTranslate = opt.data_augment["AffineTranslate"] if DoAugment else False
        self.AffineRotate = opt.data_augment["AffineRotate"] if DoAugment else False
        self.AffineShear = opt.data_augment["AffineShear"] if DoAugment else False
        self.PiecewiseAffine = opt.data_augment["PiecewiseAffine"] if DoAugment else False
        self.Fliplr = opt.data_augment["Fliplr"] if DoAugment else False
        self.Flipud = opt.data_augment["Flipud"] if DoAugment else False
        self.Multiply = opt.data_augment["Multiply"] if DoAugment else False
        self.Dropout = opt.data_augment["Dropout"] if DoAugment else False

        self.transform = []

        # resize
        if self.size and self.resize != None:
            self.scale = iaa.Resize({"height": self.size[0], "width": self.size[1]},
                                    interpolation="linear",
                                    seed=None,
                                    name=None,
                                    random_state="deprecated",
                                    deterministic="deprecated")
        else:
            self.scale = None

        if self.gaussian_blur:
            radius = random.random() * 1.2
            self.transform.append(iaa.GaussianBlur(sigma=radius, name=None, random_state=None))

        if self.gaussion_noise:
            self.transform.append(
                iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.1 * 255), per_channel=0.2, name=None,
                                          random_state=None))

        if self.Sharpen:
            self.transform.append(
                iaa.Sharpen(alpha=(0.0, 0.15), lightness=(0.8, 1.2), name=None, random_state=None))

        if self.ContrastNormalization:
            self.transform.append(
                iaa.ContrastNormalization(alpha=(0.5, 1.5), per_channel=False, name=None,
                                          random_state=None))

        scale = (0.95, 1.05) if self.AffineScale else 1.0

        if self.AffineTranslate:
            translate_percent = random.uniform(-1, 1) * 0.05
        else:
            translate_percent = random.uniform(-1, 1) * 0
        if self.AffineRotate:
            rotate = random.uniform(-1, 1) * 15
        else:
            rotate = random.uniform(-1, 1) * 0
        if self.AffineShear:
            shear = random.uniform(-1, 1) * 5
        else:
            shear = random.uniform(-1, 1) * 0

        if self.AffineScale or self.AffineTranslate or self.AffineRotate or self.AffineShear:
            self.transform.append(iaa.Affine(scale=scale,
                                                translate_percent=translate_percent,
                                                translate_px=None,
                                                rotate=rotate,
                                                shear=shear,
                                                order=1,
                                                cval=0,
                                                mode='constant',
                                                name=None,
                                                random_state=None))

        if self.PiecewiseAffine:
            self.transform.append(
                iaa.PiecewiseAffine(scale=(0.0, 0.04), nb_rows=(2, 4), nb_cols=(2, 4), order=1, cval=0, mode='constant',
                                    name=None, random_state=None))

        if self.Fliplr:
            self.transform.append(iaa.Fliplr(p=1, name=None, random_state=None))

        if self.Flipud:
            self.transform.append(iaa.Flipud(p=1, name=None, random_state=None))

        if self.Multiply:
            self.transform.append(
                iaa.Multiply(mul=(0.8, 1.2), per_channel=False, name=None, random_state=None))

        if self.Dropout:
            self.transform.append(
                iaa.Dropout(p=(0.0, 0.1), per_channel=False, name=None, random_state=None))

        if self.scale is not None:
            self.seq1 = iaa.Sequential(self.scale,
                                  random_order=False,
                                  name=None,
                                  random_state=False
                                  )
        else:
            self.seq1 = None

        if len(self.transform)!=0:
            self.seq2 = iaa.SomeOf((1, min(len(self.transform)//3+1, 3)), self.transform,
                                 random_order=False,
                                 name=None,
                                 random_state=True
                                 )
        else:
            self.seq2 = None


    def __call__(self, input):  # sourcery skip: avoid-builtin-shadow
        # do something to images
        input = np.expand_dims(input, 0) #2021/10/27由被调用处transform移植此，可导致早期版本threeMDAD数据集维度错误
        if self.seq1 is not None:
            input = self.seq1.augment_images(input)
        if self.seq2:
            input = self.seq2.augment_images(input)
        input = np.squeeze(input, 0)
        return input
    
class ClassifyDataset(Dataset):
    def __init__(self, opt, dataset, imgPath, label, transform=None):
        self.opt = opt
        assert isinstance(dataset, dict), "Dataset format is illegal."
        self.dataset = dataset
        self.ImageRoorPath = imgPath
        self.LabelFile = label
        fh = open(self.LabelFile, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if len(words) == 3:
                imgs.append((words[1], int(words[2])))
            else:
                imgs.append((words[0], int(words[1])))
        if opt.debug:
            random.shuffle(imgs)
            self.imgs = imgs[:6000]
        else:
            self.imgs = imgs[:]
        self.transform = transform
        if self.dataset["modal"] == "rgb":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif self.dataset["modal"] == "ir":
            self.mean = [0.5, 0.5, 0.5]
            self.std = [1, 1, 1]
        else:
            self.mean = [0, 0, 0]
            self.std = [1, 1, 1]
        self.imgTIFF = np.zeros((480, 640))

    def min_max_norm(self, mtx, s=1):
        mtx = s * ((mtx - mtx.min()) / (mtx.max() - mtx.min() + 0.0001))
        return mtx


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        if "HOG" in self.dataset["modal"]:
            HoG = cv2.imread(os.path.join(self.ImageRoorPath, fn), 2).astype(np.float32)
            img_rgb = np.stack([HoG, HoG, HoG], 2)
        else:
            bgr = cv2.imread(os.path.join(self.ImageRoorPath, fn))
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)  # 是否进行transform
        img_rgb = transforms.ToTensor()(img_rgb).to(dtype=self.opt.dtype)  # h,w,c->c,h,w  [0,255]->[0.0,1,0]
        labels = torch.tensor(int(label))
        # 图片标准化
        img_rgb = transforms.Normalize(self.mean, self.std)(img_rgb)
        # return {'images': img_rgb, 'labels': labels}
        # only use the return value when inferDatasetsItems
        return {'images': img_rgb, 'labels': labels, 'items': os.path.splitext(fn)[0]}

    def __len__(self):
        return len(self.imgs)

