# _*_coding:utf-8_*_
# __author:    duancong
# __date:      4/24/23 3:21 PM
# __filename:  iaaLib.py
"""
数据增强
"""
import os
import sys

sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
from pythonUtils import *

import numpy as np
from imgaug import augmenters as iaa
import random
import cv2
from threading import Thread, Lock

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
        self.Brightness = opt.data_augment["Brightness"] if DoAugment else False
        self.Saturate = opt.data_augment["Saturate"] if DoAugment else False
        # self.Dropout = opt.data_augment["Dropout"] if DoAugment else False

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
            self.transform.append(iaa.GaussianBlur(sigma=radius, name=None, deterministic=False, random_state=None))

        if self.gaussion_noise:
            self.transform.append(
                iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.1 * 255), per_channel=0.2, name=None, deterministic=False,
                                          random_state=None))

        if self.Sharpen:
            self.transform.append(
                iaa.Sharpen(alpha=(0.0, 0.15), lightness=(0.8, 1.2), name=None, deterministic=False, random_state=None))

        if self.ContrastNormalization:
            self.transform.append(
                iaa.ContrastNormalization(alpha=(0.5, 1.5), per_channel=False, name=None, deterministic=False,
                                          random_state=None))

        if self.AffineScale:
            scale = (0.95, 1.05)
        else:
            scale = 1.0
        if self.AffineTranslate:
            translate_percent = random.uniform(-1, 1) * 0.05
        else:
            translate_percent = random.uniform(-1, 1) * 0
        if self.AffineRotate:
            rotate = random.uniform(-1, 1) * 25
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
                                                deterministic=False,
                                                random_state=None))

        if self.PiecewiseAffine:
            self.transform.append(
                iaa.PiecewiseAffine(scale=(0.0, 0.04), nb_rows=(2, 4), nb_cols=(2, 4), order=1, cval=0, mode='constant',
                                    name=None, deterministic=False, random_state=None))

        if self.Fliplr:
            self.transform.append(iaa.Fliplr(p=1, name=None, deterministic=False, random_state=None))

        if self.Flipud:
            self.transform.append(iaa.Flipud(p=1, name=None, deterministic=False, random_state=None))

        if self.Multiply:
            self.transform.append(
                iaa.Multiply(mul=(0.8, 1.2), per_channel=False, name=None, deterministic=False, random_state=None))

        if self.Dropout:
            self.transform.append(
                iaa.Dropout(p=(0.0, 0.1), per_channel=False, name=None, deterministic=False, random_state=None))

        # 调整图像亮度
        if self.Brightness:
            self.transform.append(iaa.imgcorruptlike.Brightness(severity=random.randint(1, 5)))

        #调整图像饱和度
        if self.Saturate:
                self.transform.append(iaa.imgcorruptlike.Saturate(severity=random.randint(1, 5)))

        if self.scale is not None:
            self.seq1 = iaa.Sequential(self.scale,
                                  random_order=False,
                                  name=None,
                                  deterministic=False,
                                  random_state=False
                                  )
        else:
            self.seq1 = None

        if len(self.transform)!=0:
            self.seq2 = iaa.SomeOf((1, min(len(self.transform)//3+1, 3)), self.transform,
                                 random_order=True,
                                 name=None,
                                 deterministic=False,
                                 random_state=True
                                 )
        else:
            self.seq2 = None

        pass


    def __call__(self, input):
        # do something to images
        input = np.expand_dims(input, 0) #2021/10/27由被调用处transform移植此，可导致早期版本threeMDAD数据集维度错误
        if self.seq1 is not None:
            input = self.seq1.augment_images(input)
        if self.seq2:
            input = self.seq2.augment_images(input)
        input = np.squeeze(input, 0)
        return input

def augmentMutiThread(imageRoot:str, imageDist:str, images:list, fh, thread_id, transform:MyCoTransform_numpy, counts=6, lock:Lock = Lock()):
    for image_path in tqdm(images, position=0, desc=f'Thread:{thread_id+1}'):
    # for image_path in images:
        path, gt = image_path.split()
        bgr = cv2.imread(osp.join(imageRoot, path))
        if (transform.seq1 != None) and (transform.seq2 == None):
            img_rgb = transform(bgr)
            dist_name, type = osp.splitext(path)[0], osp.splitext(path)[1]
            creat_folder(osp.dirname(osp.join(imageDist, dist_name+type)), False)
            # print(osp.join(imageDist, dist_name+type))
            if not osp.exists(osp.join(imageDist, dist_name+type)):         
                try:
                    cv2.imwrite(osp.join(imageDist, dist_name+type), img_rgb)
                except Exception as e:
                    printPlus(osp.join(imageDist, dist_name+type), frontColor=31)
            # print(dist_name+type + ' ' + f'{gt}' +'\n')
            lock.acquire()
            # print(dist_name + type + ' ' + f'{gt}' +'\n')
            fh.write(dist_name + type + ' ' + f'{gt}' +'\n')
            lock.release()
        else:
            for i in range(0, counts):
                img_rgb = transform(bgr)
                dist_name, type = osp.splitext(path)[0], osp.splitext(path)[1]
                # print(osp.join(imageDist, dist_name+f'_{str(i)}'+type))
                creat_folder(osp.dirname(osp.join(imageDist, dist_name+f'_{str(i)}'+type)), False)
                # print(osp.join(imageDist, dist_name+f'_{str(i)}'+type))
                if not osp.exists(osp.join(imageDist, dist_name+f'_{str(i)}'+type)):
                    try:
                        cv2.imwrite(osp.join(imageDist, dist_name+f'_{str(i)}'+type), img_rgb)
                    except Exception as e:
                        printPlus(osp.join(imageDist, dist_name+f'_{str(i)}'+type), frontColor=31)
                lock.acquire()
                # print(dist_name+f'_{str(i)}'+type + ' ' + f'{gt}' +'\n')
                fh.write(dist_name+f'_{str(i)}'+type + ' ' + f'{gt}' +'\n')
                lock.release()
    return

def augmentOneThread(imageRoot:str, imageDist:str, images:list, fh, transform:MyCoTransform_numpy, counts=6):
    for image_path in tqdm(images, position=0, desc=f'Thread:{1}'):
        path, gt = image_path.split()
        bgr = cv2.imread(osp.join(imageRoot, path))
        if (transform.seq1 != None) and (transform.seq2 == None):
            dist_name, type = osp.splitext(path)[0], osp.splitext(path)[1]
            creat_folder(osp.dirname(osp.join(imageDist, dist_name+type)), False)
            # print(osp.join(imageDist, dist_name+type))
            cv2.imwrite(osp.join(imageDist, dist_name+type), bgr)
            # print(dist_name+type + ' ' + f'{gt}' +'\n')   
        else:        
            for i in range(0, counts):
                img_rgb = transform(bgr)
                dist_name, type = osp.splitext(path)[0], osp.splitext(path)[1]
                # print(osp.join(imageDist, dist_name+f'_{str(i)}'+type))
                creat_folder(osp.dirname(osp.join(imageDist, dist_name+f'_{str(i)}'+type)), False)
                # print(osp.join(imageDist, dist_name+f'_{str(i)}'+type))
                # cv2.imwrite(osp.join(imageDist, dist_name+f'_{str(i)}'+type), img_rgb)
                # print(dist_name+f'_{str(i)}'+type + ' ' + f'{gt}' +'\n')
                # fh.write(dist_name+f'_{str(i)}'+type + '\n')
    return



if __name__ == '__main__':
    import distutils.util
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('--dataset', default='None', type=str, help ='train dataset')
    parser.add_argument('--subset', default='None', type=str, help ='train dataset')
    parser.add_argument('--augment', default=False, type=lambda x:bool(distutils.util.strtobool(x)), help ='train dataset')     
    args = parser.parse_args()  
    print(args.augment)

    import options
    import config
    project = ProjectInfo()
    opt = options.Options(config)
    transform = MyCoTransform_numpy(opt, args.augment)

    if args.subset == 'None':
        subsets = ["Train", "Val", "Test"]
        subset = subsets[2]
    else:
        subset = args.subset
    if args.dataset != 'None':
        from config import *
        opt.dataset = eval(args.dataset)
    imageRoot = opt.dataset[f"Image{subset}Path"]
    trainLabel = osp.join(project.ROOT, opt.dataset[f"{subset}LabelPath"])
    if args.augment:
        trainNewLabel = osp.join(project.ROOT, opt.dataset[f"{subset}LabelPath"].replace('.txt', '_augment.txt'))
        imageDist = opt.dataset[f"Image{subset}Path"] + '_Augment'
    else:
        trainNewLabel = osp.join(project.ROOT, opt.dataset[f"{subset}LabelPath"].replace('.txt', '_size224.txt'))
        imageDist = opt.dataset[f"Image{subset}Path"] + '_size224'
    lables_fh = open(trainLabel, 'r+')
    lables = lables_fh.readlines()
    images = []
    for i in lables:
        if len(i.strip().split())==3:
            idx, image, gt = i.strip().split()
        elif len(i.strip().split())==2:
            image, gt = i.strip().split()
        images.append(image + ' ' + gt)

    # with open(trainNewLabel, 'w+') as f:
    #     augmentOneThread(imageRoot, imageDist, images, f, transform, 6)

    lock = Lock()
    threads = []
    cells = augList(images, 10)
    with open(trainNewLabel, 'w+') as f:
        for i in range(0, len(cells)):
            print([imageRoot, imageDist, transform])
            threads.append(Thread(target=augmentMutiThread, args=[imageRoot, imageDist, cells[i], f, i, transform, 11, lock]))
            # arg = cells[i]
            # pool.apply_async(self.runOneKind, args=(self, arg))

        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        print('主线程结束...')


