# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/16/23 10:04 AM
# __filename:  prepare.py

from pythonUtils import *
from options import Options
from dataset import ClassifyDataset, MyCoTransform_numpy
from torch.utils.data import DataLoader, ConcatDataset

class DatasetPrepare(object):
    def __init__(self, project: ProjectInfo, option: Options):
        self.project = project
        self.opt = option
        self.opt.debug = self.project.IsDebug
        self.augmental = MyCoTransform_numpy(self.opt, DoAugment=True)
        self.noaugmental = MyCoTransform_numpy(self.opt, DoAugment=False)
        self.num_workers = 6

        # self.opt.dataset = self.opt.datasetsList[0]
        self.opt.classes = self.opt.dataset["classes"]
        # 训练集（训练）
        if type(self.opt.dataset["ImageTrainPath"]) == str:
            self.trainSet = ClassifyDataset(self.opt, self.opt.dataset, self.opt.dataset["ImageTrainPath"],
                                            osp.join(self.project.ROOT, self.opt.dataset["TrainLabelPath"]),
                                            transform=self.noaugmental)
        elif type(self.opt.dataset["ImageTrainPath"]) == list:
            self.trainSet = ConcatDataset([ClassifyDataset(self.opt, self.opt.dataset, self.opt.dataset["ImageTrainPath"][i],
                                            osp.join(self.project.ROOT, self.opt.dataset["TrainLabelPath"][i]),
                                            transform=self.noaugmental) for i in range(0, len(self.opt.dataset["ImageTrainPath"]))])
        
        self.train_loader = DataLoader(self.trainSet, batch_size=self.opt.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)
        
        # 评估集 (训练)
        if type(self.opt.dataset["ImageValPath"]) == str:
            self.evaluateSet = ClassifyDataset(self.opt, self.opt.dataset, self.opt.dataset["ImageValPath"],
                                            osp.join(self.project.ROOT, self.opt.dataset["ValLabelPath"]),
                                            transform=self.noaugmental)
            self.evaluate_loader = DataLoader(self.evaluateSet, batch_size=self.opt.batch_size, shuffle=False,
                                          num_workers=self.num_workers, pin_memory=True)
            
        elif type(self.opt.dataset["ImageValPath"]) == list:
            self.evaluateSet = [ClassifyDataset(self.opt, self.opt.dataset, self.opt.dataset["ImageValPath"][i],
                                            osp.join(self.project.ROOT, self.opt.dataset["ValLabelPath"][i]),
                                            transform=self.noaugmental) for i in range(0, len(self.opt.dataset["ImageValPath"]))]
            self.evaluate_loader = [DataLoader(self.evaluateSet[i], batch_size=self.opt.batch_size, shuffle=False,
                                          num_workers=self.num_workers, pin_memory=True) for i in range(0, len(self.opt.dataset["ImageValPath"]))]
            

        if ("class_Samples" in self.opt.dataset.keys()):
            self.opt.classSamples = self.opt.dataset["class_Samples"]

        
        
        # 测试集 (测试)
        if self.opt.dataset["TestLabelPath"] != None:
            if type(self.opt.dataset["TestLabelPath"]) == str:
                self.testSet = ClassifyDataset(self.opt, self.opt.dataset, self.opt.dataset["ImageTestPath"],
                                            osp.join(self.project.ROOT, self.opt.dataset["TestLabelPath"]),
                                            transform=self.noaugmental)
                self.test_loader = DataLoader(self.testSet, batch_size=self.opt.batch_size, shuffle=False,
                                              num_workers=self.num_workers, pin_memory=True)
            elif type(self.opt.dataset["TestLabelPath"]) == list:
                self.testSet = [ClassifyDataset(self.opt, self.opt.dataset, self.opt.dataset["ImageTestPath"][i],
                                            osp.join(self.project.ROOT, self.opt.dataset["TestLabelPath"][i]),
                                            transform=self.noaugmental) for i in range(0, len(self.opt.dataset["ImageTestPath"]))]
                self.test_loader = [DataLoader(self.testSet[i], batch_size=self.opt.batch_size, shuffle=False,
                                              num_workers=self.num_workers, pin_memory=True) for i in range(0, len(self.opt.dataset["ImageTestPath"]))]
            

    def loadDataset(self):
        if type(self.evaluate_loader) is list:
            for _cell in self.evaluate_loader:
                for _data in tqdm(_cell, position=0, desc='load evaluate dataset'):
                    pass
        else:   
            for _data in tqdm(self.evaluate_loader, position=0, desc='load evaluate dataset'):
                pass
        for _data in tqdm(self.train_loader, position=0, desc='load train dataset'):
            pass
        if self.opt.dataset["TestLabelPath"] != None:
            for _data in tqdm(self.test_loader, position=0, desc='load test dataset'):
                pass



