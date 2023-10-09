# _*_coding:utf-8_*_
# __author:    duancong
# __date:      4/20/23 1:56 PM
# __filename:  loss.py
import sys
import os
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import torchvision.utils
from pythonUtils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from options import Options
import random

class ScoreLossPlus(nn.Module):
    def __init__(self, opt:Options, level:int):
        super(ScoreLossPlus, self).__init__()
        self.n_classes = opt.classes
        self.type = 'score distance with score sum'
        self.hard = True
        self.opt = opt
        self.scoreLevel = level
        self.k = 0
        self.alpha = 10
        score = np.reshape(np.arange(0, self.scoreLevel, 1), (-1, 1))
        # print(score)
        score = np.concatenate([score]*self.opt.classes, axis=1)
        # print(score)
        
        self.score_gt = torch.from_numpy(score).to(dtype=self.opt.dtype, device=self.opt.device)
        
        self.level_split = self.score_gt/(self.scoreLevel-1)
        
        score = score[None,:,:]
        # score = np.concatenate([score]*self.opt.batch_size, axis=0)
        # print(score)
        
        self.score = torch.from_numpy(score).to(dtype=self.opt.dtype, device=self.opt.device)
        
        droptLowHigh = np.reshape(np.zeros(self.scoreLevel), (-1, 1))
        valid = np.reshape(np.arange(1, self.scoreLevel-1, 1), (-1, 1))
        droptLowHigh[1:-1] = valid
        # print(droptLowHigh)
        # score = np.concatenate([score]*self.opt.batch_size, axis=0)
        # print(score)
        droptLowHigh = np.concatenate([droptLowHigh]*self.opt.classes, axis=1)
        droptLowHigh = droptLowHigh[None,:,:]
        self.droptLowHighscore = torch.from_numpy(droptLowHigh).to(dtype=self.opt.dtype, device=self.opt.device)
        
        # self.score_gt = torch.tensor([[0]*opt.classes,
        #                             [1]*opt.classes,
        #                             [2]*opt.classes,
        #                             [3]*opt.classes,
        #                             [4]*opt.classes]).to(opt.device)
        # print(self.score)
        # print(self.score_gt)
        
        # print(np.linspace(0, self.scoreLevel-1, self.scoreLevel))
        self.gt_true_mean = 0.8
        self.gt_true_std = 0.2
        self.gt_true = self._norm_func(np.linspace(0, 1, self.scoreLevel), self.gt_true_mean, self.gt_true_std)
        # print(self.gt_true)
        # self.gt_true = self.gt_true/(np.sum(self.gt_true))
        self.gt_true = self.gt_true/(np.sum(self.gt_true))
        self.gt_true = np.reshape(self.gt_true[None,:], (-1, 1))
        # print(self.gt_true, np.sum(self.gt_true))
        
        # self.gt_false = self._norm_func(np.linspace(0, self.scoreLevel-1, self.scoreLevel), random.uniform(0, self.scoreLevel/2), random.uniform(3, 5))
        # # print(self.gt_false)
        # self.gt_false = self.gt_false**2/(np.sum(self.gt_false**2))
        # self.gt_false = np.reshape(self.gt_false[None,:], (-1, 1))
        # # print(self.gt_false, np.sum(self.gt_false))
        self.gt_false_mean = (0, 0.5)
        self.gt_false_std = (0.6, 1)
        
        
    '''
    description: 正态分布概率密度函数
    param {*} x 采样x值
    param {*} u 均值
    param {*} sig 标准差
    return {*}
    '''        
    def _norm_func(self, x, u, sig):
        return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)

    def forward(self, logits, targets:torch.Tensor):
        # print(logits[0])
        # print(targets[0])
        
        batch = targets.shape[0]
        
        gt = np.zeros([batch, self.scoreLevel, self.n_classes])
        for j in range(0, batch):
            for i in range(0, gt.shape[2]):
                if targets[j].item()==i:
                    for k in range(0, self.scoreLevel):
                        gt[j][k][i] = self.gt_true[k, 0]
                else:
                    gt_false = self._norm_func(np.linspace(0, 1, self.scoreLevel), \
                        random.uniform(self.gt_false_mean[0], self.gt_false_mean[-1]), \
                            random.uniform(self.gt_false_std[0], self.gt_false_std[-1]))
                    # gt_false = gt_false/(np.sum(gt_false))
                    gt_false = gt_false/(np.sum(gt_false))
                    gt_false = np.reshape(gt_false[None,:], (-1, 1))
                    for k in range(0, self.scoreLevel):
                        gt[j][k][i] = gt_false[k, 0]
                   
        gt = torch.from_numpy(gt)
        gt = gt.to(device=self.opt.device)
        
        # print("weight:", weight)
        # print("gt:", gt)
        # print("logits:", logits)
        # print(gt.shape)
        # print(logits.shape)
        logits = logits.to(dtype=self.opt.dtype)
        gt = gt.to(dtype=self.opt.dtype)
        # print(score)
        # print(logits)
        # print()

        if self.type == 'score distance with score sum':
            # print(self.score)
            # print(logits)
            # pred_score = torch.matmul(score, logits)
            # print(self.score.shape)
            # print(logits.shape)
            pred_score = torch.mul(self.score, logits)
            # print(pred_score[0])
            # class_score = torch.sum(pred_score, dim=1)
            # print(class_score[0])
            # targets = 
            
            # print(gt)
            gt_score = torch.mul(self.score, gt)
            
            # print(gt)
            # print(gt_score)
            score_dis = torch.square(pred_score-gt_score)
            score_dis = torch.sum(score_dis)
            score_dis = torch.sqrt(score_dis)
            # loss = score_dis + self.k*((self.scoreLevel/2)/(torch.mean(pred_score))+(self.scoreLevel/2)*torch.mean(pred_score)) # sum to mean
            loss = score_dis + (self.alpha)*self.k*(1/(torch.mean(pred_score))+torch.mean(pred_score))
            # loss = score_dis + self.k*(self.alpha/(torch.mean(pred_score))+torch.mean(pred_score)/self.alpha) # sum to mean  
            # print(loss)
        return loss, score_dis, torch.mean(pred_score)
    
    def printInfo(self, file=None):
        printPlus('\nLoss Function Information: ScoreLossPlus', _file=file)
        for key, value in self.__dict__.items():
            if key[0]=='_':
                continue
            key = f"{str(key).capitalize()}:".center(30, " ")
            if (type(value) == torch.Tensor) or (type(value)==np.ndarray):
                printPlus(key + '\n' + str(value), _file=file)
            else:
                printPlus(key + str(value), _file=file)
    
                   
if __name__ == '__main__':
    import train_config as config
    opt = Options(config)
    opt.classes=10
    loss_f = ScoreLossPlus(opt, 10)
    loss_f.printInfo()
    
    