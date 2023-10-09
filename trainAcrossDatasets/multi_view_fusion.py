'''
Author: Cong Duan
Date: 2023-09-06 13:27:32
LastEditTime: 2023-09-14 17:13:07
LastEditors: your name
Description: 多视角融合
FilePath: /Driver-Action-Monitor/trainAcrossDatasets/multi_view_fusion.py
可以输入预定的版权声明、个性签名、空行等
'''
import torch
import sys
import os
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
from pythonUtils import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False, #解决负号无法显示的问题
    "font.weight": "bold"
}
rcParams.update(config)

import itertools
from collections import OrderedDict
import pandas as pd

from options import Options
from dataset import ClassifyDataset, MyCoTransform_numpy
from model import *
from metrics import Metric, computeROCandAUC, computePR, t_SNE
from loss import ScoreLossPlus
np.set_printoptions(linewidth=1000)


class Option(object):
    info = ""
    
def loadCfg(file: str):
    opt = Option()
    infos = ['Num_workers', 'Size', 'Resize', 'Classes', 'Model', 'Dtype', 'Inchannel', 'Batch_size']
    with open(file, 'r') as fh:
        info = fh.readline().strip()
        while info:
            info = fh.readline().strip()
            if len(info) == 0:
                continue
            # print(info)
            ret = re.match('\w+', info).group()
            # print(ret)
            if ret in infos:
                # print(ret) 
                key, value = info.split(': ')
                if ret in ['Num_workers', 'Classes', 'Inchannel', 'Batch_size']:
                    setattr(opt, key.lower().replace(' ', ''), int(value.replace(' ', '')))
                elif ret in ['Resize']:
                    setattr(opt, key.lower().replace(' ', ''), bool(value.replace(' ', '')))
                elif ret in ['Size']:
                    value = value.replace(' ', '')
                    value = re.search('\d+', value)[0]
                    setattr(opt, key.lower().replace(' ', ''), (int(value), int(value)))
                elif ret in ['Dtype']:
                    setattr(opt, key.lower().replace(' ', ''), eval(value.replace(' ', '')))
                else:
                    setattr(opt, key.lower().replace(' ', ''), value.replace(' ', ''))
    opt.device = 'cpu'
    return opt

# 获得Top-1. Top-3, Top5 Acc
def predict(logits:torch.Tensor, labels:torch.Tensor, loss_f:ScoreLossPlus):
    # print("pred:")
    # print(logits[0])
    logits = logits * loss_f.score
    # print("score:")
    # print(logits[0])
    # print("gt:")
    # print(labels[0])
    logits = torch.sum(logits, dim=1)
    # print("all score:")
    # print(logits[0])
    _, pred = logits.topk(5, 1, largest=True, sorted=True)
    # print("score sort:")
    # print(pred[0])

    labels = labels.view(labels.size(0), -1).expand_as(pred)
    correct = pred.eq(labels).float()

    #compute top 5
    correct_5 = correct[:, :5].sum()
    #compute top 3
    correct_3 = correct[:, :3].sum()
    #compute top1
    correct_1 = correct[:, :1].sum()
    
    return [correct_1, correct_3, correct_5, pred[:, :1]], logits

# 获得融合的Top-1. Top-3, Top5 Acc
def predictMultiChannelFusion(logits_list:list, labels:torch.Tensor, loss_f:list):
    means = np.zeros(len(logits_list))
    stds = np.zeros(len(logits_list))
    logits_list_origin = copy.deepcopy(logits_list)
    logits_list = [logits_list[i]*loss_f[i].level_split for i in range(len(logits_list))]
    # logits_list_squre = [logits_list[i]*(loss_f[i].level_split**2) for i in range(len(logits_list))]
    
    # mean = torch.zeros((_logits.shape[0], _logits.shape[1]))
    # std = torch.zeros((_logits.shape[0], _logits.shape[1]))
    logits = torch.zeros((logits_list[0].shape[0], loss_f[-1].scoreLevel, logits_list[0].shape[2])).to(loss_f[-1].opt.dtype)
    for batch in range(logits_list[0].shape[0]):
        for cagetory in range(logits_list[0].shape[2]):
            for i in range(len(logits_list)): 
                # a = logits_list[i][batch, :, cagetory]         
                mean = torch.sum(logits_list[i][batch, :, cagetory]).item()
                # X_square = torch.sum(logits_list_squre[i][batch, :, cagetory]).item()
                # Xmean = 2*(mean**2)
                # mean_square = (mean**2)*torch.sum(logits_list_origin[i][batch, :, cagetory]).item()
                # std = X_square-2*Xmean + mean_square
                
                # std = torch.std(logits_list[i][batch, :, cagetory]).item()
                
                means[i] = mean
                # stds[i] = std
            for i in range(len(logits_list)): 
                logits_list_var = [logits_list_origin[i]*((loss_f[i].level_split-means[i])**2) for i in range(len(logits_list))]
                stds[i] = torch.sum(logits_list_var[i][batch, :, cagetory]).item()
            mean_new = np.mean(means)
            # std_new = np.sqrt(np.square(stds).sum())
            std_new = np.sqrt(stds.sum())
            
            score = loss_f[-1]._norm_func(np.linspace(0, 1, logits.shape[1]), mean_new, std_new)
            score = score/np.sum(score)
            for level in range(logits.shape[1]):
                logits[batch, level, cagetory] = score[level]

    
    
    # print("pred:")
    # print(logits[0])
    logits = logits * loss_f[-1].score
    # print("score:")
    # print(logits[0])
    # print("gt:")
    # print(labels[0])
    logits = torch.sum(logits, dim=1)
    # print("all score:")
    # print(logits[0])
    _, pred = logits.topk(5, 1, largest=True, sorted=True)
    # print("score sort:")
    # print(pred[0])

    labels = labels.view(labels.size(0), -1).expand_as(pred)
    correct = pred.eq(labels).float()

    #compute top 5
    correct_5 = correct[:, :5].sum()
    #compute top 3
    correct_3 = correct[:, :3].sum()
    #compute top1
    correct_1 = correct[:, :1].sum()
    
    return [correct_1, correct_3, correct_5, pred[:, :1]], logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', default='default', type=str, required=False)
    parser.add_argument('-dataset', default='default', type=str, required=False)
    parser.add_argument('-level', default=-1, type=int, required=False)
    args = parser.parse_args()
    project = ProjectInfo()
    cfgFile = osp.join(osp.join(project.ROOT, '/home/caomen/Desktop/DC/ablationExperiments/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-08-16-20_43_44/2023-08-16-20_45_06/config.txt'))
    opt = loadCfg(cfgFile)
    
    if args.level == -1:
        args.level = 5
    
    logits = []
    # resnet50_inferD1_trainD1 = pickRead('/user/duancong/DC/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-08-29-18_08_34/2023-09-01-13_31_51/Logits-EZZ2021.pkl')
    # logits.append(resnet50_inferD1_trainD1)
    # resnet50_inferD1_trainD2 = pickRead('/home/caomen/Desktop/DC/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-08-19-11_47_41/2023-08-19-11_49_52/Logits-Driver100_Cross_Camera_Setting_D1.pkl')
    # resnet50_inferD2_trainD2 = pickRead('/user/duancong/DC/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-09-05-10_14_08/2023-09-07-14_33_54/Logits-EZZ2021.pkl')
    # logits.append(resnet50_inferD2_trainD2)
    logits.append(pickRead('/home/caomen/Desktop/DC/ablationExperiments/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-09-04-17_01_51/2023-09-04-17_03_33/Logits-Driver100_Cross_Vehicle_Type_D4_Van_Test.pkl'))
    logits.append(pickRead('/home/caomen/Desktop/DC/ablationExperiments/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-09-04-17_01_51/2023-09-06-22_10_52/Logits-Driver100_Cross_Vehicle_Type_D4_Van_Test.pkl'))
    # logits.append(pickRead('/home/caomen/Desktop/DC/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-08-27-02_00_46/2023-08-27-02_03_16/Logits-Driver100_Cross_Camera_Setting_D3_Muti_Camere_Fusion.pkl'))
    # logits.append(pickRead('/home/caomen/Desktop/DC/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-08-31-05_19_12/2023-08-31-05_21_43/Logits-Driver100_Cross_Camera_Setting_D4_Muti_Camere_Fusion.pkl'))
    
    
    # resnet50_inferD1_trainD1.sort(key=lambda x:x['name'])
    # resnet50_inferD2_trainD2.sort(key=lambda x:x['name'])
    
    # Camera1 = resnet50_inferD1_trainD1
    # Camera2 = resnet50_inferD2_trainD2
    # Camera3 = resnet50_inferD2_trainD2
   
    # length = min(len(Camera1), len(Camera2))
    # Camera1_fine = []
    # Camera2_fine = []
    # streams = []
    # for i in range(len(Camera1)):
    #     index1 = Camera1[i]['name'].rindex('_')
    #     stream1 = Camera1[i]['name'][:index1]
    #     if stream1 not in streams:
    #         streams.append(stream1)
    # for i in range(len(Camera2)):
    #     index2 = Camera2[i]['name'].rindex('_')
    #     stream2 = Camera2[i]['name'][:index2]
    #     if stream2 not in streams:
    #         print(stream2)
        
    # for i in range(len(streams)):
    #     for j in range(len(Camera1)):
    #         index1 = Camera1[j]['name'].rindex('_')
    #         stream1 = Camera1[j]['name'][:index1]
    #         if stream1 == streams[i]:             
    #             Camera1_fine.append(Camera1[j]['name'])

    # for i in range(len(streams)):
    #     for j in range(len(Camera2)):
    #         index2 = Camera2[j]['name'].rindex('_')
    #         stream2 = Camera2[j]['name'][:index2]
    #         if stream2 == streams[i]:             
    #             Camera2_fine.append(Camera2[j]['name'])
    
    
    # for i in range(len(Camera1)):
    #     # if Camera1[i]['name'].split('_')[-1][:2] == '01':
    #     if 'C10_Look_Right/P010_V1_S2' in Camera1[i]['name']:
    #         # Camera1.remove(Camera1[i])
    #         Camera1_fine.append(Camera1[i])
    # for i in range(len(Camera2)):
    #     # if Camera2[i]['name'].split('_')[-1][:2] == '02':
    #     if 'C10_Look_Right/P010_V1_S2' in Camera1[i]['name']:
    #         # Camera2.remove(Camera2[i])
    #         Camera2_fine.append(Camera2[i])
            
    # Camera1_names = []
    # Camera2_names = []
    # for i in range(len(Camera1_fine)):
    #     Camera1_names.append(Camera1_fine[i]['name'])
    # print("Camera1_names:", len(Camera1_names))
    # for i in range(len(Camera2_fine)):
    #     Camera2_names.append(Camera2_fine[i]['name'])
    # print("Camera2_names:", len(Camera2_names))
    
    # Cam1_and_Cam2 = list(set(Camera1_names) & set(Camera2_names))
    
    # for i in range(len(Camera1)):
    #     if Camera1[i]['name'] not in Cam1_and_Cam2:
    #         Camera1.remove(Camera1[i])
    # for i in range(len(Camera2)):
    #     if Camera2[i]['name'] not in Cam1_and_Cam2:
    #         Camera2.remove(Camera2[i])
            
    # assert len(Camera1) == len(Camera2), 'The length is not equality!'
    
    loss_f_s = []
    loss_f = ScoreLossPlus(opt, args.level)
    levels = [15, 15, 15, 15, 10]
    
    Cameras = []
    for i in range(len(logits)):
        logits[i].sort(key=lambda x:x['name'])
        Cameras.append(logits[i])
        loss_f_s.append(ScoreLossPlus(opt, levels[i]))
    loss_f_s.append(ScoreLossPlus(opt, levels[-1]))
    
    for i in Cameras:
        assert len(i) == len(Cameras[0]), 'Error'
    correct_1 = [0]*len(Cameras) + [0, 0]
    correct_3 = [0]*len(Cameras) + [0, 0]
    correct_5 = [0]*len(Cameras) + [0, 0]
    tmp_logit = torch.zeros_like(torch.unsqueeze(Cameras[0][0]['logits'], dim=0))
    for i in tqdm(range(len(Cameras[0]))):
        name_infer = []
        logit_infer = []
        gt_label = []
             
        for logit in Cameras:
            name_infer.append(logit[i]['name'])
            logit_infer.append(torch.unsqueeze(logit[i]['logits'], dim=0))
            gt_label.append(torch.unsqueeze(logit[i]['label'], dim=0))
            # tmp_logit += torch.unsqueeze(logit[i]['logits'], dim=0)
        tmp = copy.deepcopy(logit_infer)
        logit_infer.append(tmp_logit)
        logit_infer.append(tmp)
        
        #!此处Driver100数据集多视角要注释
        # for name in name_infer:
        #     assert name == name_infer[0], 'Error'
        
        for label in gt_label:
            assert label == gt_label[0], 'Error'
            
        # name1= Camera1[i]['name']
        # name2= Camera2[i]['name']
        # assert name1 == name2, 'The image is not the same image!'
        # label1= torch.unsqueeze(Camera1[i]['label'], dim=0)
        # label2= torch.unsqueeze(Camera2[i]['label'], dim=0)
        # assert label1.item() == label2.item(), 'The label is not the same label!'
        # logit1 = torch.unsqueeze(Camera1[i]['logits'], dim=0)
        # logit2 = torch.unsqueeze(Camera2[i]['logits'], dim=0)
        
        
        for i in range(len(logit_infer)):
            if i==len(logit_infer)-2:
                continue
            if i<len(logit_infer)-1:
                prob, _ = predict(logit_infer[i], gt_label[0], loss_f_s[i])
            elif i==len(logit_infer)-1:
                prob, _ = predictMultiChannelFusion(logit_infer[i], gt_label[0], loss_f_s)
            correct_1[i] += prob[0].cpu()
            correct_3[i] += prob[1].cpu()
            correct_5[i] += prob[2].cpu()
            
        # prob, _ = predict(logit1, label1, loss_f)
        # correct_1[0] += prob[0].cpu()
        # correct_3[0] += prob[1].cpu()
        # correct_5[0] += prob[2].cpu()
        # # logits_class = prob[3].squeeze(0)
        # # truth_class = label1

        # prob, _ = predict(logit2, label2, loss_f)
        # correct_1[1] += prob[0].cpu()
        # correct_3[1] += prob[1].cpu()
        # correct_5[1] += prob[2].cpu()
        
        # prob, _ = predict((logit1+logit2), label1, loss_f)
        # correct_1[2] += prob[0].cpu()
        # correct_3[2] += prob[1].cpu()
        # correct_5[2] += prob[2].cpu()

        # prob, _ = predictMultiChannelFusion([logit1, logit2], label1, loss_f)
        # correct_1[3] += prob[0].cpu()
        # correct_3[3] += prob[1].cpu()
        # correct_5[3] += prob[2].cpu()
        
    top_1 = list(map(lambda x : x/len(Cameras[0]), correct_1))
    print(top_1)
    print()