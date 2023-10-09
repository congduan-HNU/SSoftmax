'''
Author: Cong Duan
Date: 2023-06-22 09:28:01
LastEditTime: 2023-10-09 17:43:40
LastEditors: your name
Description: 
FilePath: /SSoftmax/trainAcrossDatasets/test_scorelossPlus.py
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


'''
description: 绘制混淆矩阵图
param {*} cmtx
param {*} num_classes
param {*} class_names
param {*} figsize
param {*} valFomat
return {*}
'''
def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None, valFomat=int):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, horizontalalignment="right", rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            # format(cmtx[i, j], "d") if cmtx[i, j] != 0 else ".",
            format(cmtx[i, j], ".2f") if (cmtx[i, j] != 0 and valFomat==float) else format(cmtx[i, j], "d") if (cmtx[i, j] != 0 and valFomat==int) else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure

'''
description: 绘制混淆矩阵
return {*}
'''
def add_confusion_matrix(
    cmtx,
    num_classes,
    subset_ids=None,
    class_names=None,
    figsize=None,
    normalize=False
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
    if normalize:
        cmtx = cmtx.astype('float') / cmtx.sum(axis=1)[:, np.newaxis]
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every classes.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
            valFomat=float if normalize else int,
        )
        # Add the confusion matrix image to writer.
        # writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)
        return sub_cmtx


class DatasetPrepare(object):
    def __init__(self, project: ProjectInfo, option: Options):
        self.project = project
        self.opt = option
        self.opt.debug = self.project.IsDebug
        self.noaugmental = MyCoTransform_numpy(self.opt, DoAugment=False)
        self.num_workers = 6

        # self.opt.dataset = self.opt.datasetsList[0]
        # self.opt.classes = self.opt.dataset["classes"]

        if ("class_Samples" in self.opt.dataset.keys()):
            self.opt.classSamples = self.opt.dataset["class_Samples"]

        # 测试集 (测试)
        self.testSet = ClassifyDataset(self.opt, self.opt.dataset, self.opt.dataset["ImageTestPath"],
                                        osp.join(self.project.ROOT, self.opt.dataset["TestLabelPath"]),
                                        transform=self.noaugmental)
        self.test_loader = DataLoader(self.testSet, batch_size=self.opt.batch_size, shuffle=False,
                                            num_workers=self.num_workers, pin_memory=True)

    def loadDataset(self):
        for _data in tqdm(self.test_loader, position=0, desc='load test dataset'):
            pass

def test(option: Options, net: nn.Module, dataLoader: DataLoader, loss_f: ScoreLossPlus):
    if option.classes is None:
        printPlus("Error: option.classes is None", frontColor=31)
        sys.exit()
    confusion_matrix = np.zeros((option.classes, option.classes), dtype=np.int32)
    # 评估计数及参数初始化
    net.eval()
    correct_1 = 0
    correct_3 = 0
    correct_5 = 0
    labels_TestDataset = []
    logits_TestDataset = []
    fault_samples = OrderedDict({'name':[],'real':[],'predict':[]})
    t_SNE_feature = []
    t_SNE_label = []
    with torch.no_grad():
        for batch, data in enumerate(dataLoader, start=0):  # 同时获得索引和值
            images = data['images'].to(device=option.device, dtype=option.dtype)
            labels = data['labels'].to(device=option.device)
            names = data['items']

            logits = net(images)
            # score loss added
            logits = logits.view(-1, loss_f.scoreLevel, opt.classes)
            logits = nn.Softmax(dim=1)(logits)

            prob, logits = predict(logits, labels, loss_f)
            correct_1 += prob[0].cpu()
            correct_3 += prob[1].cpu()
            correct_5 += prob[2].cpu()
            logits_class = prob[3].squeeze(0)
            truth_class = labels

            _logits_class = [
                torch.max(logits[i], dim=0).values.argmax(0).item()
                for i in range(0, logits.size()[0])
            ]
            logits_TestDataset.extend(logits.to('cpu').numpy().tolist())
            # logits_TestDataset.extend(_logits_class)#score loss use note!!!!!!!!!!!!!!!!!!
            labels_TestDataset.extend(labels.to('cpu').numpy().tolist())

            # 行为真实类，列为预测类(混淆矩阵)
            for i in range(0, truth_class.shape[0]):
                confusion_matrix[truth_class[i].item(), logits_class[i].item()] += 1

            # 预测错误样本输出
            _labels = list(copy.deepcopy(labels).to('cpu').numpy())
            _logits = list(copy.deepcopy(logits).argmax(1).to('cpu').numpy())
            _name = list(names)
            # _logits = _logits_class #score loss use note!!!!!!!!!!!!!!!!!! # ! 此处存在bug导致faultsamples出现问题
            for i in range(0, len(labels)):
                if _labels[i]!=_logits[i]:
                    fault_samples['real'].append(_labels[i])
                    fault_samples['predict'].append(_logits[i])
                    fault_samples['name'].append(_name[i])

            t_SNE_feature.append(copy.deepcopy(logits).to('cpu'))
            t_SNE_label.extend(list(copy.deepcopy(truth_class).to('cpu').numpy()))

    assert np.sum(confusion_matrix) == len(dataLoader.dataset), "评估结果不正确，混淆矩阵总数与被评估数据集不一致"
    metric = Metric(confusion_matrix)

    logits_TestDataset = np.asarray(logits_TestDataset)
    labels_TestDataset = np.asarray(labels_TestDataset)

    ROC_Results = {'logits_TestDataset':logits_TestDataset, 'labels_TestDataset':labels_TestDataset, 'fault_samples':fault_samples}

    accs_test = [correct_1/len(dataLoader.dataset), correct_3/len(dataLoader.dataset), correct_5/len(dataLoader.dataset)]

    t_SNE_feature = torch.cat(t_SNE_feature, 0)
    t_SNE_label = np.asarray(t_SNE_label).astype(int)
    t_SNE = {"t_SNE_feature":t_SNE_feature, "t_SNE_label":t_SNE_label}

    return metric, accs_test, ROC_Results, t_SNE

# 获得Top-1. Top-3, Top5 Acc
def predict(logits:torch.Tensor, labels:torch.Tensor, loss_f:ScoreLossPlus):
    # print("pred:")
    # print(logits[0])
    logits = logits * loss_f.droptLowHighscore
    # print("score:")
    # print(logits[0])
    # print("gt:")
    # print(labels[0])
    logits = torch.sum(logits, dim=1)
    tmp = torch.sum(logits, dim=1).unsqueeze(-1)
    logits= torch.div(logits, tmp)

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

    return opt

if __name__ == '__main__':
    import argparse
    import datasetDriver100
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', default='default', type=str, required=False)
    parser.add_argument('-dataset', default='default', type=str, required=False)
    parser.add_argument('-level', default=-1, type=int, required=False)
    args = parser.parse_args()
    
    if args.level == -1:
            args.level = 10
    else:
        pass
    
    # 工程项目准备
    project = ProjectInfo()
    if args.ckpt == 'default':
        ckptFolderFolder = '/home/caomen/Desktop/DC/ablationExperiments/Driver-Action-Monitor/trainAcrossDatasets/log/FineturnTrainScoreLossPlus-2023-09-09-14_04_03'
    else:
        ckptFolderFolder = args.ckpt

    ckptFolders = os.listdir(ckptFolderFolder)
    
    for ckptFolder in ckptFolders:
        if ckptFolder != '2023-09-11-18_51_27': #!用于控制选择哪个模型进行测试
            continue
        printPlus(f"Start: {ckptFolder}", frontColor=32)
        cfgFile = osp.join(ckptFolderFolder, ckptFolder, 'config.txt')
        opt = loadCfg(cfgFile)
        
        files = os.listdir(osp.join(ckptFolderFolder, ckptFolder))      
        for file in files:
            template = f'{opt.model}\S+valBest-0.pth'
            if re.search(template, file):
                ckptFile = re.search(template, file).group()

        ckptFile = osp.join(ckptFolderFolder, ckptFolder, ckptFile)
         
        if args.dataset == 'default':
            testDataset = datasetDriver100.Driver100_Cross_Individual_Vehicle_D4_Lynk_Test
        else:
            testDataset = eval(args.dataset)
        opt.dataset = testDataset
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net, level = eval(opt.model)(opt, level=args.level)
        state_dict = torch.load(ckptFile)
        printPlus("Loaded the pretrain model!\n", 32)
        net.load_state_dict(state_dict=state_dict, strict=False)
        if opt.device == "cuda":
            net.to('cuda')
            cudnn.benchmark = True
        
        loss_f = ScoreLossPlus(opt, level)
        
        prepared_data = DatasetPrepare(project, opt)

        metric, accs_test, ROC_Results, tSNE_Results = test(opt, net, prepared_data.test_loader, loss_f)

        saveTestResultFolder = osp.join(ckptFolderFolder, ckptFolder, f'Test_Result_On_{opt.dataset["DataName"]}')
        creat_folder(saveTestResultFolder)
        resultFile = osp.join(saveTestResultFolder, 'result.txt')

        _info = 'Test     | ACC: %.5f%% | Pre: %.5f%% | Recall: %.5f%% | F-1: %.5f%% | ACC-Top1: %.3f%% | ACC-Top3: %.3f%% | ACC-Top5: %.3f%% |' \
                        % (100.*metric.accuracy, 100.*metric.mean_precision, 100.*metric.mean_sensitivity, 100.*metric.Macro_F1, 100.*accs_test[0], 100.*accs_test[1], 100.*accs_test[2])
        printPlus(_info, _file=resultFile)

        # 输出测试结果信息
        printPlus('Export Result ...', frontColor=32)
        metric.exportResult(resultFile)

        # 输出混淆矩阵
        printPlus('Export confusion matrix ...', frontColor=32)
        class_names = opt.dataset["class_names"] if opt.dataset["class_names"] else None
        confusion_matrix_figure = add_confusion_matrix(metric.confusion_matrix, class_names=class_names,
                                num_classes=opt.classes, figsize=[12.8, 9.6], normalize=False)
        confusion_matrix_figure.savefig(osp.join(saveTestResultFolder, 'confusion_matrix.png'))

        # 输出ROC，PR曲线信息
        printPlus('Export ROC, PR ...', frontColor=32)
        ROC_macro, out_macro, out_micro = computeROCandAUC(prob=ROC_Results['logits_TestDataset'], labels=ROC_Results['labels_TestDataset'], samplesNum=len(prepared_data.testSet), n_classes=opt.classes)
        ROC_macro.savefig(osp.join(saveTestResultFolder, 'macroROCandAUC.png'))
        np.savetxt(osp.join(saveTestResultFolder, "ROCDataMacro.csv"), out_macro, fmt="%.3f", delimiter=',', header="x, y")
        np.savetxt(osp.join(saveTestResultFolder, "ROCDataMicro.csv"), out_micro, fmt="%.3f", delimiter=',', header="x, y")

        PR_macro, PR_macro_data = computePR(prob=ROC_Results['logits_TestDataset'], labels=ROC_Results['labels_TestDataset'], samplesNum=len(prepared_data.testSet), n_classes=opt.classes)
        PR_macro.savefig(osp.join(saveTestResultFolder, 'macroPR.png'))
        np.savetxt(osp.join(saveTestResultFolder, "PRDataMacro.csv"), PR_macro_data, fmt="%.3f", delimiter=',', header="x, y")

        # 输出预测错误样本路径
        printPlus('Export faultSamples ...', frontColor=32)
        faultSamples = pd.DataFrame(ROC_Results['fault_samples'])
        faultSamples.to_csv(os.path.join(saveTestResultFolder, "faultsamples.txt"),sep='\t',index=0)

        # 输出t-SNE图
        printPlus('Export t-SNE ...', frontColor=32)
        t_SNE(opt, tSNE_Results, saveTestResultFolder)

        printPlus('Finished!' , frontColor=32)
    sys.exit()

    
    



