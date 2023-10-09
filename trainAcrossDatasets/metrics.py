# _*_coding:utf-8_*_
# __author:    duancong
# __date:      4/20/23 1:54 PM
# __filename:  metrics.py
import numpy as np
from pythonUtils import *
from pythonUtils import colors as allColor

from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle
from sklearn import manifold
import pandas as pd
import re

class Metric(object):
    def __init__(self, confusion_matrix):
        """
        计算评估指标，输入为混淆矩阵形式如下
                predict1    predict2    predict3 ...
        Actual1
        Actual2
        Actual3
        ...
        :param confusion_matrix:
        """
        self.confusion_matrix = confusion_matrix
        shape = self.confusion_matrix.shape
        assert shape[0]==shape[1] and shape[0]!=0, 'confusion matrix is not good'
        self.classes = shape[0]
        self.recompute()

    def __add__(self, other):
        assert self.confusion_matrix.shape == other.confusion_matrix.shape, "Metric add is error!!"
        self.confusion_matrix = self.confusion_matrix + other.confusion_matrix
        shape = self.confusion_matrix.shape
        assert shape[0] == shape[1] and shape[0] != 0, 'confusion matrix is not good'
        self.classes = shape[0]
        new_metric = Metric(self.confusion_matrix)
        new_metric.recompute()
        return new_metric

    def recompute(self):
        self.compute()
        self.compute_accuracy()
        self.compute_precision()
        self.compute_sensitivity()
        self.compute_Macro_F1()

    def compute(self):
        """
        计算相关参数
        :return:
        """
        # cfm = copy.deepcopy(self.confusion_matrix)
        # self.TP_i = np.diag(cfm)
        # self.FP_i = np.zeros(self.TP_i.shape)
        # for i in range(0, self.FP_i.size):
        #     self.FP_i[i] = np.sum(cfm[:,i])-cfm[i,i]
        # self.TN_i = np.zeros(self.TP_i.shape)
        # for i in range(0, self.FP_i.size):
        #     self.TN_i[i] = np.sum(cfm)-np.sum(cfm[i,:])-np.sum(cfm[:,i])+cfm[i,i]
        # self.FN_i = np.zeros(self.TP_i.shape)
        # for i in range(0, self.FP_i.size):
        #     self.FN_i[i] = np.sum(cfm[i,:])-cfm[i,i]
        # #self.AA = (self.TP_i+self.TN_i)/(self.TP_i+self.FP_i+self.TN_i+self.FN_i)
        # self.P_i = self.TP_i/(self.TP_i+self.FP_i)
        # self.R_i = self.TP_i/(self.TP_i+self.FN_i)
        # self.F1_i = 2*self.P_i*self.R_i/(self.P_i+self.R_i)
        # self.F1_i_mean = np.mean(self.F1_i)

        self.TP_TN = np.diag(self.confusion_matrix).sum(0)
        self.TP_TN_FP_FN = np.zeros((1, self.classes))
        for i in range(0, self.classes):
            self.TP_TN_FP_FN[0, i] = self.TP_TN + self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - 2 * \
                                self.confusion_matrix[i, i]
        self.TP_FN = np.zeros((1, self.classes))
        for i in range(0, self.classes):
            self.TP_FN[0, i] = self.confusion_matrix[i, :].sum()
        self.TP_FP = np.zeros((1, self.classes))
        for i in range(0, self.classes):
            self.TP_FP[0, i] = self.confusion_matrix[:, i].sum()
        self.diag = np.diag(self.confusion_matrix)

        self.sum = self.confusion_matrix.sum().astype(np.float)



    def compute_accuracy(self):
        """
        计算准确度,准确率 = 正确预测的正反例数/总数, ACC = (TP+TN)/(TP+TN+FP+FN)
        :return: class_acc, mean_acc
        """
        self.classAP = []
        for i in range(0, self.classes):
            self.classAP.append(self.TP_TN.astype(np.float) / self.TP_TN_FP_FN[0, i].astype(np.float))
        self.meanAP = np.mean(self.classAP)

        self.accuracy = np.true_divide(self.TP_TN.sum(), self.sum)

    def compute_precision(self):
        """
        计算精确度,查准率、精确率=正确预测到的正例数/预测正例总数, precision = TP/(TP+FP)
        :return: class_precision, mean_precision
        """
        self.class_precision = np.true_divide(self.diag, self.TP_FP+0.000001)
        self.class_precision = np.reshape(self.class_precision, -1)
        self.mean_precision = np.mean(self.class_precision)

    def compute_sensitivity(self):
        """
        计算召回率,查全率、召回率=正确预测到的正例数/实际正例总数, sensitivity = TP/(TP+FN)
        :return: class_sensitivity, mean_sensitivity
        """
        self.class_sensitivity = np.true_divide(self.diag, self.TP_FN+0.000001)
        self.class_sensitivity = np.reshape(self.class_sensitivity, -1)
        self.mean_sensitivity = np.mean(self.class_sensitivity)

    def compute_Macro_F1(self):
        """
        计算Macro-F1指标，Macro-F1 = mean(2*(precision*sensitivity)/(precision+sensitivity))
        :return:
        """
        self.class_f1 = 2 * np.multiply(self.class_precision, self.class_sensitivity)/np.add(self.class_precision, self.class_sensitivity+0.000001)
        self.Macro_F1 = np.mean(self.class_f1)

    def exportResult(self, resultFile):
        printPlus("best meanAP:{}".format(self.meanAP), _file=resultFile)
        printPlus("best classAP:{}".format(self.classAP), _file=resultFile)
        printPlus("best mean_precision:{}".format(self.mean_precision), _file=resultFile)
        printPlus("best class_precision:{}".format(self.class_precision), _file=resultFile)
        printPlus("best mean_sensitivity:{}".format(self.mean_sensitivity), _file=resultFile)
        printPlus("best class_sensitivity:{}".format(self.class_sensitivity), _file=resultFile)
        printPlus("best Macro-F1:{}".format(self.Macro_F1), _file=resultFile)
        printPlus("best class_Macro_F1:{}".format(self.class_f1), _file=resultFile)
        printPlus("best accuracy:{}".format(self.accuracy), _file=resultFile)
        printPlus(f"confusion matrix: \n{self.confusion_matrix}", _file=resultFile)



'''
description: 计算ROC曲线和AUC使用
param {*} prob
param {*} labels
param {*} samplesNum
param {*} n_classes
return {*}
'''
def computeROCandAUC(prob, labels, samplesNum, n_classes):
    """
    prob：预测的概率值 [m, n],m为测试样本数，n为类别数，内部的值为归一化概率值
    labels:标签类别 [m, 1], m为样本数， 值为每一样本的标签值，从0开始
    """
    # labels one-hot 操作
    labels_onthot = np.zeros((prob.shape[0], prob.shape[1]))
    for i in range(labels.shape[0]):
        labels_onthot[i, labels[i]] = 1
    assert prob.shape == labels_onthot.shape, "the shape of \"input\" in fun: computeROCandAUC is error!!!"
    assert prob.shape[0] == samplesNum, "The numbers of samples is error in fun: computeROCandAUC"
    assert prob.shape[1] == n_classes, "The numbers of n_classes is error in fun: computeROCandAUC"

    y_label = labels_onthot
    y_score = prob
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    # plt.show()

    x_macro = np.expand_dims(fpr["macro"], 1)
    y_macro = np.expand_dims(tpr["macro"], 1)
    out_macro = np.concatenate([x_macro, y_macro], 1)

    x_micro = np.expand_dims(fpr["micro"], 1)
    y_micro = np.expand_dims(tpr["micro"], 1)
    out_micro = np.concatenate([x_micro, y_micro], 1)
    return plt, out_macro, out_micro

'''
description: 计算PR曲线使用
param {*} prob
param {*} labels
param {*} samplesNum
param {*} n_classes
return {*}
'''
def computePR(prob, labels, samplesNum, n_classes):
    # score_array = np.array(logits_evalDataset)
    # # 将label转换成onehot形式
    # label_tensor = torch.tensor(labels_evalDataset)
    # label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    # label_onehot = torch.zeros(label_tensor.shape[0], opt.classes)
    # label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    # label_onehot = np.array(label_onehot)

    labels_onthot = np.zeros((prob.shape[0], prob.shape[1]))
    for i in range(labels.shape[0]):
        labels_onthot[i, labels[i]] = 1
    assert prob.shape == labels_onthot.shape, "the shape of \"input\" in fun: computeROCandAUC is error!!!"
    assert prob.shape[0] == samplesNum, "The numbers of samples is error in fun: computeROCandAUC"
    assert prob.shape[1] == n_classes, "The numbers of n_classes is error in fun: computeROCandAUC"

    label_onehot = labels_onthot
    score_array = prob
    # print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
    # print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum]) onehot

    # 调用sklearn库，计算每个类别对应的precision和recall
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(n_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])
        # print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])
    # micro
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                              score_array.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["micro"]))

    # 绘制所有类别平均的pr曲线
    plt.figure()
    plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision_dict["micro"]))

    x_macro = np.expand_dims(recall_dict["micro"], 1)
    y_macro = np.expand_dims(precision_dict["micro"], 1)
    out_macro = np.concatenate([x_macro, y_macro], 1)
    return plt, out_macro



'''
description: 计算t-SNE
param {*} opt
param {*} dataDict
param {*} savePath
return {*}
'''
def t_SNE(opt, dataDict, savePath):
    """

    Args:
        opt:
        _input: 要可视化的特征，[num, dim]， num为特征的数量应该是样本数, dim为每一个特征的原始维度，应该是网络输出的向量的维度

    Returns:

    """
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(dataDict.get("t_SNE_feature"))
    # print(x_ts.shape)
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts-x_min) / (x_max-x_min+0.000000001)

    # 设置散点形状
    maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H', '+', 'x', '|', '_']
    # 设置散点颜色
    colors = [allColor.get('bisque'), allColor.get('lightgreen'), allColor.get('slategray'), allColor.get('cyan'), allColor.get('blue'), allColor.get('lime'), 'r', allColor.get('violet'), 'm', allColor.get('peru'), allColor.get('olivedrab'),
              allColor.get('hotpink'), allColor.get('olive'), allColor.get('sandybrown'), allColor.get('pink'), allColor.get('purple')]
    # 图例名称
    Label_Com = opt.dataset["class_names"]
    if Label_Com is None:
        Label_Com = [str(i) for i in range(opt.classes)]
    # 设置字体格式
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
            #  'size': 32,
             }

    S_lowDWeights = x_final
    Trure_labels = dataDict.get("t_SNE_label")
    name = 't_SNE'

    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    # print(S_data)
    # print(S_data.shape)  # [num, 3]
    plt.figure(figsize=(10, 10))
    for index in range(opt.classes):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        makerIDX = index % len(maker)
        
        edgecolorsIDX = index % len(colors)

        c_value = colors[index % len(colors)] if index < len(colors) else None

        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[makerIDX], c=c_value, edgecolors=colors[edgecolorsIDX], alpha=0.65, label=Label_Com[index])

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
    plt.legend(loc='upper right', prop=font1)
    # plt.title(name, fontsize=32, fontweight='normal', pad=20)
    # fig = plt.figure(figsize=(10, 10))
    # plt.show()
    plt.savefig(os.path.join(savePath, "t_SNE.png"), bbox_inches="tight")
    plt.close()
    return


def read_and_collect(_file:str, _key:str):
    with open(_file, 'r') as f:
        contents = f.readlines()
    values = []
    for line in contents:
        value_str = re.findall(f'{_key}:\s(.+?)\%\s\|', line,  re.IGNORECASE)
        values.append(float(value_str[0]))
    values.sort(reverse=True)
    return values

def statics(metric:list, topk:int):
    reuslt = {}
    metric_sub = metric[:topk]
    metric_sub_arr = np.asarray(metric_sub)
    reuslt.setdefault('max', metric_sub_arr.max())
    reuslt.setdefault('min', metric_sub_arr.min())
    reuslt.setdefault('mean', round(metric_sub_arr.mean(), 2))
    reuslt.setdefault('right_space', round(metric_sub_arr.max()-reuslt.get('mean'), 2))
    reuslt.setdefault('left_space', round(metric_sub_arr.min()-reuslt.get('mean'), 2))
    return reuslt

def statics_metrics(file, key, topk):
    data = read_and_collect(file, key)
    data = statics(data, topk)
    return data