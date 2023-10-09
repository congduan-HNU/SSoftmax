'''

   ┏┓　　　┏┓
 ┏┛┻━━━┛┻┓
 ┃　　　　　　　┃
 ┃　　　━　　　┃
 ┃　＞　　　＜　┃
 ┃　　　　　　　┃
 ┃...　⌒　...　┃
 ┃　　　　　　　┃
 ┗━┓　　　┏━┛
     ┃　　　┃　
     ┃　　　┃
     ┃　　　┃
     ┃　　　┃  神兽保佑
     ┃　　　┃  代码无bug　　
     ┃　　　┃
     ┃　　　┗━━━┓
     ┃　　　　　　　┣┓
     ┃　　　　　　　┏┛
     ┗┓┓┏━┳┓┏┛
       ┃┫┫　┃┫┫
       ┗┻┛　┗┻┛

Author: Cong Duan
Date: 2023-05-05 17:11:21
LastEditTime: 2023-08-21 16:13:53
LastEditors: your name
Description: 本脚本用于迁移数据集，由于SFD和AUCDDV1的数据集种类只有10种，而100Driver有22种，本脚本的函数用于数据集 ground truth 的互相转换以共多数据集测试
FilePath: /Driver-Action-Monitor/trainAcrossDatasets/datasetTransfer.py
可以输入预定的版权声明、个性签名、空行等
'''

"""
不同数据集的 Ground Truth 规则描述
SFD : 文件位置： trainAcrossDatasets/groundtruth/SFD
AUCDDV1: 文件位置： trainAcrossDatasets/groundtruth/AUCDDV1
100Driver: 文件位置: trainAcrossDatasets/groundtruth/100Driver

Details:
SFD:    Class Name     Simple Name     gt       gt(100Driver)
        Drive Safety        C1          0       0
        Text Right          C2          1       6
        Talk Right          C3          2       4
        Text Left           C4          3       5
        Talk Left           C5          4       3
        Adjust Radio        C6          5       17,18
        Drink               C7          6       15,16
        Reach Behind        C8          7       19
        Hair & Makeup       C9          8       7
        Talk Passenger      C10         9       21
        
AUCDDV1:    Class Name     Simple Name     gt
            Drive Safety        C1          0
            Text Right          C2          1
            Talk Right          C3          2
            Text Left           C4          3
            Talk Left           C5          4
            Adjust Radio        C6          5
            Drink               C7          6
            Reach Behind        C8          7
            Hair & Makeup       C9          8
            Talk Passenger      C10         9

100Driver:      Class Name              Simple Name     gt          gt(SFD/AUCDDV1)
                Drive_Safe                  C1          0               0
                Sleep                       C2          1               -1
                Yawning                     C3          2               -1
                Talk_Left                   C4          3               4
                Talk_Right                  C5          4               2
                Text_Left                   C6          5               3
                Text_Right                  C7          6               1
                Make_Up                     C8          7               8
                Look_Left                   C9          8               -1
                Look_Right                  C10         9               -1
                Look_Up                     C11         10              -1
                Look_Down                   C12         11              -1
                Smoke_Left                  C13         12              -1
                Smoke_Right                 C14         13              -1
                Smoke_Mouth                 C15         14              -1
                Eat_Left                    C16         15              6
                Eat_Right                   C17         16              6
                Operate_Radio               C18         17              5
                Operate_GPS                 C19         18              5
                Reach_Behind                C20         19              7
                Leave_Steering_Wheel        C21         20              -1
                Talk_to_Passenger           C22         21              9

"""
import torch

SFD_AUCDD_CLASSNAME = ['C1: Drive Safety',
                       'C2: Text Right',
                       'C3: Talk Right',
                       'C4: Text Left',
                       'C5: Talk Left',
                       'C6: Adjust Radio',
                       'C7: Drink',
                       'C8: Reach Behind',
                       'C9: Hair & Makeup',
                       'C10: Talk Passenger'
                       ]


def fSFD2Driver100(gt: int):
    if gt==0:
        gt_new = 0
    elif gt==1:
        gt_new = 6
    elif gt==2:
        gt_new = 4
    elif gt==3:
        gt_new = 5
    elif gt==4:
        gt_new = 3
    elif gt==5:
        gt_new = 17 # =18
    elif gt==6:
        gt_new = 15 # =16
    elif gt==7:
        gt_new = 19
    elif gt==8:
        gt_new = 7
    elif gt==9:
        gt_new = 21
    return gt_new


def fDriver1002SFD(gt: int):
    if gt==0:
        gt_new = 0
    elif gt==1:
        gt_new = -1
    elif gt==2:
        gt_new = -1
    elif gt==3:
        gt_new = 4
    elif gt==4:
        gt_new = 2
    elif gt==5:
        gt_new = 3 
    elif gt==6:
        gt_new = 1
    elif gt==7:
        gt_new = 8
    elif gt==8:
        gt_new = -1
    elif gt==9:
        gt_new = -1
    elif gt==10:
        gt_new = -1
    elif gt==11:
        gt_new = -1
    elif gt==12:
        gt_new = -1
    elif gt==13:
        gt_new = -1
    elif gt==14:
        gt_new = -1
    elif gt==15:
        gt_new = 6
    elif gt==16:
        gt_new = 6
    elif gt==17:
        gt_new = 5
    elif gt==18:
        gt_new = 5
    elif gt==19:
        gt_new = 7
    elif gt==20:
        gt_new = -1
    elif gt==21:
        gt_new = 9
    return gt_new

'''
description: 如果在计算top-k时，标签要拓展成预测的形状，但17=18，15=16. 
即SFD的标签训练的数据预测Driver100时，如果原标签是5，那么训练时是17，在计
算ACC时，要将Driver100的gt中17和18统一
param {torch} args
return {*}
'''
def gt_Driver100_when_SFD2Driver100(args: torch.Tensor):
    tmp = args.clone()
    tmp[:, 17] = torch.abs(args[:, 17] - args[:, 18])
    tmp[:, 15] = torch.abs(args[:, 15] - args[:, 16])
    return tmp


if __name__ == '__main__':
    a = torch.Tensor([[True, False, True],
                       [False, True, False],
                       [False, True, False]])

    c = a.clone()
    a[:, 1] = torch.abs(c[:, 0] - c[:, 1])
    print(a)
