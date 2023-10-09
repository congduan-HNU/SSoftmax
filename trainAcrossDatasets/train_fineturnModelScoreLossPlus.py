'''
Author: Cong Duan
Date: 2023-06-18 18:28:17
LastEditTime: 2023-10-09 17:37:18
LastEditors: your name
Description: Fineturn-training with the S-Softmax Classifer
FilePath: /SSoftmax/trainAcrossDatasets/train_fineturnModelScoreLossPlus.py
可以输入预定的版权声明、个性签名、空行等
'''


import sys
import os
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import torchvision.utils
print(sys.path)
from pythonUtils import *

Folder = osp.basename(osp.dirname(osp.abspath(__file__)))

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
# import wandb
import numpy as np

from options import Options
import train_config as config
from prepare_data import DatasetPrepare
# from eval import evaluate
from loss import ScoreLossPlus
from model import *
from metrics import Metric, statics_metrics
np.set_printoptions(linewidth=1000)

def train(project: ProjectInfo, option: Options, prepared_data: DatasetPrepare, level):
    logPath = osp.join(project.ROOT, Folder, "log", f"{'Debug' if project.IsDebug else 'FineturnTrainScoreLossPlus'}-{project.StartTime}", timeClock())
    creat_folder(logPath)

    fileLog = osp.join(logPath, "log.txt")
    trainLog = osp.join(logPath, "trainlog.txt")
    if type(prepared_data.evaluate_loader) is list:
        valLog = [osp.join(logPath, f"vallog_{i}.txt") for i in range(len(prepared_data.evaluate_loader))]
    else:
        valLog = osp.join(logPath, "vallog.txt")
    if type(prepared_data.test_loader) is list:
        testLog = [osp.join(logPath, f"testlog_{i}.txt") for i in range(len(prepared_data.test_loader))]
    else:
        testLog = osp.join(logPath, "testlog.txt")

    fileConfig = osp.join(logPath, "config.txt")
    writer = SummaryWriter(osp.join(logPath, 'runs'))

    printPlus(f"{timeClock()}: Enviroment Created!", 32, _file=fileLog)
    printPlus("Config Saved!", 32, _file=fileLog)
    option.printInfo(fileConfig)

    net, level = eval(option.model)(option, level)
    if option.write:
        writer.add_graph(net, torch.rand(1, option.inchannel, *option.size))

    if option.pretrain and osp.exists(option.pretrain_pth):
        state_dict = torch.load(option.pretrain_pth)
        printPlus("Loaded the pretrain model!\n", 32, fileLog)
        net.load_state_dict(state_dict=state_dict, strict=False)

    # if option.paraminit:
    #     net.apply(weights_init)
    printPlus("Fineturen Pretrained Model, Do not Weights initialized.", 31, _file=fileLog)

    if option.device == "cuda":
        net.to('cuda')
        cudnn.benchmark = True
    if option.device == "cuda" and option.mul_gpu_train:
        net = nn.DataParallel(net)

    # loss
    loss_f = ScoreLossPlus(option, level)
    loss_f.printInfo(file=fileLog)

    # optim
    net.requires_grad_ (True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.000001, betas=(0.9, 0.999), weight_decay=0.001)
    
    for name, paramer in net.named_parameters():
        printPlus(f'{name} {paramer.requires_grad}', _file=fileLog)
    
    # lr
    scheduler = MultiStepLR(optimizer, milestones=option.lr_decay_milestones, gamma=option.lr_decay_gamma)

    startEpoch = 1
    global_batch = 0
    best_val_acc = [-1]*len(prepared_data.evaluate_loader) if (type(prepared_data.evaluate_loader) is list) else -1
    best_test_acc = [-1]*len(prepared_data.test_loader) if (type(prepared_data.test_loader) is list) else -1
    
    num_epoch = 5 if project.IsDebug else option.epochs
    option.test_rate = 1 if project.IsDebug else option.test_rate
    # prepared_data.loadDataset()
    for epoch in range(startEpoch, num_epoch+1):
        epoch_loss = 0.0
        epoch_score_dis= 0.0
        epoch_score =0.0
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0
        total = 0
        batchCount=0
        for batch, data in enumerate(prepared_data.train_loader, start=0):
            net.train()
            global_batch += 1
            batchCount += 1
            images = data['images'].to(device=option.device, dtype=option.dtype)
            labels = data['labels'].to(device=option.device)
            logits = net(images) 
            #  score loss added
            logits = logits.view(-1, level, opt.classes)
            logits = nn.Softmax(dim=1)(logits)
            
            loss, score_dis, score = loss_f(logits, labels)

            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_score_dis = score_dis.item()
            batch_score = score.item()
            epoch_score_dis +=batch_score_dis
            epoch_score += batch_score

            # the S-Softmax Classifer
            prob = predict(logits, labels, loss_f)
            correct_1 += prob[0].cpu()
            correct_3 += prob[1].cpu()
            correct_5 += prob[2].cpu()
            total += labels.size(0)

            if global_batch % 50 == 0:
                _info = 'Train    | Epoch: %d (lr=%e) | LocalBatch: %d | Step: %d | batchLoss: %.5f(%.5f,%.5f) | epochLoss: %.5f(%.5f,%.5f) | Acc-Top1: %.3f%% (%d/%d)| Acc-Top3: %.3f%% (%d/%d)| Acc-Top5: %.3f%% (%d/%d)|' \
                        % (epoch, scheduler.get_last_lr()[0], batch+1, global_batch, batch_loss, batch_score_dis, batch_score, epoch_loss / (batch + 1), epoch_score_dis/(batch+1), epoch_score/(batch+1), \
                           100. * float(correct_1) / total, correct_1, total, 100. * float(correct_3) / total, correct_3, total, 100. * float(correct_5) / total, correct_5, total)
                printPlus(_info, _file=trainLog)


            if global_batch % (len(prepared_data.train_loader) // (option.test_rate)) == 0:
                if type(prepared_data.evaluate_loader) is list:
                    for i in range(len(prepared_data.evaluate_loader)):
                        eval_result = evaluate(option, net, prepared_data.evaluate_loader[i], loss_f)
                        _info = 'Evaluate-%d| Epoch: %d (lr=%e) | LocalBatch: %d | Step: %d | evalLoss: %.5f(%.5f,%.5f) | Acc-Top1: %.3f%% | Acc-Top3: %.3f%% | Acc-Top5: %.3f%% |' % (
                            i, epoch, scheduler.get_last_lr()[0], batch + 1, global_batch, eval_result[3],eval_result[4],eval_result[5], 100. * eval_result[0], 100. * eval_result[1], 100. * eval_result[2])
                        printPlus(_info, _file=valLog[i])
                        if eval_result[0] > best_val_acc[i]:
                            best_val_acc[i] = max(eval_result[0], best_val_acc[i])
                            ckpt_name = f'{option.model}-{option.dataset["DataName"]}-valBest-{i}.pth'
                            saveModel(option, net, osp.join(logPath, ckpt_name))
                        if option.write:
                            writer.add_scalars(f'Val-{i}', {'val_top-1': 100. * eval_result[0],
                                                    'val_top-3': 100. * eval_result[1],
                                                    'val_top-5': 100. * eval_result[2]}, global_batch)
                            writer.add_scalars('loss', {f'val_loss-{i}': eval_result[3]},  global_batch)
                            writer.add_scalars('score_dis', {f'score_dis-{i}': eval_result[4]},  global_batch)
                            writer.add_scalars('score', {f'score-{i}': eval_result[5]},  global_batch)
                else:     
                    eval_result = evaluate(option, net, prepared_data.evaluate_loader, loss_f)
                    _info = 'Evaluate | Epoch: %d (lr=%e) | LocalBatch: %d | Step: %d | evalLoss: %.5f(%.5f,%.5f) | Acc-Top1: %.3f%% | Acc-Top3: %.3f%% | Acc-Top5: %.3f%% |' % (
                        epoch, scheduler.get_last_lr()[0], batch + 1, global_batch, eval_result[3],eval_result[4],eval_result[5], 100. * eval_result[0], 100. * eval_result[1], 100. * eval_result[2])
                    printPlus(_info, _file=valLog)
                    if eval_result[0] > best_val_acc:
                        best_val_acc = max(eval_result[0], best_val_acc)
                        ckpt_name = f'{option.model}-{option.dataset["DataName"]}-valBest.pth'
                        saveModel(option, net, osp.join(logPath, ckpt_name))
                        
                    if option.write:
                            writer.add_scalars(f'Val', {'val_top-1': 100. * eval_result[0],
                                                    'val_top-3': 100. * eval_result[1],
                                                    'val_top-5': 100. * eval_result[2]}, global_batch)
                            writer.add_scalars('loss', {'val_loss': eval_result[3]},  global_batch)
                            writer.add_scalars('score_dis', {f'score_dis': eval_result[4]},  global_batch)
                            writer.add_scalars('score', {f'score': eval_result[5]},  global_batch)
                            
                if option.write:
                    writer.add_scalars('loss', {'train_loss': epoch_loss / (batch + 1)},  global_batch)
                    writer.add_scalars('Train', {'train_top-1': 100. * float(correct_1) / total,
                                                'train_top-3': 100. * float(correct_3) / total,
                                                'train_top-5': 100. * float(correct_5) / total}, global_batch)
                    writer.add_scalars('lr', {'learn rate': scheduler.get_last_lr()[0], 'epoch': epoch}, global_batch)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if option.dataset["TestLabelPath"] != None:
            if type(prepared_data.test_loader) is list:
                for i in range(len(prepared_data.test_loader)):
                    metric, accs_test = test(option, net, prepared_data.test_loader[i], loss_f)
                    _info = 'Test     | Epoch: %d (lr=%e) | ACC: %.5f%% | Pre: %.5f%% | Recall: %.5f%% | F-1: %.5f%% | ACC-Top1: %.3f%% | ACC-Top3: %.3f%% | ACC-Top5: %.3f%% |' \
                            % (epoch, scheduler.get_last_lr()[0], 100.*metric.accuracy, 100.*metric.mean_precision, 100.*metric.mean_sensitivity, 100.*metric.Macro_F1, 100.*accs_test[0], 100.*accs_test[1], 100.*accs_test[2])
                    printPlus(_info, _file=testLog[i])
                    if option.write:
                        writer.add_scalars('Test-{i}', {'test_top-1': 100. * accs_test[0],
                                                    'test_top-3': 100. * accs_test[1],
                                                    'test_top-5': 100. * accs_test[2]}, epoch)
                    if metric.accuracy > best_test_acc[i]:
                        best_test_acc[i] = max(metric.accuracy, best_test_acc[i])
                        ckpt_name = f'{option.model}-{option.dataset["DataName"]}-testBest-{i}.pth'
                        saveModel(option, net, osp.join(logPath, ckpt_name))
            else:
                metric, accs_test = test(option, net, prepared_data.test_loader, loss_f)
                _info = 'Test     | Epoch: %d (lr=%e) | ACC: %.5f%% | Pre: %.5f%% | Recall: %.5f%% | F-1: %.5f%% | ACC-Top1: %.3f%% | ACC-Top3: %.3f%% | ACC-Top5: %.3f%% |' \
                        % (epoch, scheduler.get_last_lr()[0], 100.*metric.accuracy, 100.*metric.mean_precision, 100.*metric.mean_sensitivity, 100.*metric.Macro_F1, 100.*accs_test[0], 100.*accs_test[1], 100.*accs_test[2])
                printPlus(_info, _file=testLog)
                if option.write:
                    writer.add_scalars('Test', {'metric-top1': metric.accuracy,
                                                'test_top-1': 100. * accs_test[0],
                                                'test_top-3': 100. * accs_test[1],
                                                'test_top-5': 100. * accs_test[2]}, epoch)
                if metric.accuracy > best_test_acc:
                    best_test_acc = max(metric.accuracy, best_test_acc)
                    ckpt_name = f'{option.model}-{option.dataset["DataName"]}-testBest.pth'
                    saveModel(option, net, osp.join(logPath, ckpt_name))



        if option.write:
            # weight and grad visulization
            for name, param in net.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                    writer.add_histogram(
                        f'{name}/grad',
                        param.grad.clone().cpu().data.numpy(),
                        epoch,
                    )

                    # for idx, (name, m) in enumerate(net.named_modules()):
                    #     if name == 'layer_0':
                    #         m = m.layer[0]
                    #         print(m.weight.shape)
                    #         # out_channels, in_channels, k_h, k_w = *list(m.weight.shape)
                    #         in_channles = m.weight.shape[1]
                    #         out_channels = m.weight.shape[0]
                    #         k_w, k_h = m.weight.shape[3], m.weight.shape[2]
                    #         kernel_all = m.weight.view(-1, 1, k_w, k_h)
                    #         kernel_grid = torchvision.utils.make_grid(kernel_all, nrow=in_channles)
                    #         writer.add_image(f'{name}_kernel', kernel_grid, global_step=epoch)
        scheduler.step()
        
    static_metirc(fileLog, 'Acc-Top1', 5, [*valLog, *testLog])
    return

def evaluate(option: Options, net: nn.Module, dataLoader: DataLoader, loss_f: ScoreLossPlus):
    # Evaluate
    epoch_loss = 0.0
    epoch_score_dis = 0.0
    epoch_score = 0.0
    net.eval()
    correct_1 = 0
    correct_3 = 0
    correct_5 = 0
    # total = 0
    
    with torch.no_grad():
        for batch, data in enumerate(dataLoader, start=0):  
            images = data['images'].to(device=option.device, dtype=option.dtype)
            labels = data['labels'].to(device=option.device)

            logits = net(images)
             #  score loss added
            logits = logits.view(-1, loss_f.scoreLevel, opt.classes)
            logits = nn.Softmax(dim=1)(logits)

            loss, score_dis, score = loss_f(logits, labels)

            batch_loss = loss.item()*len(labels)
            epoch_loss += batch_loss
            batch_score_dis = score_dis.item()*len(labels)
            batch_score = score.item()*len(labels)
            epoch_score_dis += batch_score_dis
            epoch_score += batch_score
            # total += len(labels)

            # the S-Softmax Classifer
            prob = predict(logits, labels, loss_f)

            correct_1 += prob[0].cpu()
            correct_3 += prob[1].cpu()
            correct_5 += prob[2].cpu()
    return [correct_1/len(dataLoader.dataset), correct_3/len(dataLoader.dataset), correct_5/len(dataLoader.dataset), epoch_loss / len(dataLoader.dataset), epoch_score_dis/len(dataLoader.dataset), epoch_score/len(dataLoader.dataset)]

def test(option: Options, net: nn.Module, dataLoader: DataLoader, loss_f: ScoreLossPlus):
    if option.classes is None:
        printPlus("Error: option.classes is None", frontColor=31)
        sys.exit()
    confusion_matrix = np.zeros((option.classes, option.classes), dtype=np.int_)
    # Test
    net.eval()
    correct_1 = 0
    correct_3 = 0
    correct_5 = 0
    with torch.no_grad():
        for batch, data in enumerate(dataLoader, start=0):  
            images = data['images'].to(device=option.device, dtype=option.dtype)
            labels = data['labels'].to(device=option.device)

            logits = net(images)
             #  score loss added
            logits = logits.view(-1, loss_f.scoreLevel, opt.classes)
            logits = nn.Softmax(dim=1)(logits)

            prob = predict(logits, labels, loss_f)
            correct_1 += prob[0].cpu()
            correct_3 += prob[1].cpu()
            correct_5 += prob[2].cpu()
            logits_class = prob[3].squeeze(0)
            truth_class = labels
            # total += len(labels)
            # raws are real, cols are predict 
            for i in range(0, truth_class.shape[0]):
                confusion_matrix[truth_class[i].item(), logits_class[i].item()] += 1
    assert np.sum(confusion_matrix) == len(dataLoader.dataset), "评估结果不正确，混淆矩阵总数与被评估数据集不一致"
    metric = Metric(confusion_matrix)
    return metric, [correct_1/len(dataLoader.dataset), correct_3/len(dataLoader.dataset), correct_5/len(dataLoader.dataset)]

def saveModel(option: Options, net:nn.Module, savePthFile):
    if option.mul_gpu_train:
        torch.save(net.module.state_dict(), savePthFile)
    else:
        torch.save(net.state_dict(), savePthFile)

# 获得Top-1. Top-3, Top5 Acc
def predict(logits:torch.Tensor, labels:torch.Tensor, loss_f:ScoreLossPlus):
    # print(logits[0])
    
    # print(labels[0])
    
    # print(loss_f.score_gt)
    logits = logits * loss_f.score
    # print(logits[0])
    logits = torch.sum(logits, dim=1)
    # print(logits[0])
    # 按得分数排序：
    
    _, pred = logits.topk(5, 1, largest=True, sorted=True)

    labels = labels.view(labels.size(0), -1).expand_as(pred)
    correct = pred.eq(labels).float()

    #compute top 5
    correct_5 = correct[:, :5].sum()
    #compute top 3
    correct_3 = correct[:, :3].sum()
    #compute top1
    correct_1 = correct[:, :1].sum()
    return [correct_1, correct_3, correct_5, pred[:, :1]]

def write_kernel(write:SummaryWriter, model:nn.Module):
    for index, module in enumerate(model.modules()):
        print(module)

        if isinstance(module, nn.Conv2d):
            kernels = module.weight
            print(tuple(kernels.shape))
            kernel_nums, kernel_channels, kernel_width, kernel_height = tuple(kernels.shape)

            # 单独可视化每个kenel
            for kernel_id in range(kernel_nums):
                kernel = kernels[kernel_id, :, :, :].unsqueeze(1)
                kernel_grid = torchvision.utils.make_grid(kernel, normalize=True, scale_each=True, nrow=kernel_channels)
                write.add_image(f'{index}_Conv2Dlayer_split_in_channel', kernel_grid, global_step=kernel_id)

            # 可视化当前层所有的kernel
            # kernel_all = kernels.view(-1, kernel_channels, kernel_height, kernel_width)
            # kernel_grid = torchvision.utils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)
            # write.add_image(f'{index}_all', kernel_grid, global_step=322)

            break

def static_metirc(savePath, key, topk, logs):
    for log in logs:
        try:
            result = statics_metrics(log, key, topk)
            _info = f'{osp.basename(log)}---{key}:{result}'
            printPlus(_info, _file=savePath)
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    # 工程项目准备
    parser = argparse.ArgumentParser(description='project')
    parser.add_argument('--dataset', default='None', type=str, help ='train dataset')
    parser.add_argument('--level', action='append', help ='train dataset')      
    args = parser.parse_args()
  
    project = ProjectInfo()
    random.seed(project.Seed)
    torch.manual_seed(project.Seed)
    torch.cuda.manual_seed(project.Seed)
    torch.cuda.manual_seed_all(project.Seed)
    np.random.seed(project.Seed)
    # 训练配置准备
    opt = Options(config)
    if args.dataset != 'None':
        from datasetDriver100 import *
        opt.dataset = eval(args.dataset)
    # 数据集准备
    prepared_data = DatasetPrepare(project, option=opt)
    prepared_data.loadDataset()
    
    if args.level==None:
        args.level = ['5'] 
        
    level_list = []
    for item in args.level:
        level_list.append(int(item))
    
      
    for model in opt.model_group:
        opt.model = model
        for _level in level_list:
            train(project, opt, prepared_data, _level)
    print('Over')
    sys.exit()