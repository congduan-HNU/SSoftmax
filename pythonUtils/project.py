# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/15/23 5:36 PM
# __filename:  project.py
import os.path as osp
import time
import sys

class ProjectInfo(object):
    def __init__(self, comment=''):
        self.ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
        print("Project Root Path: ", self.ROOT)
        self.IsDebug = True
        self.StartTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.comment = comment
        self.homePath = osp.expanduser('~')
        # sys.path.append(self.ROOT)
        self.Seed = 42


projectInfo = ProjectInfo()


