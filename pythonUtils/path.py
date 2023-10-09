# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/15/23 5:51 PM
# __filename:  path.py
import os
import os.path as osp
from .print_plus import printPlus
def creat_folder(folder, info=True):
    if osp.exists(folder):
        if info:
            printPlus(f"Warning: The folder {folder} is existed.", 33)
        else:
            pass
    else:
        try:
            os.makedirs(folder)
            printPlus(f"The folder {folder} is be created.", 32)
        except OSError:
            printPlus(f"Can not Created the folder: {folder}.", 31)
            raise ""