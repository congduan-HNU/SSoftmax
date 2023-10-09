# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/16/23 9:42 AM
# __filename:  timePlus.py
import time
def timeClock():
    return time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
