# _*_coding:utf-8_*_
# __author:    duancong
# __date:      4/24/23 4:18 PM
# __filename:  listPlus.py

def augList(origin_list:list, n:int):
        """
        列表切分, 任意长度列表近似等分，用于多线程
        :param origin_list:
        :param n:
        :return:
        """
        result = []
        # if len(origin_list) % n == 0:
        #     cnt = len(origin_list) // n
        # else:
        cnt = len(origin_list) // n #+ 1
        rest = len(origin_list) % n
        for i in range(0, n):
            result.append(origin_list[i * cnt:(i + 1) * cnt])
        for j in range(0, rest):
            result[j].append(origin_list[n*cnt+j])
        return result