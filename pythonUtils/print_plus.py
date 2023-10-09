# _*_coding:utf-8_*_
# __author:    duancong
# __date:      2/15/23 5:15 PM
# __filename:  printPlus.py

def printPlus(content, frontColor=37, backColor=40, type=0, _file=None):
    """
    +-----+-------------+------------+------+---+
    |     | front color | back color | 显示方式 |   |
    +-----+-------------+------------+------+---+
    | 黑色  | 30          | 40         | 默认   | 0 |
    | 红色  | 31          | 41         | 高亮   | 1 |
    | 绿色  | 32          | 42         | 下划线  | 2 |
    | 黄色  | 33          | 43         | 闪烁   | 3 |
    | 蓝色  | 34          | 44         | 反白   | 4 |
    | 紫红色 | 35          | 45         | 不可见  | 5 |
    | 青蓝色 | 36          | 46         |      |   |
    | 白色  | 37          | 47         |      |   |
    +-----+-------------+------------+------+---+
    """
    if _file != None:
        with open(_file, "a", encoding='UTF-8') as f:
            f.write(f"{content}\n")
    return print(f"\033[0;{str(frontColor)};{str(backColor)}m{content}\033[{str(type)}m")
