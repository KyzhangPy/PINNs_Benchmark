###
@author: Zhang Shuo
###

### 调入sys库
import sys
# 临时添加本地库（添加import库的搜索路径），在列表的任意位置添加目录，新添加的目录会优先于其它目录被import检查
sys.path.insert(0, '../Utilities/')## 此处需手动输入位置路径

### 调库
import torch
# 调用Pytorch
from collections import OrderedDict
# 调用OrderedDict，OrderedDict是Python标准库中的一个数据结构，是一个有序的字典，可以记住元素的插入顺序，可以按照元素插入的顺序来迭代字典中的键值对
import numpy as np
# 导入numpy-科学计算工具库
import matplotlib.pyplot as plt
# 导入pyplot-画图功能库
import scipy.io
# 导入scipy.io-物理常量/单位库、常用的输入输出函数库
from scipy.interpolate import griddata
# 导入griddata-差值函数库（非规则网格的数据差值）
from plotting import newfig, savefig
# 导入plotting-可以结合各种视觉元素和工具创建可视化图形的库
from mpl_toolkits.axes_grid1 import make_axes_locatable
# 导入axes_grid1-辅助类几何工具，可以用于显示多个图像
import matplotlib.gridspec as gridspec
# 导入gridspec-专门指定画布中的子图位置的模块

### 导入warnings模块-用于处理警告消息，使程序在运行时可捕捉并处理一些非致命性问题，不中断程序的执行，通常用于提醒开发者一些潜在问题或不推荐用法，但不阻止程序继续执行
import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore')-运行时忽略掉所有警告
# warnings.filterwarnings("ignore", category=DeprecationWarning)-忽略特定类型的警告
# warnings.filterwarnings("ignore", message="some specific warning message")-忽略特定消息的警告

### 设置随机数种子
np.random.seed(1234)

### 设置计算单元CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### 定义类
class DNN(torch.nn.Module):
# 定义的神经网络模型类DNN，继承自nn.Module
# nn.Module是PyTorch中所有神经网络模型的基类
    def __init__(self, layers):
    # 构造一个初始化函数，用于初始化类的实例，layers是该函数的参数
    # self指的是实例Instance本身，python类中规定：函数的第一个参数是实例对象本身，并且约定俗成，其名字写为self，不能省略
        super(DNN, self).__init__()
        # 类DNN把类nn.Module的__init__()放到自己的__init__()当中，类DNN就有了类nn.Module的__init__()的那些东西
        # 对继承自父类nn.Module的属性进行初始化，而且是用nn.Module的初始化方法来初始化继承的属性

        self.depth = len(layers) - 1
        # 定义参数：depth（DNN层数）

        self.activation = torch.nn.Tanh
        # 定义参数：activation（激活函数）
        
        layer_list = list()
        # 为序列中的每个元素都分配一个数字(它的位置index)
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            



    
