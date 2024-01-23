"""
@author: Zhang Shuo
"""

# 导入sys库
import sys 
# 临时添加本地库（添加import库的搜索路径），在列表的任意位置添加目录，新添加的目录会优先于其它目录被import检查
sys.path.insert(0, '../../main/') 

# 调库
import tensorflow as tf 
## 调用tensorflow开发工具
import numpy as np 
## 调用科学计算工具 
import matplotlib.pyplot as plt 
## 调用python的画图功能库
import scipy.io 
## 调用物理常量/单位库、常用的输入输出函数
from scipy.interpolate import griddata 
# 调用griddata差值函数（非规则网格的数据差值）
import time 
## 调用时间日期函数
from itertools import product, combinations 
## itertools是用于迭代工具的标准库，itertools.product计算多个可迭代对象的笛卡尔积，itertools.combinations生成可迭代对象的所有长度为r的组合
from mpl_toolkits.mplot3d import Axes3D
## mpl_toolkits.mplot3d用于绘制和可视化三维图形，Axes3D是绘制3D图形的库
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
## 
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
