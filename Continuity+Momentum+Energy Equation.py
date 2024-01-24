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
## 调用griddata差值函数（非规则网格的数据差值）
import time 
## 调用时间日期函数
from itertools import product, combinations 
## itertools是用于迭代工具的标准库，itertools.product计算多个可迭代对象的笛卡尔积，itertools.combinations生成可迭代对象的所有长度为r的组合
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
## mpl_toolkits是matplotlib的绘图工具包，mpl_toolkits.mplot3d用于绘制和可视化三维图形
from plotting import newfig, savefig
## plotting可以结合各种视觉元素和工具创建可视化图形的库
from mpl_toolkits.axes_grid1 import make_axes_locatable
## axes_grid1是辅助类几何工具，可以用于显示多个图像
import matplotlib.gridspec as gridspec
## gridspec是专门指定画布中的子图位置的模块

# 设置随机数种子
np.random.seed(1234)
tf.set_random_seed(1234)

# 定义PINN的类（类是一种数据类型，代表着一类具有相同属性和方法的对象的集合）
class PhysicsInformedNN:
  #初始化类（定义类中的变量）
  def __init__(self, x, y, t, u, v, layers):
    
    ## 数组的组合拼接，1表示按列拼接，X表示自变量的矩阵
    X = np.concatenate([x, y, t], 1)
    
    ## 0表示返回矩阵中每一列的最小值，1表示返回每一行的最小值，即表示
    self.lb = X.min(0)
    self.ub = X.max(0)
    ## 保存类的自变量矩阵
    self.X = X
    ## 分别保存类的自变量 X[：,0:1]取所有数据的第m到n-1列数据，即含左不含右
    self.x = X[:,0:1]
    self.y = X[:,1:2]
    self.t = X[:,2:3]
    ## 分别保存类的因变量
    self.u = u
    self.v = v
    ## 保存类的NN层数
    self.layers = layers
   
    ## 初始化神经网络
    self.weights, self.biases = self.initialize_NN(layers)
    ## 初始化参数，定义变量类型，0.0表示定义变量初值，dtype表示创建一个数据类型对象
    self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
    self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
    ## tf.Session用来创建一个新的tensorflow会话
    ## tensorflow的计算图只是描述了计算执行的过程，没有真正执行计算，真正的计算过程是在tensorflow的会话中进行的
    ## Session提供了求解张量，执行操作的运行环境，将计算图转化为不同设备上的执行步骤。包括创建会话（tf.Session）、执行会话（sess.run）、关闭会话（sess.close）
    ## tf.ConfigProto作用是配置tf.Session的运算方式，比如GPU运算或CPU运算
    ##
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    ## 
    self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])








  









