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

### 定义神经网络模型的类DNN，继承自nn.Module
class DNN(torch.nn.Module):
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
        # 创建表达layers结构的空张量
        for i in range(self.depth - 1):
            layer_list.append(  ( 'layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]) )  )
            # nn.Linear(in_feature,out_feature,bias)表示线性变换
            # in_feature表示输入Tensor最后一维的通道数，int型，out_feature表示输出Tensor最后一维的通道数，int型，bias表示是否添加bias偏置，bool型
            # 表示在layer_list的第i列插入上述线性变换
            layer_list.append(  ( 'activation_%d' % i, self.activation() )  )
            # 表示在layer_list的
        layer_list.append(  ( 'layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]) )  )
        layerDict = OrderedDict(layer_list)
        # OrderedDict()是一个特殊的字典子类，保持了字典中元素被插入时的顺序，当你遍历一个OrderedDict时，元素会按照它们被插入的顺序出现，而不是按照它们的键的排序顺序

        self.layers = torch.nn.Sequential(layerDict)
        # nn.Sequential是一个序列容器，可以看成是有多个函数运算对象，串联成的神经网络，其返回的是Module类型的神经网络对象
        # 与一层一层的单独调用模块组成序列相比，nn.Sequential() 可以允许将整个容器视为单个模块（即相当于把多个模块封装成一个模块）
        # nn.Sequential()按照内部模块的顺序自动依次计算并输出结果，这就意味着我们可以利用nn.Sequential() 自定义自己的网络层
    
    def forward(self, x):
        out = self.layers(x)
        return out

### 定义PINN神经网络的类
class PhysicsInformedNN():
     def __init__(self, X, u, layers, lb, ub):

         # 边界条件
         self.lb = torch.tensor(lb).float().to(device)
         self.ub = torch.tensor(ub).float().to(device)
         
         # 数据
         self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
         self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
         self.u = torch.tensor(u).float().to(device)

         # 设置
         self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
         self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(device)
         
         self.lambda_1 = torch.nn.Parameter(self.lambda_1)
         self.lambda_2 = torch.nn.Parameter(self.lambda_2)

         # Deep Neural Network
         self.dnn = DNN(layers).to(device)
         self.dnn.register_parameter('lambda_1', self.lambda_1)
         self.dnn.register_parameter('lambda_2', self.lambda_2)
         
         # 优化
         self.optimizer = torch.optim.LBFGS(
             self.dnn.parameters(),
             lr=1.0,
             max_iter=50000,
             max_eval=50000,
             history_size=50,
             tolerance_grad=1e-5,
             tolerance_change=1.0 * np.finfo(float).eps,
             line_search_fn="strong_wolfe"
         )
         
         self.optimizer_Adam = torch.optim.Adam( self.dnn.parameters() )
         self.iter = 0
         
     def net_u(self, x, t): 
         u = self.dnn( torch.cat([x, t], dim=1) )
         return u
     def net_f(self, x, t):
         """ The pytorch autograd version of calculating residual """
         lambda_1 = self.lambda_1
         lambda_2 = torch.exp(self.lambda_2)
         u = self.net_u(x, t)
         
         u_t = torch.autograd.grad(
             u, t,
             grad_outputs=torch.ones_like(u),
             retain_graph=True,
             create_graph=True
         )[0]
         u_x = torch.autograd.grad(
             u, x,
             grad_outputs=torch.ones_like(u),
             retain_graph=True,
             create_graph=True
         )[0]
         u_xx = torch.autograd.grad(
             u_x, x,
             grad_outputs=torch.ones_like(u_x),
             retain_graph=True,
             create_graph=True
         )[0]
         
         f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
         return f    
     
     def loss_func(self):
         u_pred = self.net_u(self.x, self.t)
         f_pred = self.net_f(self.x, self.t)
         loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)
         self.optimizer.zero_grad()
         loss.backward()
         
         self.iter += 1






    
