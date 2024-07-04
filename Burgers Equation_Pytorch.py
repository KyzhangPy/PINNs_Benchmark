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


### 神经网络模型的类DNN，继承自nn.Module
class DNN(torch.nn.Module):
# nn.Module是PyTorch中所有神经网络模型的基类
    def __init__(self, layers):
    # 构造一个初始化函数，用于初始化类的实例，layers是该函数的参数
    # self指的是实例Instance本身，python类中规定：函数的第一个参数是实例对象本身，并且约定俗成，其名字写为self，不能省略
        super(DNN, self).__init__()
        # 类DNN把类nn.Module的__init__()放到自己的__init__()当中，类DNN就有了类nn.Module的__init__()的那些东西
        # 对继承自父类nn.Module的属性进行初始化，而且是用nn.Module的初始化方法来初始化继承的属性

        self.depth = len(layers) - 1
        # 定义depth（DNN层数）

        self.activation = torch.nn.Tanh
        # 定义activation（激活函数）
        
        layer_list = list()
        # 创建表达layers结构的空张量
        for i in range(self.depth - 1):
            layer_list.append(  ( 'layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]) )  )
            # nn.Linear(in_feature,out_feature,bias)表示线性变换
            # in_feature表示输入神经元的个数，int型，out_feature表示输出神经元的个数，int型，bias表示是否添加偏置，bool型
            layer_list.append(  ( 'activation_%d' % i, self.activation() )  )
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
         if self.iter % 100 == 0:
             print(
                 'Loss: %e, l1: %.5f, l2: %.5f' %
                 (
                     loss.item(),
                     self.lambda_1.item(),
                     torch.exp(self.lambda_2.detach()).item()
                 )
             )
         return loss
     
     def train(self, nIter):
         self.dnn.train()
         for epoch in range(nIter):
             u_pred = self.net_u(self.x, self.t)
             f_pred = self.net_f(self.x, self.t)
             loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)
             
             # 反向传播与优化
             self.optimizer_Adam.zero_grad()
             loss.backward()
             self.optimizer_Adam.step()
             
             if epoch % 100 == 0:
                 print(
                     'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' % 
                     (
                         epoch,
                         loss.item(),
                         self.lambda_1.item(),
                         torch.exp(self.lambda_2).item()
                     )
                 )
         
         # 反向传播与优化
         self.optimizer.step(self.loss_func)

     def predict(self, X):
         x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
         t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
         
         self.dnn.eval()
         u = self.net_u(x, t)
         f = self.net_f(x, t)
         u = u.detach().cpu().numpy()
         f = f.detach().cpu().numpy()
         return u, f


### 配置
nu = 0.01/np.pi
N_u = 2000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('data/burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0) 


### 无噪声数据训练
%%time

noise = 0.0            

# 创建训练集
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx,:]
u_train = u_star[idx,:]

# 训练
model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
model.train(0)


### 误差估计
u_pred, f_pred = model.predict(X_star)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

lambda_1_value = model.lambda_1.detach().cpu().numpy()
lambda_2_value = model.lambda_2.detach().cpu().numpy()
lambda_2_value = np.exp(lambda_2_value)

error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

print('Error u: %e' % (error_u))    
print('Error l1: %.5f%%' % (error_lambda_1))                             
print('Error l2: %.5f%%' % (error_lambda_2))  


### 带噪声数据训练
noise = 0.01    

# 创建训练集
u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

# 训练
model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
model.train(10000)


### 可视化
""" The aesthetic setting has changed. """
####### Row 0: u(t,x) ######

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15) 

ax.plot(
    X_u_train[:,1], 
    X_u_train[:,0], 
    'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=.5
)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.9, -0.05), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.show()

####### Row 1: u(t,x) slices ######

""" The aesthetic setting has changed. """

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = 0.25$', fontsize = 15)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.50$', fontsize = 15)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])    
ax.set_title('$t = 0.75$', fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()


### evaluations 误差估计
u_pred, f_pred = model.predict(X_star)

lambda_1_value_noisy = model.lambda_1.detach().cpu().numpy()
lambda_2_value_noisy = model.lambda_2.detach().cpu().numpy()
lambda_2_value_noisy = np.exp(lambda_2_value_noisy)

error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

print('Error u: %e' % (error_u))    
print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
print('Error l2: %.5f%%' % (error_lambda_2_noisy)) 

####### Row 3: Identified PDE ######

fig = plt.figure(figsize=(14, 10))

gs2 = gridspec.GridSpec(1, 3)
gs2.update(top=0.25, bottom=0, left=0.0, right=1.0, wspace=0.0)

ax = plt.subplot(gs2[:, :])
ax.axis('off')

s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
s3 = r'Identified PDE (1\% noise) & '
s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
s5 = r'\end{tabular}$'
s = s1+s2+s3+s4+s5
ax.text(0.1, 0.1, s, size=25)

plt.show()
