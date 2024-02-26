"""
@author: Zhang Shuo
"""

# 导入sys库
import sys 
# 临时添加本地库（添加import库的搜索路径），在列表的任意位置添加目录，新添加的目录会优先于其它目录被import检查
sys.path.insert(0, '../../main/')  ## 此处需手动输入位置路径

# 调库
import tensorflow as tf 
## 调用tensorflow开发工具
import numpy as np 
## 调用numpy科学计算工具 
import matplotlib.pyplot as plt 
## 调用python的画图功能库
import scipy.io 
## 调用物理常量/单位库scipy、常用的输入输出函数
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
## 随机种子应为整数,手动输入

# 定义类（类是一种数据类型，代表着一类具有相同属性和方法的对象的集合）
class PhysicsInformedNN:
  
  #定义初始化类的函数
  def __init__(self, x, y, t, u, v, layers):
    
    X = np.concatenate([x, y, t], 1)
    ## 数组的组合拼接，1表示按列拼接，X表示自变量的矩阵
    
    self.lb = X.min(0)
    self.ub = X.max(0)
    ## 0表示返回矩阵中每一列的最小值，1表示返回每一行的最小值
    ## self.1b即为[x,y,t]的最小值
    ## self.ub即为[x,y,t]的最大值
    
    self.X = X
    ## 保存类的自变量矩阵X
    
    self.x = X[:,0:1]
    self.y = X[:,1:2]
    self.t = X[:,2:3]
    ## 分别保存类的自变量 X[：,0:1]取所有数据的第m到n-1列数据，即含左不含右，分别保存类的向量x,y,t
    
    self.u = u
    self.v = v
    ## 分别保存类的因变量u,v
    
    self.layers = layers
    ## 保存类的NN层数向量layers
    
    # 定义initialize_NN函数
    self.weights, self.biases = self.initialize_NN(layers)
    ## 根据layers变量可导出weights和biases变量
    
    self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
    self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
    ## 初始化两个参数lambda1和lambda2，0.0表示定义对象的初值，dtype表示定义对象的数据类型
    ## 此处lambda1和lambda2分别用来表示NS方程中的两个未知参数
    
    # 定义session指令
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    ## tf.Session用来创建一个新的tensorflow会话，用于执行真正的计算
    ## tensorflow的计算图只是描述了计算执行的过程，没有真正执行计算，真正的计算过程是在tensorflow的会话中进行的
    ## Session提供了求解张量，执行操作的运行环境，将计算图转化为不同设备上的执行步骤。包括创建会话（tf.Session）、执行会话（sess.run）、关闭会话（sess.close）
    ## tf.ConfigProto作用是配置tf.Session的运算方式，比如GPU运算或CPU运算
    ## allow_soft_placemente表示当运行设备不满足时，是否自动分配GPU或CPU
    ## log_device_placement表示是否打印设备分配日志

    # 定义占位符
    self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
    self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
    self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])    
    
    self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
    self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
    ## tf.placeholder是常用的处理输入数据的工具，允许在定义计算图时创建占位符节点，tf.placeholder(dtype,shape=none,name=none)
    ## dtype指定占位符的数据类型，如tf.float32,tf.int32
    ## shape指定占位符的形状，如不指定，可接受任意形状的输入数据
    ## name是给占位符名称指定一个可选的名称
    ## shape[0]表示矩阵的行数，shape[1]表示矩阵的列数
    ## shape=[None, self.x.shape[1]]表示接受任意行、1列的输入数据
    
    # 构建神经网络的输入输出关系
    self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)
    ## 定义net_NS函数
    ## net_NS函数根据输入的(x,y,t)计算输出(u,v,p)和NS方程的值(f_u_pred,f_v_pred)

    # 定义损失函数的计算方法
    self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                tf.reduce_sum(tf.square(self.f_u_pred)) + \
                tf.reduce_sum(tf.square(self.f_v_pred))
    ## tf.reduce_sum表示对矩阵中所有元素进行求和，并将结果返回至一维
    ## tf.square表示对矩阵中的每个元素求平方

    ## 定义优化方法
    self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                            method = 'L-BFGS-B', 
                                                            options = {'maxiter': 50000,
                                                                       'maxfun': 50000,
                                                                       'maxcor': 50,
                                                                       'maxls': 50,
                                                                       'ftol' : 1.0 * np.finfo(float).eps})
    ## tf.contrib.opt.ScipyOptimizerInterface是tensorflow的模块，提供将Scipy优化器与tensorflow集成的接口，可使用Scipy中的优化算法来优化Tensorflow模型中的变量
    ## L-BFGS-B表示优化方法
    ## maxiter定义最大迭代次数，int
    ## maxfun定义函数计算的最大数量，int
    ## maxcor定义有限内存矩阵的最大可变度量校正数（有限内存BFGS方法不存储完整的hessian，而是使用多项校正数来近似），int
    ## maxls定义最大的线性搜索步数，int，默认值20
    ## ftol表示当f^k-f^[(k+1)/max[f^k,f^(k+1),1]]小于ftol值时，迭代停止
    ## np.finfo用于生成一定格式，且数值较小的偏置项eps，以避免分母或对数变量为0
    
    # 定义优化函数
    self.optimizer_Adam = tf.train.AdamOptimizer()
    self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 
    ## tf.train.AdamOptimizer表示adam优化算法，是一个选取全局最优点的优化算法，引入了二次方梯度修正，来最小化损失函数
    ## 通过adam算法最小化loss来进行优化
    
    # 初始化模型的参数，sess.run
    init = tf.global_variables_initializer()
    self.sess.run(init)

    # 定义initialize_NN（用来根据layers变量来初始化神经网络中权重weights、偏移biases参数值的函数）
    def initialize_NN(self, layers):  
      weights = []
      biases = []
      num_layers = len(layers)  
      ## num_layers为层向量的长度，即为神经元的层数
      for l in range(0,num_layers-1):
        W = self.xavier_init(size=[layers[l], layers[l+1]]) 
        ## xavier_init()是随机初始化参数的分布范围，此处是初始化每两层之间的权重参数W，是一个l层（m个输入）到l+1层（n个输出）的m*n矩阵
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)  
        ##  tf.zeros()表示生成全为0的tensor张量，此处是初始化每层的偏移参数b,从第l+1层开始（n个输出）是一个1*n的向量，初始值为0
        weights.append(W)
        biases.append(b)  
        ## append表示在变量末尾增加元素，此处即把每次循环的w，b都存进空矩阵weight，biases中，即weight,biases为罗列出所有层之间的权重和偏移参数的矩阵
      return weights, biases
    
    # 定义xavier_init，用来控制初始化参数的分布范围
    ## xavier_init函数在initialize_NN函数中有使用
    ## 该初始化方法由Bengio等人提出，为了保证前向传播和反向传播时每一层的方差一致，根据每层的输入输出个数来决定参数随机初始化的分布范围
    def xavier_init(self, size):
      in_dim = size[0]  
      ## in_dim表示输入层的参数个数，即上述的m
      out_dim = size[1]  
      ## out_dim表示输出层的参数个数，即上述的n       
      xavier_stddev = np.sqrt(2/(in_dim + out_dim))  
      ## np.sqrt()计算数组中各元素的平方根，某一层的权重W的方差即为sqrt(2/(该层的输入个数+该层的输出个数))，Bengio论文中有相关推导
      return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
      ## tf.truncated_normal表示截断地产生正态分布的函数（平均值、标准差可设定），产生的值如果与均值之差大于2倍标准差则重新选择

    # 定义neural_net，用于构建神经网络的计算关系
    ## 根据weights和biases矩阵，从输入层算到输出层的计算方法
    def neural_net(self, X, weights, biases):
      num_layers = len(weights) + 1  
      ## NN的总层数
      H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  
      ## self.1b表示自变量矩阵[x,y,t]的各列最小值向量，self.ub表示自变量矩阵[x,y,t]的各列最大值向量
      ## 对初始的自变量矩阵进行归一化处理
      ## 2*归一化结果再-1，即把输入X处理成关于0对称
      for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))  
        ## tf.matmul表示矩阵相乘，tf.add表示矩阵相加
        ## 用循环的方式构建神经网络的计算关系，逐层计算
        ## 激活函数是tanh，其值在-1到1的范围内
      W = weights[-1]  ## 权重参数矩阵weights的最后一行，即最后两层之间的权重参数
      b = biases[-1]  ## 偏移参数矩阵biases的最后一行，即最后两层之间的偏移参数
      Y = tf.add(tf.matmul(H, W), b) ##  输出矩阵Y，即输出层的结果
      return Y
    
    # 定义NS方程
    def net_NS(self, x, y, t):
      lambda_1 = self.lambda_1
      lambda_2 = self.lambda_2 
      ## 初始化的两个变量，用于表示NS方程中的未知参数
      psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
      ## tf.concat([tensor1,tensor2,tensor3,...],axis)用于拼接张量（tensor），
      ## axis表示拼接的维度，axis=0,1,2,...，0表示在第0个维度拼接，1表示在第1个维度拼接，第0个维度为最外层方括号下的子集，第1个维度为倒数第二层方括号下的子集
      ## 此处，输出数据的个数是2个，即p和psi
      ## 采用neural_net函数，根据输入层的[x,y,t]和[weights,biases]计算到输出层psi_and_p
      psi = psi_and_p[:,0:1]  ##第1维中取所有数据，第2维中取第0个数据，此处指位移值
      p = psi_and_p[:,1:2]  ##第1维中取所有数据，第2维中取第1个数据，此处指压力值
      
      u = tf.gradients(psi, y)[0]  ## 位移对y方向的偏导为u？？？
      v = -tf.gradients(psi, x)[0]  ## 位移对x方向的偏导为v？？？

      ## u对x,y,t的二阶偏导
      u_t = tf.gradients(u, t)[0]
      u_x = tf.gradients(u, x)[0]
      u_y = tf.gradients(u, y)[0]
      u_xx = tf.gradients(u_x, x)[0]
      u_yy = tf.gradients(u_y, y)[0]

      ## v对x,y,t的二阶偏导
      v_t = tf.gradients(v, t)[0]
      v_x = tf.gradients(v, x)[0]
      v_y = tf.gradients(v, y)[0]
      v_xx = tf.gradients(v_x, x)[0]
      v_yy = tf.gradients(v_y, y)[0]
      
      p_x = tf.gradients(p, x)[0]  ## 压力对x的偏导
      p_y = tf.gradients(p, y)[0]  ## 压力对y的偏导
      
      f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy)  ## x方向上的动量守恒方程
      f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)  ## y方向上的动量守恒方程
      
      return u, v, p, f_u, f_v
    
    # 输出当前的Loss和lambda值
    def callback(self, loss, lambda_1, lambda_2):
      print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))
    ## %.3e表示科学计数法，保留三位小数
    ## %.3f表示常规计数法，保留三位小数
    ## %.5f表示常规计数法，保留五位小数
    
    # 定义训练函数train
    def train(self, nIter): 
      
      tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                 self.u_tf: self.u, self.v_tf: self.v}
      ## 定义字典，创建键值对，键值对是一种数据结构，由键和与之关联的值组成
      ## 形式的意义为tf_dict = {变量1：值1，变量2：值2，变量3：值3}
      
      start_time = time.time()
      ## time.time()表示当前时间的时间戳，即1970年1月1日00:00:00到当前时间的秒数的浮点数
      
      for it in range(nIter):
        self.sess.run(self.train_op_Adam, tf_dict)
        ## 运行session
        ## tensorflow的赋值操作只是确定一个空壳子，需要使用sess.run来让数据流动起来
        
        if it % 10 == 0:
        # it % 10 == 0 表示取余，即每十步记录一次
          elapsed = time.time() - start_time
          loss_value = self.sess.run(self.loss, tf_dict)
          lambda_1_value = self.sess.run(self.lambda_1)
          lambda_2_value = self.sess.run(self.lambda_2)
          print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
          ## 输出迭代步数值、Loss值、lambda_1的值、lambda_2的值
          ## %d表示十进制整数；%.3e表示科学计数法，保留三位小数；%.3f表示常规计数法，保留三位小数；%.5f表示常规计数法，保留五位小数；%.2f表示常规计数法，保留两位小数；
          start_time = time.time()

      self.optimizer.minimize(self.sess,
                              feed_dict = tf_dict,
                              fetches = [self.loss, self.lambda_1, self.lambda_2],
                              loss_callback = self.callback)
      ## feed_dict是给使用spaceholder创建出来的tensor赋值
      ## fetches表示获取操作op所对应的结果

    # 定义预测函数predict
    def predict(self, x_star, y_star, t_star):
      
      tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
      ## 创建键值对，放置x_star，y_star，t_star
      
      u_star = self.sess.run(self.u_pred, tf_dict)
      v_star = self.sess.run(self.v_pred, tf_dict)
      p_star = self.sess.run(self.p_pred, tf_dict)
      
      return u_star, v_star, p_star

# 定义绘图求解域的函数、以及绘图
def plot_solution(X_star, u_star, index):

  lb = X_star.min(0)
  ## x_star数据中列的最小值
  ub = X_star.max(0)
  ## x_star数据中列的最大值
  nn = 200
  ## 数据分隔值总数
  x = np.linspace(lb[0], ub[0], nn)
  y = np.linspace(lb[1], ub[1], nn)
  ## linspace是通过定义均匀间隔创建数值序列，指定间隔的起止点、终止点以及分隔值总数，返回类均匀分布的数值序列
  X, Y = np.meshgrid(x,y)
  ## np.meshgrid是一个在给定多维网格状情况下生成网格点坐标的函数，它将向量生成为矩阵，并返回多个坐标矩阵的列表

  U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
  ## 二维插值函数griddata，
  
  plt.figure(index)
  plt.pcolor(X,Y,U_star, cmap = 'jet')
  plt.colorbar()
      

      




       







      
     


    








  









