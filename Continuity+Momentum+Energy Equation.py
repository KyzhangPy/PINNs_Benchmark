"""
@author: Zhang Shuo
"""

# 导入sys库
import sys 
# 临时添加本地库（添加库的搜索路径），在列表的任意位置添加目录，新添加的目录会优先于其它目录被import检查
sys.path.insert(0, '../../main/') 

# 调库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
