# -*- encoding: utf-8 -*-
'''
@NAME      :Aerosol.py
@TIME      :2021/03/25 11:10:28
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
'''
# Inbuild package list

import math

# Site-package list
import scipy
from scipy import integrate
import numpy as np
import miepython


# Userdefined package list


# CODING CONTENT

# 气溶胶光学厚度
def AOD(beta=[], height=[]):
    '''
    气溶胶光学厚度:介质的消光系数在垂直高度上的积分,描述气溶胶对光的削减作用
    '''
    AOD = []
    for idx in range(len(beta)):
        inter_s, err = integrate.quad(beta[idx], height[0], height[-1])
        AOD.append(inters)
    return pd.DataFrame(AOD)


# 气溶胶质量浓度

# STEP 1:消光效率的计算
def Q_ext(mj, r_min, r_max, r_step):
    '''
    m: 负折射率,虚数
    r ：直径,微米[um]
    * m,r可以为数组,但长度必须相同
    numbda ：波长,微米[um]
    '''
    r = np.arange(r_min, r_max, r_step)
    num_r = r.shape[0]  # 获取x的长度

    m = np.empty(num_r, dtype=complex)  # 创建空数组
    m.fill(mj)  # 为空数组填充复折射率

    Q_ext, Q_sca, Q_back, g = miepython.mie(m, r)
    print(Q_ext)


Q_ext(mj=1.5 - 1j, r_min=0.05, r_max=10.05, r_step=0.05)


# STEP 2:气溶胶粒子谱的反演: 消光法遥感气溶胶谱反演方法
# 求解第一类Fredholm 积分方程：https://www.guangshi.io/posts/fredholm-equation/
# core algorithm of non-negative Tikhonov regularization with equality constraint (NNETR)
def NNETR(K, f, Delta, epsilon, alpha):
    # the first step
    A_nn = np.vstack((K, alpha * np.identity(K.shape[1])))
    b_nn = np.hstack((f, np.zeros(K.shape[1])))

    # the second step
    A_nne = np.vstack((epsilon * A_nn, np.full(A_nn.shape[1], 1.0)))
    b_nne = np.hstack((epsilon * b_nn, 1.0))

    # Use NNLS solver provided by scipy
    sol, residue = scipy.optimize.nnls(A_nne, b_nne)

    # solution should be divided by Delta (grid size)
    sol = sol / Delta
    return sol, residue


# STEP 3:颗粒物质量浓度的计算
def MEE():
    pi = math.pi
    r_max = 10  # [um]
    r_min = 0.01  # [um]
    AOD = []
    rou = 2  # [g/m]
    Q_ext = 2.33

# if __name__ == "__main__":
#     print('开始读取RCS数据')
#     RCS_PATH = r"./Results/Channels/RCS/Channel_Rcs1.csv"
#     Height_PATH = r"./Results/Channels/Height/Channel_Height1.csv"

#     rcs_channel = pd.read_csv(RCS_PATH,header=None,index_col=0)
#     h_channel = pd.read_csv(Height_PATH,header=None)

#     rcs = np.array(rcs_channel[1][:])
#     h = np.array(h_channel)

#     # print(h)
#     # print(rcs)

#     print('使用Kletthod方法反演雷达方程')
#     RM = Klett()

#     print('### 计算消光系数alpha')
#     alpha = RM.alpha_z(rcs,h)
#     print(alpha)