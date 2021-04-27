# -*- encoding: utf-8 -*-
'''
@NAME      :Klett method.py
@TIME      :2021/03/25 11:10:28
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
'''
# Inbuild package list
from math import log, exp, sqrt

# Site-package list
from scipy import integrate
import numpy as np
import pandas as pd
# Userdefined package list


# CODING CONTENT
class Klett():
    def __init__(self):
        self.height = []
        self.alpha = []
        self.bata = []
        self.k = 0.67
        self.LR_z = 50

        self.bin_width = 7.5  # 单位:m
        self.data_points = 16221  # number_of_datapoints
        self.ref = 3

    def cal_z(self):
        # 计算 高度z
        self.height = [self.bin_width * bin_number +
                       (self.bin_width / 2.0) for bin_number in range(self.data_points)]
        # print(self.height[:800])
        return self.height

    def alpha_z(self, rcs=[]):
        '''
        根据Klett雷达反演方法求解激光雷达消光系数alpha_z
        ref = 3 # 该值为高度列表处于第4位的高度值
        k = [0.67 - 1]
        LR_z = [2 - 100] 
        '''
        for h_index in range(len(self.height)):
            h = self.height[h_index]  # 对应高度
            # print(h)

            def _delta_s(x):
                '''
                构造范围矫正信号(RCS)的自然对数值函数,
                对于每一个高度z都有一个特定的函数。
                '''
                s = log(x ** 2 * rcs[h_index])  # 范围矫正信号(RCS)的自然对数值
                s_ref = log(self.height[self.ref] **
                            2 * rcs[self.ref])  # 参考高度ref下的RCS自然对数值
                delta_s = exp((s-s_ref) / self.k)
                return delta_s

            var_exp = _delta_s(h)
            # print(var_exp)

            # 计算参考高度0处范围矫正信号的自然对数
            var_s_0 = log(self.height[0] ** 2 * rcs[0])
            # 计算参考高度z_ref处范围矫正信号的自然对数
            var_s_ref = log(self.height[self.ref] ** 2 * rcs[self.ref])

            # 根据Collis斜率法计算参考高度下的大气消光系数
            alpha_ref = 0.5 * (var_s_0 - var_s_ref) / \
                (self.height[self.ref] - self.height[0])

            # 求RCS构造函数在[h,h_ref]区间内的定积分
            inter_s, err = integrate.quad(_delta_s, h, self.height[self.ref])
            # print(inter_s) # 积分结果

            # 根据Klett方法计算大气消光系数
            var_alpha_z = var_exp / ((1/alpha_ref) - (2/self.k)*inter_s)

            # 保存计算结果
            self.alpha.append(var_alpha_z)
        return self.alpha
        # print(self.alpha)

    def beta_z(self):
        '''
        # 计算后向散射系数beta_z
        '''
        np_alpha = np.array(self.alpha)  # 列表转化为numpy数组
        posi_alpha = np.where(np_alpha > 0, np_alpha, 0)  # 将数组中小于0的元素置0
        self.bata = [self.LR_z * (var_alpha_z ** self.k)
                     for var_alpha_z in posi_alpha]  # 计算bata_z
        return self.bata
        # print(self.bata)


if __name__ == "__main__":
    print('开始读取RCS数据')
    RM_PATH = r"./Data/3.RCS/RM1912000021110.csv"
    rcs_file = pd.read_csv(RM_PATH)
    rcs_chnnel = rcs_file['00355.p.AU']

    print('使用Kletthod方法反演雷达方程')
    RM = Klett()
    height = RM.cal_z()
    print('### 计算消光系数alpha')
    alpha = RM.alpha_z(rcs_chnnel)
    print('### 计算后向散射系数beta')
    beta = RM.beta_z()
    results = []
    results.append(height)
    results.append(beta)
    results.append(alpha)

    header = ['height', 'alpha', 'beta']
    dataframe = pd.DataFrame(results).T
    dataframe.columns = header
    # print(dataframe.head(5))
    print('开始保存结果')
    dataframe.to_csv('./Results/Results_Coefficient.csv', sep=',')
    print('程序运行结束')
