# -*- encoding: utf-8 -*-
'''
@NAME      :Klett method.py
@TIME      :2021/03/25 11:10:28
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
'''
# Inbuild package list
from math import log, exp, sqrt, pi

# Site-package list
from scipy import integrate
import numpy as np
import pandas as pd


# CODING CONTENT
class Klett():
    def __init__(self,height, rcs, bin_width = 7.5, data_points = 16221, ref = 3):
        self.height = height
        self.rcs = rcs

        self.alpha = []
        self.bata = []
        self.k = 0.67
        self.LR_z = 50

        self.bin_width = bin_width # 单位:m
        self.data_points = data_points  # number_of_datapoints
        self.ref = ref

    def alpha_z(self):
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
                s = log(x ** 2 * self.rcs[h_index])  # 范围矫正信号(RCS)的自然对数值
                s_ref = log(self.height[self.ref] **
                            2 * self.rcs[self.ref])  # 参考高度ref下的RCS自然对数值
                delta_s = exp((s-s_ref) / self.k)
                return delta_s

            var_exp = _delta_s(h)
            # print(var_exp)

            # 计算参考高度0处范围矫正信号的自然对数
            var_s_0 = log(self.height[0] ** 2 * self.rcs[0])
            # 计算参考高度z_ref处范围矫正信号的自然对数
            var_s_ref = log(self.height[self.ref] ** 2 * self.rcs[self.ref])

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

class Fernald():
    def __init__(self):
        pass

    def beta_mol(self, r):
        '''
        大气分子后向散射系数与高度r的函数关系式
        '''
        c1 = 2.3850e-8
        c2 = 1012.8186 - 111.5569 * r + 3.8645 * (r ** 2)
        c3 = 294.9838 - 5.2159 * r - 0.0711 * (r ** 2)
        bata_mol = c1 * (c2 / c3)
        return bata_mol

    def alpha_mol(self, r, LR_mol):
        '''
        大气分子消光系数与高度r的函数关系式
        '''
        c4 = LR_mol
        bata_mol = self.beta_mol(r)
        alpha_mol = c4 * bata_mol
        return alpha_mol

    def inter_bata_mol(self, r, r_ref):
        '''
        求解bata_mol在[r,ref]区间的定积分
        '''
        v, err = integrate.quad(self.beta_mol, r, r_ref)
        print(v)

    def bata_aer(self, r):
        '''
        计算气溶胶后向散射系数
        '''
        pass

    def bata_r(self, rcs, r, r_ref=3, LR_aer=40, c=1):
        '''
        计算总后向散射系数
        rcs为list类型
        '''
        # 激光雷达比
        LR_mol = 8 * pi / 3
        rcs_ref = rcs[r_ref]

        # print(rcs_ref)

        # 计算参考高度ref处的后向散射系数bata_mol_ref
        def bata_mol(r):
            '''
            大气分子后向散射系数与高度r的函数关系式
            '''
            c1 = 2.3850e-8
            c2 = 1012.8186 - 111.5569 * r + 3.8645 * (r ** 2)
            c3 = 294.9838 - 5.2159 * r - 0.0711 * (r ** 2)
            return c1 * (c2 / c3)

        bata_mol_ref = bata_mol(r_ref)

        c4, err = integrate.quad(bata_mol, r, r_ref)

        # bata_r计算公式的分子部分
        c5 = rcs[r] * exp(2 * (LR_aer - LR_mol) * c4)

        # bata_r计算公式的分母部分

        c6 = rcs_ref / (c * bata_mol_ref)
        # print(rcs_ref,c4,c5,c6)

        c7 = rcs[r] * exp(2 * (LR_aer - LR_mol) * c4)
        print(c7)

        # c7, err = integrate.quad(c6, r, r_ref)

        # bata_r = c5 / (c6 + 2*LR_aer*c7)

        # return bata_r

    rcs = [5.241926131, 3.26489019, 3.795168048, 2.361559713, 2.075695638, 1.972212843]
    bata_r(rcs, r=1)


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
