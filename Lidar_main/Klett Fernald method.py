# -*- encoding: utf-8 -*-
'''
@NAME      :Klett Fernald method.py
@TIME      :2021/03/25 11:10:28
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
'''
# Inbuild package list
from math import log, exp, pi

# Site-package list
from scipy import integrate
import numpy as np
import pandas as pd


# CODING CONTENT
class Klett():
    """
    alpha: 消光系数 extintion coefficient
    beta : 后向散射系数 backscatter coefficient
    """

    def __init__(self, height, rcs, bin_width=7.5, data_points=16221, k = 1, ref=-10):
        self.height = height
        self.rcs = rcs

        self.alpha = []
        self.beta = []

        self.k = k
        self.LR_z = 50

        self.bin_width = bin_width  # 单位:m
        self.data_points = data_points  # number_of_datapoints
        self.ref = ref

    def alpha_z(self):
        '''
        根据Klett雷达反演方法求解激光雷达消光系数alpha_z
        '''
        print("开始计算消光系数alpha")
        # 根据Collis斜率法计算参考高度下的 大气消光系数
        s_0 = log(self.rcs[0])
        s_m = log(self.rcs[self.ref])
        alpha_ref = 0.5 * (s_0 - s_m) / (self.height[self.ref] - self.height[0])
        # print(alpha_ref)

        for h in range(len(self.height)+self.ref): # 仅能计算参考高度以下的数值
        # for h in range(10):
            s = log(self.rcs[h])
            s_m = log(self.rcs[self.ref])
            # print(exp((s - s_m)/self.k))
            exp_s = (s - s_m) / self.k

            def fun_rcs(x):
                return exp_s

            interg_s,err = integrate.quad(fun_rcs, h, self.ref)
            # print(intergra_s)

            # 根据Klett方法计算大气消光系数
            var_alpha_z = interg_s / ((1/alpha_ref) - (2/self.k)*exp_s)

            # 保存计算结果
            self.alpha.append(var_alpha_z)
        return self.alpha
        # print(self.alpha)

    def beta_z(self):
        '''
        # 计算后向散射系数beta_z
        '''
        print("开始计算后向散射系数beta")
        self.alpha_z()
        for index,alpha in enumerate (self.alpha):
            _beta = self.LR_z * alpha ** self.k
            self.beta.append(_beta)
            del  _beta
        # return self.beta
        # print(len(self.beta))


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
        beta_mol = c1 * (c2 / c3)
        return beta_mol

    def alpha_mol(self, r, LR_mol):
        '''
        大气分子消光系数与高度r的函数关系式
        '''
        c4 = LR_mol
        beta_mol = self.beta_mol(r)
        alpha_mol = c4 * beta_mol
        return alpha_mol

    def inter_beta_mol(self, r, r_ref):
        '''
        求解beta_mol在[r,ref]区间的定积分
        '''
        v, err = integrate.quad(self.beta_mol, r, r_ref)
        print(v)

    def beta_aer(self, r):
        '''
        计算气溶胶后向散射系数
        '''
        pass

    def beta_r(self, rcs, r, r_ref=3, LR_aer=40, c=1):
        '''
        计算总后向散射系数
        rcs为list类型
        '''
        # 激光雷达比
        LR_mol = 8 * pi / 3
        rcs_ref = rcs[r_ref]

        # print(rcs_ref)

        # 计算参考高度ref处的后向散射系数beta_mol_ref
        def beta_mol(r):
            '''
            大气分子后向散射系数与高度r的函数关系式
            '''
            c1 = 2.3850e-8
            c2 = 1012.8186 - 111.5569 * r + 3.8645 * (r ** 2)
            c3 = 294.9838 - 5.2159 * r - 0.0711 * (r ** 2)
            return c1 * (c2 / c3)

        beta_mol_ref = beta_mol(r_ref)

        c4, err = integrate.quad(beta_mol, r, r_ref)

        # beta_r计算公式的分子部分
        c5 = rcs[r] * exp(2 * (LR_aer - LR_mol) * c4)

        # beta_r计算公式的分母部分

        c6 = rcs_ref / (c * beta_mol_ref)
        # print(rcs_ref,c4,c5,c6)

        c7 = rcs[r] * exp(2 * (LR_aer - LR_mol) * c4)
        print(c7)

        # c7, err = integrate.quad(c6, r, r_ref)

        # beta_r = c5 / (c6 + 2*LR_aer*c7)

        # return beta_r

    # rcs = [5.241926131, 3.26489019, 3.795168048, 2.361559713, 2.075695638, 1.972212843]
    # beta_r(rcs, r=1)


if __name__ == "__main__":
    print('开始读取RCS数据')
    data_path = "Results/plotdata/Klett.xlsx"
    h = np.array(pd.read_excel(data_path, sheet_name="H", header=0, index_col=0))
    rcs = np.array(pd.read_excel(data_path, sheet_name="RCS", header=0, index_col=0))
    print('使用Kletthod方法反演雷达方程')
    RM = Klett(height=h, rcs=rcs)
    # alpha = RM.alpha_z()
    alpha = RM.beta_z()
    print('程序运行结束')
