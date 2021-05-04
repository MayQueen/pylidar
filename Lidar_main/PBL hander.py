# -*- encoding: utf-8 -*-
"""
@NAME      :Klett Fernald method.py
@TIME      :2021/03/25 11:10:28
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
"""

import numpy as np
import pandas as pd


class ubl_cal(object):
    """
    说明：边界层反演算法
    """

    def __init__(self, pd_rcs= [], pd_h= []):
        self.rcs = pd_rcs
        self.h = pd_h
        self.pbl = []

    def CRGM(self, to_csv=False):
        """
        说明：重力波梯度法反演边界层高度
        输入：DataFrame格式,rcs和h数据[mxn]矩阵
        输出：DataFrame或csv格式,pbl数据[mx1]行向量
        """

        # STEP1: 求RCS值的开立方
        cbrt_rcs = np.cbrt(self.rcs)
        # print(cbrt_rcs)

        # STEP2: 求梯度值
        for index, value in enumerate(self.rcs):
            # print(value)
            grad_cbrt_rcs = np.gradient(cbrt_rcs[value])
            # print(grad_cbrt_rcs)

            # STEP3: 返回梯度最小值处对应散射信号的高度
            i_min = np.argmin(grad_cbrt_rcs, axis=0)  # 最小值索引
            # print(i_min)
            _pbl = self.h[value][i_min]
            # print(_pbl)
            self.pbl.append(_pbl)
        self.pbl = pd.DataFrame(self.pbl)
        # print(self.pbl)

        if to_csv == True:
            self.pbl.to_csv("./Results/ubl/pbl2.csv", header=['PBL(km)'])
            print("PBL数据保存完成")
        else:
            print(self.pbl.head())


class mlh_cal():
    def __init__(self, pd_rcs=[], pd_h=[]):
        self.rcs = pd_rcs
        # self.file_header = self.rcs.columns
        self.h = pd_h
        self.pbl = []

    def fderi(self):
        pass


if __name__ == '__main__':
    RCS_path = "Results/plotdata/RCS20190120-00.CSV"
    H_path = "Results/plotdata/H20190120-00.CSV"
    RCS = pd.read_csv(RCS_path,header=None)
    H = pd.read_csv(H_path,header=None)

    pbl_c = ubl_cal(RCS, H)
    pbl_c.CRGM(to_csv=True)
