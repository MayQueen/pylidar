# -*- encoding: utf-8 -*-
"""
@NAME      :Klett method.py
@TIME      :2021/03/25 11:10:28
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
"""

import numpy as np
import pandas as pd


class pbl_cal(object):
    """
    边界层反演算法
    """

    def __init__(self, pd_rcs=[], pd_h=[]):
        self.rcs = rcs
        self.file_header = self.rcs.columns
        self.h = h
        self.pbl = []

    def CRGM(self):
        """
        重力波梯度法反演边界层高度
        """

        # STEP1: 求RCS值的开立方
        cbrt_rcs = np.cbrt(self.rcs)
        # print(cbrt_rcs)

        # STEP2: 求梯度值
        for ichannel in range(len(self.file_header)):
            header = self.file_header[ichannel]  # 行标题“RM1” “RM2” “RM3”等
            grad_cbrt_rcs = np.gradient(cbrt_rcs[header])

            # STEP3: 返回梯度最小值处对应散射信号的高度
            i_min = np.argmin(grad_cbrt_rcs, axis=0)  # 最小值索引
            _pbl = self.h[header][i_min]
            self.pbl.append(_pbl)
        self.pbl = pd.DataFrame(self.pbl, index=self.file_header)
        # print(self.pbl)


data_path = "rcs.xlsx"
h = pd.read_excel(data_path, sheet_name="H", header=0, index_col=0)
rcs = pd.read_excel(data_path, sheet_name="RCS", header=0, index_col=0)

pbl_c = pbl_cal(rcs, h)
pbl_c.CRGM()
