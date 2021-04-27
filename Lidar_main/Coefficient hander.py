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
from b_Klett_method import Klett


if __name__ == "__main__":
    print('开始读取RCS数据')
    RM_CSV_PATH = r"./Results/RCS/RM1912000021110.csv"  # 这个RCS结果由RM官方软件导出
    rcs_file = pd.read_csv(RM_CSV_PATH)
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
    dataframe.to_csv('./Results/Coefficient/Results_Coefficient.csv', sep=',')
    print('程序运行结束')
