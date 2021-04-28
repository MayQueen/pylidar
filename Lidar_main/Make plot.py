# -*- encoding: utf-8 -*-
"""
@NAME      :Make plot.py
@TIME      :2021/03/03 14:57:45
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class plot (object):
    """
    数据可视化函数
    """

    def __init__(self):
        pass

    def plot_pbl(self,pd_pbl=[],):
        # Data for plotting
        x = pd_pbl["Time"]
        y = pd_pbl["pbl"]

        y_mean = pd.DataFrame.mean(y)
        # print(x.iloc[-9])

        fig, ax1 = plt.subplots()
        ax1.plot(x, y, 'go-')
        ax1.set(title='PBL trend graph over time', ylabel='Height(km)')
        ax1.text(x.iloc[-9],y.iloc[-10],"Mean PBL height \n %d (km)" % y_mean)
        plt.axhline(y=y_mean, c="r", ls="--", lw=2) # 水平参考线

        plt.xticks(rotation=90)
        plt.show()

    def plot_heatmap(self,pd_x, pd_y, pd_z, date_range):
        """
        根据x,y,z数据绘制热力图
        """

        # 绘图
        fig, ax = plt.subplots()
        c = ax.pcolor(pd_x, pd_y, pd_z, cmap='gist_rainbow', shading='auto')

        # 设置标题、坐标轴名称
        plot_title = r"Lidar singal heatmap" + r"[" + str(date_range) + r"]"
        ax.set(title = plot_title, xlabel='Time',ylabel='Height(km)')

        # 设置坐标轴日期格式
        dt = []
        for s in range(1,29):
            if s < 10:
                _dt = r"23:" + r"0%s" % s
            else:
                _dt = r"23:" + r"%s" % s
            dt.append(_dt)

        plt.xticks([_x for _x in range(1, 29, 1)], dt)
        plt.xticks(rotation=90)

        # plt.grid() # 显示网格

        # 颜色条设置
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.set_ylabel('RCS Intensity')
        plt.show()

rcs_path = "plotdata/plot_data.xlsx"
pbl_path = "./Results/pbl/pbl.csv"
# x = pd.read_excel(rcs_path,sheet_name="x", index_col=None)
# y = pd.read_excel(rcs_path,sheet_name="y", index_col=None)
# z = pd.read_excel(rcs_path,sheet_name="z", index_col=None)
pbl = pd.read_csv(pbl_path)
# print(pbl)
# date_range = r"2019-01-20"
#
tplot = plot()
# tplot.plot_heatmap(x, y, z, date_range)
tplot.plot_pbl(pbl)
