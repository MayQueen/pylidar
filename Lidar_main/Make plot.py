# -*- encoding: utf-8 -*-
"""
@NAME      :Make plot.py
@TIME      :2021/03/03 14:57:45
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
"""

import matplotlib.pyplot as plt
import pandas as pd

class plot (object):
    pass

data_path = "plot_data.xlsx"
x = pd.read_excel(data_path,sheet_name="x", index_col=None)
y = pd.read_excel(data_path,sheet_name="y", index_col=None)
z = pd.read_excel(data_path,sheet_name="z", index_col=None)


def plot_heatmap(pd_x, pd_y, pd_z, str_date_range):
    """
    根据x,y,z数据绘制热力图
    """

    # 绘图
    fig, ax = plt.subplots()
    c = ax.pcolor(x, y, z, cmap='gist_rainbow', shading='auto')

    # 设置标题、坐标轴名称
    plt.xlabel('Time')
    plt.ylabel('Height(km)')
    plot_title = r"Lidar singal heatmap" + r"[" + str(date_range) + r"]"
    ax.set_title(plot_title)

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


date_range = r"2019-01-20"

plot_heatmap(x, y, z, date_range)
