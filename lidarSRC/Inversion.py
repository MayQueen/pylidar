import numpy as np
import pandas as pd
from adtk.detector import LevelShiftAD  # 時間序列異常檢測


# /*--------------------物理量反演-------------------------------*\
def calcSNR(datain, snr=0.85, snr_bound=3000):
    """
    计算雷达信号的信噪比
    Args:
        datain: pandas.Series
        snr: threshold
        snr_bound: min altitude snr < threshold

    Returns:
        信噪比->int
    # 如果data是DataFrame
    snr = data.apply(_calc_snr,snr=0.9)
    """
    height_unit = 1000
    noise_altitude = (18, 20)
    data_in_resolu = datain.index[1] - datain.index[0]
    noise = datain.iloc[int(noise_altitude[0] * height_unit / data_in_resolu):int(
        noise_altitude[1] * height_unit / data_in_resolu)].mean()
    snr_range = datain / (datain + abs(noise))
    snr_small_height = datain.index[np.where(snr_range < snr)[0][0]]
    if snr_small_height < snr_bound:
        return datain.index[np.where(snr_range < snr)[0][1]]
    return snr_small_height

def calcPBL(rawdata_df, max_height=1000, min_height=80):
    """
    梯度法反演边界层高度
    Args:
        data_in: pandas.Series
        max_p: 最大高度,
        lower_in: 最低高度, max_pblh,lower_index

    Returns:
        邊界層高度
    """
    height_reso = rawdata_df.index[1] - rawdata_df.index[0]   # 垂直分辨率
    higher_index = int(max_height / height_reso)  # 构造索引
    rdata_pick = rawdata_df.iloc[min_height:higher_index]  # 截取数据
    data_pick = np.cbrt(rdata_pick)     # 开立方
    da_pick_diff = np.diff(data_pick)   # 计算梯度
    min_gradient_index = np.where(da_pick_diff == np.nanmin(da_pick_diff))  # 找到忽略nan值梯度最小索引
    try:
        pblh_index = min_height + min_gradient_index[0][0]
        pblh = rawdata_df.index[pblh_index]    # 获取索引
    except:
        pblh_index = np.nan
        pblh = np.nan

    return pblh, pblh_index

def calcCloud(rawdata_df, min_level_height=None, start_level_height=None, detect_chose=None):
    """
    計算雲底高度
    Args:
        # beta_aero:np.ndarray
        # 阈值法判断当前时刻是否有云,在有云存在时再检测云底和云顶高度
        # beta_aero的column是时间,raw是高度,高度范围小于raw_df
        raw_df: pandas.DataFrame
        min_level_height:
        start_level_height:
        detect_chose:

    Returns:
        雲相關參數
    """
    # 参数初始化
    cloud_top_height = []
    cloud_top_height_index = []
    cloud_base_height = []
    cloud_base_height_index = []
    len_h, len_t = rawdata_df.shape

    if min_level_height is None:
        max_pblh = 8000
    else:
        max_pblh = min_level_height
    if start_level_height is None:
        lower_index = 150
    else:
        lower_index = start_level_height
    if detect_chose is None:
        detect_chose = 'both'
    else:
        detect_chose = detect_chose

    height_reso = rawdata_df.index[1] - rawdata_df.index[0]
    higher_index = int(max_pblh / height_reso)
    level_shift_ad = LevelShiftAD(c=30, side='both', window=3)  # 时间序列异常检测
    for ii in range(len_t):
        s = rawdata_df.iloc[lower_index:higher_index, ii]
        s = pd.Series(abs(np.diff(s)))
        fake_index = np.arange(0, len(s))
        s.index = pd.to_datetime(fake_index)
        anomalies = level_shift_ad.fit_detect(s)
        try:
            index1 = int(str(anomalies[(anomalies == 1)].index[(-1)])[-3:]) + lower_index + 1
            index2 = int(str(anomalies[(anomalies == 1)].index[0])[-3:]) + lower_index
        except:
            index1 = 0
            index2 = 0

        if abs(index1 - index2) >= 200 or abs(index1 - index2) <= 4:
            cloud_top_height.append(np.nan)
            cloud_top_height_index.append(np.nan)
            cloud_base_height.append(np.nan)
            cloud_base_height_index.append(np.nan)
        else:
            if index1 > index2:
                top_index = index1
                base_index = index2
            else:
                top_index = index2
                base_index = index1
            cloud_top_index = top_index
            cloud_base_index = base_index
            cld_top_h = rawdata_df.index[cloud_top_index]
            cld_base_h = rawdata_df.index[cloud_base_index]
            cloud_top_height.append(cld_top_h)
            cloud_top_height_index.append(cloud_top_index)
            cloud_base_height.append(cld_base_h)
            cloud_base_height_index.append(cloud_base_index)

    return (cloud_base_height, cloud_base_height_index, cloud_top_height, cloud_top_height_index)

def calcAOD(Extdata_df: pd.DataFrame):
    """
    将消光系数随高度进行积分,就能够得到气溶胶的光学厚度(AOD)
    """
    len_h, len_t = Extdata_df.shape # 获取行,列即高度,时间
    height_reso = Extdata_df.index[1] - Extdata_df.index[0]    # 计算垂直分辨率
    aod = np.zeros(len_t)
    # t = np.trapz(Extdata_df.iloc[:,0],dx=height_reso)
    for ii in np.arange(len_t):
        # print(Extdata_df.iloc[:,ii])
        aod[ii] = np.trapz(Extdata_df.iloc[:,ii],dx=height_reso)    # np.trapz() 使用复合梯形规则沿给定轴进行积分。
    return aod
    # aod.to_csv('aod.csv')

def calcVIS(Extdata_df: pd.DataFrame):
    """
    基于激光雷达的垂直能见度反演算法及其误差评估 doi: 10.11884/HPLPB202032.190250
    """
    len_h, len_t = Extdata_df.shape # 获取行,列即高度,时间
    height_reso = Extdata_df.index[1] - Extdata_df.index[0]    # 计算垂直分辨率
    vis = np.zeros(len_t)
    for ii in np.arange(len_t):
        tmp_vis = np.zeros(len_h)
        for jj in np.arange(len_h):
            tmp_vis[jj] = np.trapz(Extdata_df.iloc[:jj,ii],dx=height_reso)    # np.trapz() 使用复合梯形规则沿给定轴进行积分。
        # print(vis)
        pd_vis = pd.DataFrame(tmp_vis, index=Extdata_df.index)
        _vis = pd_vis.iloc[:,0]
        index_list = _vis[_vis >= 3 ].index.tolist()
        # print(index_list)
        if len(index_list): # index_list 不为0
            vis[ii] = index_list[0]
        else: # 积分所有值都<3时，VIS为最大高度
            vis[ii] = Extdata_df.index.tolist()[-1]
    
    # vis.to_csv('vis.csv')
    return vis
    # print(vis)