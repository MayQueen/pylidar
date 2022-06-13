import numpy as np


# /*--------------------激光雷达数据修正-------------------------------*\

def smooth(x, window_len: 'int' = 11, window: 'str' = 'hanning'):
    """
    对信号进行平滑处理
    Args:
        x: 待处理的信号
        window_len: 平滑窗口的长度
        window: 平滑窗口的类型

    Returns:
        平滑後信號[numpy.array]
    """
    if type(x) is not np.array:
        # print('請檢查輸入數據是否為np.array類型')
        x = np.array(x)
    else:
        if x.ndim != 1:
            raise ValueError('smooth only accepts 1 dimension arrays.')
        if x.size < window_len:
            raise ValueError('Input vector needs to be bigger than window size.')

        if window_len < 3:
            if window not in ('flat', 'hanning', 'hamming', 'bartlett', 'blackman'):
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            if window == 'flat':
                w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        s = np.r_[(2 * x[0] - x[window_len - 1::-1], x, 2 * x[(-1)] - x[-1:-window_len:-1])]
        y = np.convolve((w / w.sum()), s, mode='same')
        return y[window_len:-window_len + 1]


def overlopCorrect(olData, lidarData):
    """
    对雷达信号进行重叠信号修正
    Args:
        olData: 几何重叠因子数据,pandas.DataFrame
        lidarData: 激光雷达原始信号,为pandas.DataFrame结构

    Returns:
        重叠修正后的数据
    """
    len_ol, len_lidar = len(olData), len(lidarData)
    len_diff = len_lidar - len_ol
    ol_corect = lambda x: x / olData.iloc[:, 1].values
    data_corrected = lidarData.iloc[len_diff:, :].apply(ol_corect)
    return data_corrected


def noiseCorrect(ol_corect_data, height_resolution: float, noise_altitude=None):
    """
    对雷达信号进行背景噪声去除
    Args:
        ol_corect_data: 几何重叠因子订正后的回波信号,数据接口采用pandas的Series或DataFrame
        height_resolution: 激光雷达信号垂直分辨率，单位为：km,与noise_altitude单位一致
        noise_altitude: 背景噪声选取高度，单位为:km，为（）元祖，代表起、始高度

    Returns:
        返回背景噪声订正后的回波信号->DataFrame

    # Add single profile surport 20210811
    # 示例: ol_corect_data为单条廓线或一段时间内数据，数据结构采用Series或DataFrame
            index为高度，单位可为m或km
            height_resolution是高度垂直分辨率，与noise_altitude单位一致
            noise_altitude为背景噪声选取高度，采用元表代表起始高度，如（8，10）
            采用pandas数据结构的好处是，index包含高度信息，无需再传入高度作为参数，保持接口的简洁
    """
    if noise_altitude is None:
        noise_altitude = (17, 18)
    else:
        noise_altitude = noise_altitude
    start_index, end_index = int(noise_altitude[0] / height_resolution), int(noise_altitude[1] / height_resolution)
    if len(ol_corect_data.shape) == 2:
        noise_frame = ol_corect_data.iloc[start_index:end_index + 1, :].mean()
    else:
        noise_frame = ol_corect_data.iloc[start_index:end_index + 1].mean()
    data_corrected = ol_corect_data - noise_frame
    return data_corrected


def disCorrect(bak_corect_data, height_unit=None):
    """
    对雷达信号进行距离平方订正
    Args:
        bak_corect_data: 进行背景修正后的雷达数据
        height_unit: 单位与后散单位一致,默认为km,0.001

    Returns:
        距离修正后的雷达数据
    """
    if height_unit is None:
        height_unit = 0.001
    else:
        height_unit = height_unit
    height = bak_corect_data.index.values * height_unit
    dis_correct = lambda x: x * height ** 2
    if len(bak_corect_data.shape) == 2:
        data_corrected = bak_corect_data.apply(dis_correct)
    else:
        data_corrected = bak_corect_data * height ** 2
    return data_corrected


# /*--------------------激光雷达方程反演-------------------------------*\
def calcMolecularBeta(height, lamda_length=None, ref_height=None):
    """
    计算空气分子后向散射系数计算
    Args:
        height: 单位与所求的后散单位一致，默认为km
        lamda_length: 单位为nm,与ref_lamda_length相除后无量纲
        ref_height: 单位为km

    Returns:
        分子后向散射系数
    """
    if lamda_length is None:
        lamda_length = 532
    else:
        lamda_length = lamda_length
    if ref_height is None:
        ref_height = 7
    else:
        ref_height = ref_height
    ref_lamda_length = 532
    beta_mol = (ref_lamda_length / lamda_length) ** 4.09 * np.exp(-height / ref_height) * 1.54 * 0.001
    return beta_mol


def fernaldAlgorithm(dis_corect_data, s_aer, height_resolution, beta_mol, inver_height=None):
    """
    采用fernald法反演雷达方程
    Args:
        # 数据接口为DataFrame或者Series
        dis_corect_data: 几何重叠因子订正、背景噪声订正和距离平方订正后的回波信号
        s_aer: 气溶胶激光雷达比
        height_resolution: 信号垂直分辨率,单位是km,和inver_height一致
        beta_mol: 空气分子后向散射系数廓线
        inver_height: 反演选取的边界高度，单位是km

    Returns:
        气溶胶后后向散射系数
    """
    if inver_height is None:
        inver_height = 12
    else:
        inver_height = inver_height
    lidar_sig = np.array(dis_corect_data)
    if len(lidar_sig.shape) == 2:
        ls_row, ls_col = lidar_sig.shape
    else:
        lidar_sig = lidar_sig.reshape((len(lidar_sig), 1))
        ls_col = 1
        ls_row = len(dis_corect_data)
    inver_height_index = int(inver_height / height_resolution)
    beta_aer = np.zeros([inver_height_index, ls_col])
    s_mol = 8 * np.pi / 3.0 # 分子后向散射系数比
    for ii in np.arange(ls_col):
        nzc = inver_height_index - 1
        beta_aer[(nzc - 1, ii)] = beta_mol[nzc]
        for jj in np.arange(nzc - 1, 0, -1):
            try:
                # Fernald 方法后向积分式，采用梯形面积法计算积分
                psy = np.exp((s_aer - s_mol) * (beta_mol[(jj - 1)] + beta_mol[jj]) * height_resolution)
                a1 = lidar_sig[(jj - 1, ii)] * psy
                b1 = lidar_sig[(jj, ii)] / (beta_mol[jj] + beta_aer[(jj, ii)])
                c1 = s_aer * ((lidar_sig[(jj, ii)] + lidar_sig[(jj - 1, ii)]) * psy) * height_resolution
                beta_aer[(jj - 1, ii)] = a1 / (b1 + c1) - beta_mol[(jj - 1)]
            except:
                beta_aer[(jj - 1, ii)] = np.nan

    return beta_aer