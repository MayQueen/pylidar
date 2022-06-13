import os
import struct   # 读取二进制文件
import datetime

import numpy as np
import pandas as pd


# /*-----------------------文件读写----------------------------*\
def get_all_file(fileDir, filter_str):
    """
    獲取路徑下所有licel文件信息
    Args:
        fileDir: fileDir Licel文件夹路径
        filter_str: filter_str:文件特殊字符,如‘L','RM'等

    Returns:
        [所有licel文件个数,所有licel文件的文件名,所有licel文件的路径]
    """
    all_fn = os.listdir(fileDir)
    filter_fn = sorted([af for af in all_fn if filter_str in af])
    licel_file = [os.path.join(fileDir, ff) for ff in filter_fn]
    return len(licel_file), filter_fn, licel_file


def gen_obs_time_from_fn(fileDir, filter_str):
    """
    從lice文件名中提起日期序列
    Args:
        fileDir:
        filter_str:

    Returns:

    """
    data_num, all_fn, all_fn_dir = get_all_file(fileDir, filter_str)
    if filter_str == 'RM':
        time_str = [af[7:9] + ':' + af[10:12] + ':' + af[12:14] for af in all_fn]
    else:
        if filter_str == 'DQS':
            time_str = [af[11:13] + ':' + af[14:16] + ':' + af[16:18] for af in all_fn]
        else:
            time_str = [af[6:8] + ':' + af[9:11] + ':' + af[11:13] for af in all_fn]
    return time_str


def get_obs_time_from_single_file(fname: str):
    """
    從單個licel文件中獲取日期
    Args:
        fname: 輸入文件名

    Returns:
        pandas datetime 類型
    """

    time_str = fname.split('.')[0].split('_')[(-2)] + ' ' + fname.split('.')[0].split('_')[(-1)]
    return pd.to_datetime(time_str)


# /*--------------------文件读取-------------------------------*\
def read_single_licel(fileDir, headerline=None, line_spliter=None):
    """
    读取单个licel格式雷达数据文件
    Args:
        fileDir: Licel文件存储路径,exp'/fidir/L**.**'
        headerline:  文件头行数,不同licel可能不一样,默认羊八井大气所lidar的headerline为3行
        line_spliter: 每个通道数据后是否有回车,raman没有,iap有

    Returns:
        chanel_info_lst: 头文件信息
        data_dict: 字典,每个通道BC0等为key,字典值是原始信号
    eg.
    函数调用样例:read_single_licel(文件路径,headerline=3,line_spliter=2)
    """
    h_lines = []
    with open(fileDir, 'rb') as (f):
        if headerline:
            h_lines.append([f.readline() for hl in range(int(headerline))])
        else:
            filename = f.readline()     # licel文件第1行，文件名
            local_info = f.readline()   # licel文件第2行，站点信息
            laser_info = f.readline()   # licel文件第3行，激光发射器信息
        chanel_info = []    
        chanel_info_lst = []
        data_dict = {}
        FLAGREADHEADER = True
        while FLAGREADHEADER:
            temp_line = f.readline()    # licel文件第4—6行，通道信息
            current_line_len = len(temp_line)
            if current_line_len != 2:
                chanel_info.append(temp_line)
            else:
                FLAGREADHEADER = False

        for ii, line in enumerate(chanel_info):
            line_strip = line.strip().split()
            chanel_info_lst.append(line_strip)

        for j, cil in enumerate(chanel_info_lst):
            data_len = int(chanel_info_lst[j][3])
            if line_spliter is None:
                temp_data = f.read(data_len * 4)    # licel文件数据块
                data_int = struct.unpack(str(data_len) + 'i', temp_data)
            else:
                temp_data = f.read(data_len * 4 + 2)
                data_int = struct.unpack(str(data_len) + 'i2s', temp_data)
                data_int = data_int[0:-1]
            data_dict[str(chanel_info_lst[j][(-1)], 'utf-8')] = data_int

    return chanel_info_lst, data_dict


def read_batch_files(fileDir, filter_str, pickChanel=None, profileLen=None, headerline=None, line_spliter=None):
    """
    批量读取licel格式雷达数据文件
    Args:
        fileDir: 原始数据路径
        filter_str: 原始数据包含的特殊字符串,如'L','RM'等
        pickChanel: 拟挑选的通道名称,如BT0,BC0等
        profileLen: 每个通道的数据长度,批量读取时需要制定,单独文件不需要
        headerline: 头文件函数,基本都为3行
        line_spliter: 每个通道后是否添加了回车,默认没有,有的话赋值line_spliter=2

    Returns:.
        timeindex:每个廓线的时间点
        return_dict:当天所有时刻廓线组成的data_frame
    eg.
        heig_rev, timeindex, dataFrame = read_batch_files('../20190201',
        filter_str='RM',pickChanel='BC0',profileLen=16380,line_spliter=None)
    """
    # channel设置
    if pickChanel:
        pickChanel = pickChanel
    else:
        pickChanel = ('BT0', 'BT1', 'BC0', 'BC1')

    # 文件头设置
    if line_spliter:
        line_spliter = line_spliter
    elif headerline:
        headerline = headerline
    else:
        headerline = 3

    file_num, fn, fn_dir = get_all_file(fileDir, filter_str)  # 獲取路徑下的licel文件的數量,文件名和路徑
    del fn
    # print(fn_dir)
    time_index = gen_obs_time_from_fn(fileDir, filter_str)

    # 获取channel信息
    chanel_info, _ = read_single_licel((fn_dir[0]), headerline=headerline, line_spliter=line_spliter)
    chanel_dict = {}
    for ii, info in enumerate(chanel_info):
        chanel_dict[str(info[(-1)], 'utf-8')] = [
            int(info[3]), float(info[6]), str(info[7], 'utf-8')]

    # 获取channel数据
    if profileLen:
        profileLen = profileLen
    else:
        try:
            profileLen = chanel_dict[pickChanel[0]][0]
        except:
            print('Chanel %s does not exist for this profile!' % pickChanel[0])
            profileLen = chanel_dict['BC0'][0]
            
        # 获取335.p(BT0 BT1)和335.s(BC0 BC1)数据,387.0数据并非每个文件都有
        data_series_BT0 = np.zeros((profileLen, file_num))
        data_series_BT1 = np.zeros((profileLen, file_num))
        data_series_BC0 = np.zeros((profileLen, file_num))
        data_series_BC1 = np.zeros((profileLen, file_num))

        for ii, fd in enumerate(fn_dir):
            # print('Reading File %s:' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            try:
                _channeldict, temp_data = read_single_licel(fd, headerline=headerline, line_spliter=line_spliter)
                del _channeldict
                data_series_BT0[:, ii] = temp_data['BT0']
                data_series_BT1[:, ii] = temp_data['BT1']
                data_series_BC0[:, ii] = temp_data['BC0']
                data_series_BC1[:, ii] = temp_data['BC1']
            except:
                data_series_BT0[:, ii] = np.nan
                data_series_BT1[:, ii] = np.nan
                data_series_BC0[:, ii] = np.nan
                data_series_BC1[:, ii] = np.nan
                print('File %s is Wrong!' % fd)

        data_dict = {'BT0': data_series_BT0, 'BT1': data_series_BT1,
                     'BC0': data_series_BC0, 'BC1': data_series_BC1}
        return_dict = {key: data_dict[key] for key in pickChanel}
        return time_index, return_dict


# /*--------------------数据重采样-------------------------------*\
def gen_dataframe(datain, timeindex, ver_resolution=None):
    """
    根据时间段截取雷达数据
    Args:
        datain:
        timeindex:
        ver_resolution:

    Returns:

    """
    if ver_resolution is None:
        ver_resolution = 7.5
    else:
        ver_resolution = ver_resolution
    data = pd.DataFrame(datain, columns=timeindex)
    data_len = data.shape[0]
    height = np.arange(ver_resolution, data_len * ver_resolution + ver_resolution, ver_resolution)
    data.index = height
    return data

# /*-------------------结果文件保存---------------------------*\
def save_all_result(chose_date: 'str', out_csv: 'str', df_data_list:'list'):
    """

    Args:
        schose_date:
        out_csv:
        total_raw:
        total_rcs:
        dr_data:
        beta_aero:
        extinc_aero:
        pblh_and_index:
        ch:

    Returns:

    """
    # 保存文件
    if not os.path.exists(out_csv):
        os.makedirs(out_csv)

    tt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fn_name_list = ['RAW','DepolarizationRatio','RCS','pblHeight','Cloud',
                    'BackScattering','Extinction','AerosolOpticalDepth',
                    'Visibility']    
    for index,value in enumerate (df_data_list):
        var_fn = chose_date + "_" + fn_name_list[index] + ".csv"
        var_fn_dir = os.path.join(out_csv,var_fn)
        if not os.path.exists(var_fn_dir):
            try:
                print("[%s]" % tt, "\t", "准备保存{}".format(fn_name_list[index]))
                df_data_list[index].to_csv(var_fn_dir, mode='w')  # 写入硬盘
            except Exception as e:
                print("[%s]" % tt, "\t", "{}保存失败".format(fn_name_list[index]), e)