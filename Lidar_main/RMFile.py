# -*- encoding: utf-8 -*-
"""
@NAME      :RMFile.py
@TIME      :2021/03/04 16:01:52
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
"""

import numpy as np
import pandas as pd
import os


class RM_reader(object):
    """
    接受一个符合Licel格式的Lidar文件的python类文件对象(file-like-object),
    返回文件标题信息、通道信息,原始数据\n
    输入:\n
        `file-like`对象(可通过`python with-open`方法创建)
    输出:\n
        `标题信息,list,header_info`;
        `通道信息,list,channel_info`;
        `文件数据,list,file_data`;
    """

    def __init__(self, single_RM_PATH):
        # 文件路径
        self.path = single_RM_PATH

        # 头文件信息
        self.header_label = ['Filename', 'Site', 'Start_date', 'Start_time', 'End_date', 'End_time', 'Altitude',
                             'Longitude', 'Latitude',
                             'Zenith_angle', 'Azimuth_angle', 'Temperature', 'Pressure',
                             'LS1_number_of_shots', 'LS1_Frequency', 'LS2_number_of_shots', 'LS2_Frequency',
                             'Number_of_datasets']
        self.header_info = {}

        # 通道信息
        self.channel_header = ['Active', 'Analog_Ghoton', 'Laser_used', 'Number_of_datapoints', '1', 'HV',
                               'Bin_width', 'Wavelength', 'D1', 'D2', 'D3', 'D4', 'ADCbits', 'Number_of_shots',
                               'Input_range', 'ID']
        self.channel_info = []

        # 变量信息
        self.num_points = []
        self.bin_width = []
        self.wavelength = []
        self.adcbits = []
        self.num_shots = []
        self.input_range = []
        self.dataset_mode = []

        self.file_data = []

    # 读取文件第一行
    def read_first_line(self, f):
        """
        读取文件第一行:文件名
        """
        self.header_info["Filename"] = f.readline().decode().strip()

    # 读取文件第二行
    def read_second_line(self, f):
        """
        读取文件第二行:站点信息
        """
        second_line = f.readline().decode().strip()

        # 格式化第二行信息,以空格位分节符取出数据
        self.header_info['Site'] = str(second_line.split()[0])
        self.header_info['Start_date'] = str(second_line.split()[1])
        self.header_info['Start_time'] = str(second_line.split()[2])
        self.header_info['End_date'] = str(second_line.split()[3])
        self.header_info['End_time'] = str(second_line.split()[4])
        self.header_info['Altitude'] = float(second_line.split()[5])
        self.header_info['Longitude'] = float(second_line.split()[6])
        self.header_info['Latitude'] = float(second_line.split()[7])
        self.header_info['Zenith_angle'] = float(second_line.split()[8])
        self.header_info['Azimuth_angle'] = float(second_line.split()[9])
        self.header_info['Temperature'] = float(second_line.split()[10])
        self.header_info['Pressure'] = float(second_line.split()[11])

    # 读取文件第三行
    def read_third_line(self, f):
        """
        读取文件第三行:雷达发射器信息
        """
        third_line = f.readline().decode().strip()
        self.header_info['LS1_number_of_shots'] = int(third_line.split()[0])
        self.header_info['LS1_Frequency'] = int(third_line.split()[1])
        self.header_info['LS2_number_of_shots'] = int(third_line.split()[2])
        self.header_info['LS2_Frequency'] = int(third_line.split()[3])
        self.header_info['Number_of_datasets'] = int(third_line.split()[4])

    # 从文件header中读取雷达channel信息
    def get_channel_info(self, f):
        """
        获取文件的通道信息
        """
        for _idx1 in range(int(self.header_info["Number_of_datasets"])):  # ”Number of datasets“ 数值即为文件包含的通道数
            channel_line = f.readline().decode().strip().split()
            self.channel_info.append(channel_line)

    # 获取num_points
    def get_num_points(self):
        """
        获取每条通道的数据的条目数
        """
        self.num_points = [int(self.channel_info[_idx2][3])
                           for _idx2 in range(len(self.channel_info))]

    # 读取file_data
    def read_rest_line(self, f):
        """
        读取文件剩余所有行:数据
        """
        for _idx3 in self.num_points:
            # print(c4)
            f.readline().strip()  # 注意:此处空行尤为重要

            byte_data = np.fromfile(
                f, "i4", count=int(_idx3))  # 一次性读取出num_points行的数据(一个channel所有的数据)

            self.file_data.append(byte_data)
        # print(self.file_data)

    # 获取bin_width
    def get_bin_width(self):
        """
        读取文件的bin width
        """
        self.bin_width = [float(self.channel_info[_idx4][6])
                          for _idx4 in range(len(self.channel_info))]

    # 获取wavelength
    def get_wavelength(self):
        """
        获取每个channel的波长和极化方式.o=无极化;s=垂直;p=平行
        """
        self.wavelength = [str(self.channel_info[_idx5][7])
                           for _idx5 in range(len(self.channel_info))]

    # 获取adcbits
    def get_adcbits(self):
        """
        获取文件的adcbits
        """
        self.adcbits = [int(self.channel_info[_idx6][-4])
                        for _idx6 in range(len(self.channel_info))]

    # 获取num_shots
    def get_num_shots(self):
        """
        获取文件的发射次数（number of shots）
        """
        self.num_shots = [int(self.channel_info[_idx7][-3])
                          for _idx7 in range(len(self.channel_info))]

    # 获取input_range
    def get_input_range(self):
        """
        获取文件的input range
        """
        self.input_range = [float(self.channel_info[_idx8][-2])
                            for _idx8 in range(len(self.channel_info))]

    # 获取dataset_mode：数据类型
    def get_dataset_mode(self):
        """
        获取每个channel的数据集的类型，BC=模拟信号数据集;BT=光子信号数据集;PD=模拟和光子混合数据集
        """

        for _idx9 in range(len(self.channel_info)):
            self.dataset_mode.append(
                str(self.channel_info[_idx9][-1][0:2]))  # BT=analogue dataset, BC=photon counting

    # !!! 读取单个RM File文件
    def single_reader(self):
        """
        类函数操作入口:RM文件读取操作函数
        """
        with open(self.path, "rb") as f:
            # 读取文件第一行
            self.read_first_line(f)
            # 读取文件第二行
            self.read_second_line(f)
            # 读取文件第三行
            self.read_third_line(f)

            # 读取雷达信息
            self.get_channel_info(f)
            self.get_num_points()
            self.read_rest_line(f)

            # 主要参数获取
            self.get_bin_width()
            self.get_wavelength()
            self.get_adcbits()
            self.get_num_shots()
            self.get_input_range()
            self.get_dataset_mode()


class muti_RM_reader(RM_reader):
    """
    说明：读取多个RM文件
    输入：文件路径
    可用的方法：save_header_info;save_channel_info
    属性：muti_header_info;muti_channel_info;muti_file_data
    """

    def __init__(self, RM_PATH_LIST):
        # self.muti_header_info = []  # 多个文件的头文件

        # self.muti_channel_info = []  # 多个文件的通道
        # self.muti_file_data = []  # 多个文件的数据

        self.path_list = RM_PATH_LIST  # 多个文件的路径

        print("该目录下共有%s" % len(self.path_list) + '个RM文件' + '\n')
        muti_header_info, muti_channel_info, muti_file_data, muti_wavelength, muti_dataset_mode = [], [], [], [], []
        for ipx, value in enumerate(self.path_list):
            # 继承RM_Reader类
            single_RM = RM_reader(value)

            print('# 准备读取第%s' % str(ipx + 1) + '个文件')
            single_RM.single_reader()  # 主要

            muti_header_info.append(single_RM.header_info)  # 列表字典嵌套[{},{}]
            muti_channel_info.append(single_RM.channel_info)  # 列表嵌套[[[],[]],[[],[]]]
            muti_file_data.append(single_RM.file_data)  # 列表嵌套[[array[],array[],array[]],[array[],array[],array[],]]
            muti_wavelength.append(single_RM.wavelength)  # 列表嵌套[[array[],array[],array[]],[array[],array[],array[],]]
            muti_dataset_mode.append(
                single_RM.dataset_mode)  # 列表嵌套[[array[],array[],array[]],[array[],array[],array[],]]

        self.muti_header_info = pd.DataFrame(muti_header_info)
        self.muti_channel_info = pd.DataFrame(muti_channel_info, index=self.muti_header_info['Filename']).T
        self.muti_file_data = pd.DataFrame(muti_file_data, index=self.muti_header_info['Filename'],
                                           columns=muti_wavelength[0]).T
        del muti_header_info, muti_channel_info, muti_file_data, muti_wavelength, muti_dataset_mode
        # print(self.muti_header_info)

    def save_header_info(self):
        _csv_path = './Results/RMFiles/RM_File_header_info.csv'
        self.muti_header_info.to_csv(_csv_path)
        print('# RM_File_info文件保存完成')

    def save_channel_info(self):
        _csv_path = './Results/RMFiles/RM_File_channel_info.csv'
        self.muti_channel_info.to_csv(_csv_path)
        print('# RM_File_Channel_info文件保存完成')


class RM_Cal(RM_reader):
    def __init__(self, SINGLE_RM_PATH):
        self.single_RM = RM_reader(SINGLE_RM_PATH)
        self.single_RM.single_reader()  # 主要

        self.bin_width = self.single_RM.bin_width
        self.num_points = self.single_RM.num_points

        self.dataset_mode = self.single_RM.dataset_mode
        self.wavelength = self.single_RM.wavelength

        self.input_range = np.array(self.single_RM.input_range)
        self.adcbits = np.array(self.single_RM.adcbits)
        self.num_shots = np.array(self.single_RM.num_shots)
        self.file_data = self.single_RM.file_data

        self.channel_info = self.single_RM.channel_info
        self.channel_header = self.single_RM.channel_header
        self.channel = pd.DataFrame(self.channel_info, columns=self.channel_header)

        self.header_info = self.single_RM.header_info
        self.header_label = self.single_RM.header_label

        self.Filename = self.single_RM.header_info["Filename"]

    def get_height(self):
        """
        根据channel的信息计算height
        输入:self.bin_with, self.num_points
        输出:self.height
        """
        print('# 开始计算Height')
        t_height = []
        for index, value in enumerate(self.bin_width):
            # print(value)
            num_bin = np.array([num_bin for num_bin in range(1, int(self.num_points[index]) + 1)])
            # print(num_bin)
            _height = num_bin * float(value)
            # print(_height)
            t_height.append(_height)
        self.height = pd.DataFrame(t_height).T
        del _height, t_height
        # print(self.height)

    def get_raw(self):
        # 根据dataset_mode类型,选择计算公式
        print('# 开始计算Raw')
        _raw = []
        for index in range(len(self.dataset_mode)):
            mode = self.dataset_mode[index]
            channel_data = self.file_data[index]
            # print(channel_data.shape)
            if mode == 'BT':  # 模拟信号
                # print('模拟信号')
                _mVolts = (channel_data * self.input_range[index] * 1000) / (
                        2 ** self.adcbits[index] * self.num_shots[index])
                # print(_mVolts)
                _raw.append(_mVolts)
            elif mode == 'BC':  # 光信号
                # print('光信号')
                _MHz = (channel_data * (150 / self.bin_width[index])) / self.num_shots[index]
                # print(_MHz)
                _raw.append(_MHz)
            else:  # 混合信号
                # print('混合信号')
                _mix = (channel_data * self.input_range[index] * 1000) / (
                        2 ** self.adcbits[index] * self.num_shots[index])
                # print(_mix)
                _raw.append(_mix)
        self.raw = pd.DataFrame(_raw).T
        print('[Raw]计算完成' + '\n')
        # print(self.raw)

    def get_rcs(self):
        """
        计算范围矫正信号(RCS,range corrected signal)
        """
        self.get_height()
        self.get_raw()  # 计算rcs前必须先计算Raw

        print('# 开始计算RCS')
        # print(self.height)
        # print(self.raw)

        _rcs = []
        for index, value in enumerate(self.raw):
            # print(index)
            # STEP 1:Overlap Corrected signal 去除低空重叠信号
            OCS = self.raw[index][20:]
            h = self.height[index][20:]
            # print(OCS.shape)
            # print(h.shape)

            # STEP 2:Background Corrected signal 背景修正
            background_error = OCS.min()
            # print(background_error)
            BCS = np.array((OCS - background_error))  # 去除噪声信号

            # STEP 3:Range Corrected Signal 计算距离修正信号
            h_2 = np.array(np.square(h))  # height^2
            # print(h_2)

            RCS = np.multiply(BCS, h_2)
            # print(RCS)
            _rcs.append(RCS)
        self.rcs = pd.DataFrame(_rcs).T
        # return self.rcs
        # print(self.rcs)

    def to_csv(self, save_header=True, save_channel=True, save_height=False, save_raw=False, save_rcs=True):
        if save_header:
            self.header_label = np.array(self.header_label).T
            self.header = pd.DataFrame(self.header_info.values(), index=self.header_label)
            print('准备保存[Header]数据')
            self.header.to_csv('./Results/RMFiles/File/HeaderInfo_%s' % str(self.Filename) + '.csv',
                               float_format='%.6f', header=None)
            print('[Header]保存完成' + '\n')

        if save_channel:
            print('准备保存[Channel]数据')
            self.channel.to_csv('./Results/RMFiles/File/ChannelInfo_%s' % str(self.Filename) + '.csv',
                                float_format='%.6f')
            print('[Channel]保存完成' + '\n')
        if save_height:
            self.get_height()
            print('准备保存[Height]数据')
            self.height.to_csv('./Results/RMFiles/File/Height_%s' % str(self.Filename) + '.csv',
                               float_format='%.6f', header=self.wavelength)
            print('[Height]保存完成' + '\n')
        if save_raw:
            self.get_raw()
            print('准备保存[Raw]数据')
            self.raw.to_csv('./Results/RMFiles/File/Raw_%s' % str(self.Filename) + '.csv',
                            float_format='%.6f', header=self.wavelength)
            print('[Raw]保存完成' + '\n')
        if save_rcs:
            self.get_rcs()
            print('准备保存[RCS]数据')
            self.rcs.to_csv('./Results/RMFiles/File/RCS_%s' % str(self.Filename) + '.csv',
                            float_format='%.6f', header=self.wavelength)
            print('[RCS]保存完成' + '\n')


class muti_RM_Cal(RM_Cal):
    def __init__(self, muti_path_list):
        self.path_list = muti_path_list

    def cal_raw(self):
        _raw = []
        for index, value in enumerate(self.path_list):
            print('准备处理第%s个文件' % str(index + 1) + '\n')
            # print(index)
            rm_cal = RM_Cal(value)
            rm_cal.get_raw()
            _raw.append(rm_cal.raw)
        self.muti_raw = np.array(_raw)

    def save_raw(self):
        pass


if __name__ == "__main__":
    print('\n' + '程序开始运行!' + '\n')
    RM_PATH = "..\Data" + "\\"
    path_list = []

    file_path = os.listdir(RM_PATH)
    for ifile in file_path:
        if ifile.startswith('RM'):
            file_name = ifile
            path_list.append(str(RM_PATH) + str(file_name))

    print('共有%s个文件需要处理' % str(len(path_list)) + '\n')
    # rm_cal = RM_Cal(path_list[0])
    # for index, value in enumerate(path_list):
    #     print('准备处理第%s个文件' % str(index+1)+'\n')
    #     # print(index)
    #     rm_cal = RM_cal(value)
    #     rm_cal.to_csv(1, 1, 1, 1, 1)

    mRC = muti_RM_Cal(path_list)
    mRC.cal_raw()
