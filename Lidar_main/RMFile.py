# -*- encoding: utf-8 -*-
"""
@NAME      :RMFile.py
@TIME      :2021/03/04 16:01:52
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
"""

import numpy as np
import pandas as pd


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
        self.Channel_header = ['Active', 'Analog_Ghoton', 'Laser_used', 'Number_of_datapoints', '1', 'HV',
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
        self.num_points = [self.channel_info[_idx2][3]
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
        self.bin_width = [self.channel_info[_idx4][6]
                          for _idx4 in range(len(self.channel_info))]

    # 获取wavelength
    def get_wavelength(self):
        """
        获取每个channel的波长和极化方式.o=无极化;s=垂直;p=平行
        """
        self.wavelength = [self.channel_info[_idx5][7]
                           for _idx5 in range(len(self.channel_info))]

    # 获取adcbits
    def get_adcbits(self):
        """
        获取文件的adcbits
        """
        self.adcbits = [self.channel_info[_idx6][-4]
                        for _idx6 in range(len(self.channel_info))]

    # 获取num_shots
    def get_num_shots(self):
        """
        获取文件的发射次数（number of shots）
        """
        self.num_shots = [self.channel_info[_idx7][-3]
                          for _idx7 in range(len(self.channel_info))]

    # 获取input_range
    def get_input_range(self):
        """
        获取文件的input range
        """
        self.input_range = [self.channel_info[_idx8][-2]
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
    可用的方法：save_header_info;save_info
    属性：muti_header_info;muti_channel_info;muti_file_data
    """

    def __init__(self, RM_PATH_LIST):
        self.muti_header_info = []  # 多个文件的头文件

        self.muti_channel_info = []  # 多个文件的通道
        self.muti_file_data = []  # 多个文件的数据

        self.path_list = RM_PATH_LIST  # 多个文件的路径
        self.num_files = len(self.path_list)
        print("该目录下共有%s" % self.num_files + '个RM文件' + '\n')

        for ipx in range(len(self.path_list)):
            # 继承RM_Reader类
            single_RM = RM_reader(self.path_list[ipx])

            print('# 准备读取第%s' % str(ipx + 1) + '个文件')
            single_RM.single_reader()  # 主要

            self.muti_header_info.append(single_RM.header_info)

            self.muti_channel_info.append(single_RM.channel_info)

            single_data = pd.DataFrame(
                single_RM.file_data, index=single_RM.dataset_mode).T
            # print(single_data)
            self.muti_file_data.append(single_data)
        # print(self.muti_file_data[0])

    def save_header_info(self):
        _csv_path = './Results/RMFiles/RM_File_header_info.csv'

        self.key_headerInfo = self.muti_header_info[0].keys()

        _DF_headerInfo = pd.DataFrame([self.muti_header_info[i].values(
        ) for i in range(self.num_files)], columns=self.key_headerInfo).T

        _DF_headerInfo.to_csv(_csv_path, header=False)
        # print(_DF)
        print('# RM_File_info文件保存完成')

    def save_channel_info(self):
        _csv_path = './Results/RMFiles/RM_File_channel_info.csv'
        # _Channel_header = ['active', 'analog_photon', 'laser_used', 'number_of_datapoints', '1', 'HV', 'bin_width', 'wavelength', 'd1', 'd2', 'd3', 'd4', 'ADCbits', 'number_of_shots', 'input_range', 'ID']
        _Channel_index = [self.muti_header_info[i]['Filename'] for i in range(self.num_files)]

        _DF_ChannInfo = pd.DataFrame(self.muti_channel_info, index=_Channel_index).T
        # print(_DF_ChannInfo)
        _DF_ChannInfo.to_csv(_csv_path, index=False)
        print('# RM_File_Channel_info文件保存完成')


class RM_Calculate():
    """
    说明：处理RM File 中每个频道(channel)的数据
    输入：
        muti_channel_info, muti_file_data
    """

    def __init__(self, muti_header_info, muti_channel_info, muti_file_data):
        self.muti_header_info = muti_header_info
        self.muti_channel_info = muti_channel_info

        self.num_files = len(self.muti_channel_info)
        self.num_channels = len(self.muti_channel_info[0])

        self.muti_file_data = np.array(muti_file_data)

        # self.height = []
        # self.raw = []
        # self.rcs = []
        self.start_time = []
        self.end_time = []

        self.adcbits = []  # "ADCbits"
        self.data_points = []  # number_of_datapoints
        self.bin_width = []  # bin_width, 单位:m
        self.shot_number = []  # "number_of_shots"
        self.input_range = []  # BC:input_range;BT:discriminator
        self.dataset_mode = []  # dataset_mode

        for idx in range(self.num_files):
            print('# 开始处理第%s' % str(idx + 1) + '个文件')
            for idc in range(self.num_channels):
                self.adcbits.append(
                    int(self.muti_channel_info[idx][idc][-4]))  # "ADCbits"
                # "number_of_datapoints
                self.data_points.append(
                    int(self.muti_channel_info[idx][idc][3]))
                self.bin_width.append(
                    float(self.muti_channel_info[idx][idc][6]))  # bin_width, 单位:m
                self.shot_number.append(
                    int(self.muti_channel_info[idx][idc][-3]))  # "number_of_shots"
                # BC:input_range;BT:discriminator
                self.input_range.append(
                    float(self.muti_channel_info[idx][idc][-2]))
                # BT:analogue dataset; BC:photon counting
                self.dataset_mode.append(
                    str(self.muti_channel_info[idx][idc][-1][:2]))
                self.start_time.append(
                    str(self.muti_header_info[idx]['Start_time']))

                self.end_time.append(
                    str(self.muti_header_info[idx]['End_time']))

    def get_height(self):
        """
        根据channel的信息计算height
        """

        print('# 开始计算Height')
        _height = []
        for ibin in range(len(self.bin_width)):
            print("Height:第%s次计算" % str(ibin+1))
            _height.append([self.bin_width[ibin] * num_bin for num_bin in range(self.data_points[ibin])])
        self.height = pd.DataFrame(_height)
        return self.height
        # print(self.height)
        # print(self.start_time)
        # print(self.end_time)

    def _get_mVolts(self, single_file, ichannel, num_p):
        _input_range = float(self.input_range[ichannel])
        _adcbits = int(self.adcbits[ichannel])
        _shot_number = int(self.shot_number[ichannel])
        _bin_width = float(self.bin_width[ichannel])

        _file_data = float(single_file.iloc[ichannel][num_p])

        _mVolts = (_file_data * _input_range * 1000) / \
                  (2 ** _adcbits * _shot_number)
        return _mVolts

    def get_raw_data(self):
        """
        根据channel的信息计算raw_data
        """
        self.get_height() # 计算Raw data 前必须先计算height

        _raw = []

        print('# 开始计算raw')
        for num_f in range(self.num_files):
            single_file = pd.DataFrame(self.muti_file_data[num_f], columns=[
                'BT', 'BC', 'BT', 'BC', 'BT', 'BC', 'PD']).T
            # print(single_file[0])
            for ichannel in range(self.num_channels):
                print("Raw Data:第%s次计算" % str(ichannel+1))
                _input_range = float(self.input_range[ichannel])
                _adcbits = int(self.adcbits[ichannel])
                _shot_number = int(self.shot_number[ichannel])
                _bin_width = float(self.bin_width[ichannel])
                _num_points = int(self.data_points[ichannel])
                # _num_points = 10

                # 根据dataset_mode类型,选择计算公式
                if single_file.iloc[ichannel].name == 'BT':  # 模拟信号
                    _signal_mVolts = []
                    for num_p in range(_num_points):
                        # print (single_file.iloc[ichannel][num_p])
                        _mVolts = self._get_mVolts(
                            single_file, ichannel, num_p)
                        _signal_mVolts.append(_mVolts)
                    # print(signal_mVolts)
                    _raw.append(_signal_mVolts)

                elif single_file.iloc[ichannel].name == 'BC':  # 光信号
                    _signal_MHz = []
                    for num_p in range(_num_points):
                        # print (single_file.iloc[ichannel][num_p])

                        _file_data = float(single_file.iloc[ichannel][num_p])
                        _MHz = (_file_data * (150 / _bin_width)) / _shot_number

                        _signal_MHz.append(_MHz)
                    # print(signal_MHz)
                    _raw.append(_signal_MHz)

                else:  # 混合信号
                    _signal_mix = []
                    for num_p in range(_num_points):
                        # _file_data = float(single_file.iloc[ichannel][num_p])
                        if num_p < int(self.shot_number[0]):  # 混合信号分界线
                            # print(num_p)
                            _mix_mv = self._get_mVolts(
                                single_file, ichannel, num_p)
                            _signal_mix.append(_mix_mv)
                        else:
                            _file_data = float(
                                single_file.iloc[ichannel][num_p])
                            _mix_MHz = (_file_data * (150 / _bin_width)) / _shot_number

                            _signal_MHz.append(_mix_MHz)
                    _raw.append(_signal_mix)

        self.raw = pd.DataFrame(_raw)
        del _raw, _signal_mVolts, _signal_MHz, _signal_mix  # 删除变量
        return self.raw
        # print(pd.DataFrame(self.raw))
        print('# raw计算完成')

    def get_rcs(self):
        """
        计算范围矫正信号(RCS,range corrected signal)
        """
        self.get_raw_data() # 计算rcs前必须先计算Raw
        _rcs = []
        print('# 开始计算RCS')

        for ichannel in range(self.num_channels * self.num_files):
            print("RCS :第%s次计算" % str(ichannel))
            # STEP 1:Overlap Corrected signal 去除低空重叠信号
            # OCS = self.raw.loc[ichannel][29:1009]
            OCS = self.raw.loc[ichannel][20:1001]

            h = self.height.loc[ichannel][20:1001]
            # h = self.height.loc[ichannel][21:1001]

            # STEP 2:Background Corrected signal 背景修正
            background_error = 5.04
            BCS = np.array((OCS - background_error))  # 去除噪声信号

            # STEP 3:Range Corrected Signal 计算距离修正信号
            h_2 = np.array(np.square(h))     # height^2
            RCS = np.multiply(BCS, h_2)
            # RCS = np.where(_RCS > 0, _RCS, 0.000001)  # 避免去除噪声信号之后出现负值

            _rcs.append(RCS)
        self.rcs = pd.DataFrame(_rcs)
        return self.rcs
        print("RCS:计算完成!")
        # print(self.rcs)

    def to_csv(self,data):
        def _save(self,data, channel_data):
            print('# 准备[%s]数据' % str(data))
            for num_c in range(0, self.num_files * self.num_channels, self.num_channels):
                _data = channel_data[num_c:num_c + self.num_channels]
                # print(_height)
                _data.T.to_csv('./Results/RMFiles/File %s' % str(num_c) +  '%s.csv' % str(data),
                                    float_format='%.6f')
            print('# 保存完成!')

        if data == "Height":
            self.get_height()
            channel_data = self.height
            _save(self, data=data, channel_data=channel_data)
        elif data == "Raw":
            self.get_raw_data()
            channel_data = self.raw
            _save(self, data=data, channel_data=channel_data)
        else:
            self.get_rcs()
            channel_data = self.rcs
            _save(self, data=data, channel_data=channel_data)
