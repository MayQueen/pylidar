# -*- encoding: utf-8 -*-
'''
@NAME      :main.py
@TIME      :2021/04/12 13:07:20
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
'''

import os
from RMFile import muti_RM_reader, RM_Calculate


class rm_processing():
    """
    多个RM文件读取和计算操作
    """

    def __init__(self, RM_PATH):
        self.path_list = []

        file_path = os.listdir(RM_PATH)
        for ifile in file_path:
            if ifile.startswith('RM'):
                file_name = ifile
                self.path_list.append(str(RM_PATH) + str(file_name))

    def reader(self):
        """
        说明：muti_RM_reader类的继承
        主要方法(详情见RM_reader)：
            save_header_info()
            save_channel_info()
            save_file_data()
        主要属性：
            muti_header_info;muti_channel_info;muti_file_data;
            num_files;num_channels;
            raw;height;rcs_list;
            adcbits;
            data_points;
            bin_width;
            shot_number;
            input_range;
            dataset_mode;
        """
        self.RM_reader = muti_RM_reader(self.path_list)  # RM文件读取
        return self.RM_reader

    def calculate(self):
        """
        说明：RM_Calculate类的继承
        主要方法(详情见RM_reader)：
            get_height();get_raw_data();get_rcs();
            save_channel_height();
            save_channel_raw();
            save_channel_rcs();
        主要属性：
            raw;height;rcs;
        """
        RM_cal = RM_Calculate(self.RM_reader.muti_header_info,
            self.RM_reader.muti_channel_info, self.RM_reader.muti_file_data)
        return RM_cal


if __name__ == "__main__":
    print('\n' + '程序开始运行!' + '\n')
    RM_PATH = "..\Data" + "\\"

    RM_main = rm_processing(RM_PATH)
    RM_reader = RM_main.reader()  # 读取文件
    RM_cal = RM_main.calculate()  # 开始计算

    # RM_reader
    # print(RM_reader.muti_header_info)
    # print(RM_reader.muti_channel_info)
    # print(RM_reader.muti_file_data)

    # RM_reader.save_header_info()
    # RM_reader.save_channel_info()s
    # RM_reader.save_file_data()

    # RM_Calculate
    # print(RM_cal.muti_channel_info)
    # print(RM_cal.input_range)

    RM_cal.to_csv('RCS') # ['Height','Raw','RCS']
    # RM_cal.get_raw_data()

    print('\n' + '### 程序运行完成!')
