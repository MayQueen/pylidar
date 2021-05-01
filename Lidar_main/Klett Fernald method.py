# -*- encoding: utf-8 -*-
'''
@NAME      :Klett Fernald method.py
@TIME      :2021/03/25 11:10:28
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
'''
# Inbuild package list
from math import log, exp, pi

# Site-package list
from scipy import integrate
import numpy as np
import pandas as pd


# CODING CONTENT
class Klett():
    """
    alpha: 消光系数 extintion coefficient
    beta : 后向散射系数 backscatter coefficient
    """

    def __init__(self, height, rcs, bin_width=7.5, data_points=16221, k=1, ref=-10):
        self.height = height
        self.rcs = rcs

        self.alpha = []
        self.beta = []

        self.k = k
        self.LR_z = 50

        self.bin_width = bin_width  # 单位:m
        self.data_points = data_points  # number_of_datapoints
        self.ref = ref

    def alpha_z(self):
        """
        根据Klett雷达反演方法求解激光雷达消光系数alpha_z
        """
        print("开始计算消光系数alpha")
        # 根据Collis斜率法计算参考高度下的 大气消光系数
        s_0 = log(self.rcs[0])
        s_m = log(self.rcs[self.ref])
        alpha_ref = 0.5 * (s_0 - s_m) / (self.height[self.ref] - self.height[0])
        # print(alpha_ref)

        for h in range(len(self.height) + self.ref):  # 仅能计算参考高度以下的数值
            # for h in range(10):
            s = log(self.rcs[h])
            s_m = log(self.rcs[self.ref])
            # print(exp((s - s_m)/self.k))
            exp_s = (s - s_m) / self.k

            def fun_rcs(x):
                return exp_s

            interg_s, err = integrate.quad(fun_rcs, h, self.ref)
            # print(intergra_s)

            # 根据Klett方法计算大气消光系数
            var_alpha_z = interg_s / ((1 / alpha_ref) - (2 / self.k) * exp_s)

            # 保存计算结果
            self.alpha.append(var_alpha_z)
        print("计算完成")
        return self.alpha
        # print(self.alpha)

    def beta_z(self):
        """
        # 计算后向散射系数beta_z
        """
        self.alpha_z()
        print("开始计算后向散射系数beta")
        for index, alpha in enumerate(self.alpha):
            _beta = self.LR_z * alpha ** self.k
            self.beta.append(_beta)
            del _beta
        print("计算完成")
        return self.beta
        # print(len(self.beta))

    def to_csv(self):
        self.beta_z()
        pd_height = pd.DataFrame(self.height[:self.ref])
        pd_alpha = pd.DataFrame(self.alpha)
        pd_beta = pd.DataFrame(self.beta)

        pd_data = pd.concat([pd_height, pd_alpha, pd_beta], axis=1)  # 合并结果
        pd_data.columns = ["Height", "Alpha", "Beta"]  # 设置文件列名
        pd_data.to_csv("Results/Coefficient/Klett.csv",index=None)  # 保存Klett反演结果
        print("Klett反演结果保存成功")


class Fernald():
    """
    使用Fernald方法求解雷达方程
    STEP1:计算分子后向散射系数[beta_mol]
    STEP2:计算总后向散射系数[beta_total]
    STEP3:计算气溶胶后向散射系数[beta_aer]
    STEP4:计算气溶胶消光系数[alpha_aer]
    """
    def __init__(self, season, longitude, wavelength, height, rcs=[]):
        self.season = season  # string
        self.longitude = longitude  # float
        self.wavelength = wavelength  # int  [nm]
        self.height = height  # list
        self.rcs = rcs  # list
        self.LR_mol = (8 * pi) / 3
        self.LR_aer = 60
        self.ref = -10
        self.c = 1

    def get_temp(self):
        """
        温度[K]随高度[km]的变化函数
        """
        _temp = []

        for index, h in enumerate(self.height):
            h = h / 1000  # 单位[m]转化为[km]
            if self.season == "Summer":  # 夏季[4-9]
                # print("采用[夏]时令[温度]计算方案")

                # 低纬度 [小于22°]
                if self.longitude < 22:
                    print("采用[低纬度]计算方案")

                # 中纬度[22~45°]
                elif 22 <= self.longitude <= 45:
                    # print("采用[中纬度]计算方案")
                    if 0 <= float(h) < 13:  # 单位:m
                        _t = 294.9838 - 5.2159 * h - 0.07109 * pow(h, 2)
                    elif 13 <= h < 17:
                        _t = 215.5
                    elif 17 <= h < 47:
                        _t = 215.5 * exp((h - 17) * 0.008128)
                    elif 47 <= h < 53:
                        _t = 275
                    elif 53 <= h < 80:
                        _t = 275 + (1 - exp((h - 53) * 0.06)) * 20
                    elif 80 <= h < 100:
                        _t = 175
                    else:
                        _t = -999
                        # print("无法计算,因为高度大于100km")

                # 高纬度[大于45°]
                else:
                    print("采用[高纬度]计算方案")

            else:  # 冬季[10-12,1-3]
                # print("采用[冬]时令[温度]计算方案")

                # 低纬度 [小于22°]
                if self.longitude < 22:
                    print("采用[低纬度]计算方案")

                # 中纬度[22~45°]
                elif 22 <= self.longitude <= 45:
                    # print("采用[中纬度]计算方案")
                    if 0 <= h < 10:
                        _t = 272.7241 - 3.6217 * h - 0.1759 * h ** 2
                    elif 10 <= h < 33:
                        _t = 218
                    elif 33 <= h < 47:
                        _t = 218 + (h - 33) * 3.3571
                    elif 47 <= h < 53:
                        _t = 265
                    elif 53 <= h < 80:
                        _t = 265 - (h - 53) * 2.0370
                    elif 80 <= h < 100:
                        _t = 210
                    else:
                        _t = -999
                        # print("无法计算,因为高度大于100km")

                # 高纬度[大于45°]
                else:
                    print("采用[高纬度]计算方案")

            _temp.append(_t)
        self.temp = pd.DataFrame(_temp)

        return self.temp
        del _temp, _t
        # print(self.temp)

    def get_press(self):
        """
        压力[hPa]随高度[km]的变化函数
        """
        _press = []
        for index, h in enumerate(self.height):
            h = h / 1000  # 单位[m]转化为[km]
            if self.season == "Summer":  # 夏季[4-9]
                # print("采用[夏]时令[压力]计算方案")

                # 低纬度[小于22°]
                if self.longitude < 22:
                    print("采用[低纬度]计算方案")

                # 中纬度[22~45°]
                elif 22 <= self.longitude <= 45:
                    # print("采用[中纬度]计算方案")
                    if 0 <= h < 10:
                        _p = 1012.8186 - 111.5569 * h + 3.8646 * h ** 2
                    elif 10 <= h < 72:
                        # p_10 = 1012.8186-111.5569*10+3.8646*10**2
                        p_10: float = 283.7096
                        _p = p_10 * exp(-0.147 * (h - 10))
                    elif 72 <= h < 100:
                        # p_72 = p_10*exp(-0.147*(72-10))
                        p_72: float = 0.03124
                        _p = p_72 * exp(-0.165 * (h - 72))
                    else:
                        _p = -999
                        # print("无法计算,因为高度大于100km")
                # 高纬度 [大于45°]
                else:
                    print("采用[高纬度]计算方案")
            else:  # 冬季[10-12,1-3]
                # print("采用[冬]时令[压力]计算方案")

                # 低纬度[小于22°]
                if self.longitude < 22:
                    print("采用[低纬度]计算方案")

                # 中纬度[22~45°]
                elif 22 <= self.longitude <= 45:
                    # print("采用[中纬度]计算方案")
                    if 0 <= h < 10:
                        _p = 1018.8627 - 124.2954 * h + 4.8307 * h ** 2
                    elif 10 <= h < 72:
                        # p_10 = 1018.8627-124.2954*10+4.8307*10**2
                        p_10: float = 258.9787
                        _p = p_10 * exp(-0.147 * (h - 10))
                    elif 72 <= h < 100:
                        # p_72 = p_10*exp(-0.147*(72-10))
                        p_72: float = 0.028517
                        _p = p_72 * exp(-0.155 * (h - 72))
                    else:
                        _p = -999
                        # print("无法计算,因为高度大于100km")

                # 高纬度[大于45°]
                else:
                    print("采用[高纬度]计算方案")
            _press.append(_p)
        self.press = pd.DataFrame(_press)
        return self.press
        del _press, _p
        # print(self.press)

    def get_beta_mol(self):
        """
        计算大气分子后向散射系数beta_mol
        """
        self.get_temp()
        self.get_press()

        print("开始计算[分子后向散射系数]")

        numbda = str(self.wavelength)
        t0 = 25.7  # [℃]
        p0 = 1006.6  # [hPa]
        n_s = {"355": 1.000285745, "387": 1.000283480, "532": 1.000278235, "1064": 1.000273943}
        c_numb = {"355": 3.62854e-08, "387": 3.57125e-08, "532": 3.44032e-08, "1064": 3.33502e-08}
        Fk = {"355": 1.05289, "387": 1.05166, "532": 1.04899, "1064": 1.04721}

        _beta_mol = []
        for index, value in enumerate(self.temp):
            n_r = n_s[numbda] * (t0 / p0) * (self.press[index] / self.temp[index])
            c1 = 9 * pow(pi, 2)
            c2 = pow(self.wavelength, 4) * pow(n_s[numbda], 2)
            _betamol = n_r * (c1 / c2) * c_numb[numbda] * Fk[numbda]
            _beta_mol.append(_betamol)
        self.beta_mol = pd.DataFrame(_beta_mol).T

        return self.beta_mol
        del _beta_mol, _betamol
        # print(self.beta_mol)

    def get_alpha_mol(self):
        """
        大气分子消光系数alpha_mol
        """
        _alpha_mol = []
        self.get_beta_mol()
        print("开始计算[分子消光系数]")
        # print(self.beta_mol)
        for index, value in enumerate(self.beta_mol[0].tolist()):
            # print(value)
            _alphamol = self.LR_mol * value
            _alpha_mol.append(_alphamol)
        self.alpha_mol = pd.DataFrame(_alpha_mol)

        return self.alpha_mol
        del _alpha_mol, _alphamol
        # print(self.alpha_mol)

    def get_beta_total(self):
        """
        计算总后向散射系数Beta_total = Beta_mol + Beta_aer
        """
        self.get_beta_mol()
        print('开始计算[总后向散射系数]')
        _beta_total = []

        _rcs_ref = self.rcs[self.ref]
        _beta_ref = self.beta_mol.loc[len(self.beta_mol) + self.ref]

        for index, value in enumerate(self.height[:self.ref]):
            _beta_mol = self.beta_mol.loc[index]

            def fun_betamol(x):
                return _beta_mol

            v_betamol, err = integrate.quad(fun_betamol, self.height[index], self.height[self.ref])
            c1 = self.rcs[index] * exp(2 * (self.LR_aer - self.LR_mol)) * v_betamol

            def fun_rcs(x):
                return c1

            v_rcs, err = integrate.quad(fun_rcs, self.height[index], self.height[self.ref])

            c2 = _rcs_ref / (self.c * _beta_ref)
            c3 = 2 * self.LR_aer * v_rcs

            _betatotal = c1 / (c2 + c3)
            _beta_total.append(_betatotal)

        self.beta_total = pd.DataFrame(_beta_total, index=[index for index in range(len(_beta_total))])

        return self.beta_total
        # print(self.beta_total)

    def get_beta_aer(self):
        self.get_beta_total()
        print("开始计算[气溶胶后向散射系数]")
        self.beta_aer = (self.beta_total - self.beta_mol)[:self.ref]

        return self.beta_aer
        # print(self.beta_aer)

    def get_alpha_aer(self):
        self.get_beta_aer()
        print("开始计算[气溶胶消光系数]")
        self.alpha_aer = self.beta_aer * self.LR_aer
        return self.alpha_aer
        # print(self.alpha_aer)

    def to_csv(self):
        self.get_alpha_mol()
        self.get_alpha_aer()

        pd_height = pd.DataFrame(self.height[:self.ref])

        pd_beta_mol = self.beta_mol[:self.ref]
        pd_beta_aer = self.beta_aer
        pd_beta_total = self.beta_total

        pd_alpha_aer = self.alpha_aer
        pd_alpha_mol = self.alpha_mol[:self.ref]

        pd_data = pd.concat([pd_height, pd_beta_mol, pd_beta_aer, pd_beta_total, pd_alpha_aer, pd_alpha_mol],
                            axis=1)  # 合并结果
        pd_data.columns = ["Height", "beta_mol", "beta_aer", "beta_total", "alpha_aer", "alpha_mol"]  # 设置文件列名
        pd_data.to_csv("Results/Coefficient/Fernald.csv", index=None)  # 保存Fernald反演结果
        print("Fernald反演结果保存成功")


if __name__ == "__main__":
    print('开始读取RCS数据')
    data_path = "Results/plotdata/Fernald.xlsx"
    h = pd.read_excel(data_path, sheet_name="H", header=0, index_col=0)
    rcs = pd.read_excel(data_path, sheet_name="RCS", header=0, index_col=0)

    h_list = h[0].tolist()
    rcs_list = rcs[0].tolist()
    print('使用Klett方法反演雷达方程')

    RM = Klett(height=h_list, rcs=rcs_list)
    RM.to_csv()

    # print('使用Fernald方法反演雷达方程')
    # Fernald = Fernald(wavelength=355, season='Summer', longitude=22.02, height=h_list, rcs=rcs_list)
    # Fernald.to_csv()
    print('程序运行结束')
