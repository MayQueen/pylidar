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

    def __init__(self, height, rcs, bin_width=7.5, data_points=16221, k = 1, ref=-10):
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
        '''
        根据Klett雷达反演方法求解激光雷达消光系数alpha_z
        '''
        print("开始计算消光系数alpha")
        # 根据Collis斜率法计算参考高度下的 大气消光系数
        s_0 = log(self.rcs[0])
        s_m = log(self.rcs[self.ref])
        alpha_ref = 0.5 * (s_0 - s_m) / (self.height[self.ref] - self.height[0])
        # print(alpha_ref)

        for h in range(len(self.height)+self.ref): # 仅能计算参考高度以下的数值
        # for h in range(10):
            s = log(self.rcs[h])
            s_m = log(self.rcs[self.ref])
            # print(exp((s - s_m)/self.k))
            exp_s = (s - s_m) / self.k

            def fun_rcs(x):
                return exp_s

            interg_s,err = integrate.quad(fun_rcs, h, self.ref)
            # print(intergra_s)

            # 根据Klett方法计算大气消光系数
            var_alpha_z = interg_s / ((1/alpha_ref) - (2/self.k)*exp_s)

            # 保存计算结果
            self.alpha.append(var_alpha_z)
        print("计算完成")
        return self.alpha
        # print(self.alpha)

    def beta_z(self):
        '''
        # 计算后向散射系数beta_z
        '''
        self.alpha_z()
        print("开始计算后向散射系数beta")
        for index,alpha in enumerate (self.alpha):
            _beta = self.LR_z * alpha ** self.k
            self.beta.append(_beta)
            del  _beta
        print("计算完成")
        return self.beta
        # print(len(self.beta))

    def to_csv(self):
        self.beta_z()
        pd_height = pd.DataFrame(self.height[:self.ref])
        pd_alpha = pd.DataFrame(self.alpha)
        pd_beta = pd.DataFrame(self.beta)

        pd_data= pd.concat([pd_height,pd_alpha,pd_beta],axis=1) # 合并结果
        pd_data.columns = ["Height","Alpha","Beta"] # 设置文件列名
        pd_data.to_csv("Results/Coefficient/Klett.csv") # 保存Klett反演结果
        print("Klett反演结果保存成功")


class Fernald():
    def __init__(self):
        pass

    def temp(self,season, longitude,h):
        """
        温度[K]随高度[km]的变化函数
        """
        if season =="Summer":# 夏季[4-9]
            print("采用[夏]时令[温度]计算方案")

            # 低纬度 [小于22°]
            if longitude < 22:
                print("采用[低纬度]计算方案")

            # 中纬度[22~45°]
            elif longitude >= 22 and longitude <= 45:
                print("采用[中纬度]计算方案")
                if h >= 0 and h < 13:
                    return 294.9838 - 5.2159 * h - 0.07109 * h ** 2
                elif h >= 13 and h < 17:
                    return 215.5
                elif h >= 17 and h < 47:
                    return 215.5 * exp((h - 17) * 0.008128)
                elif h >= 47 and h < 53:
                    return 275
                elif h >= 53 and h < 80:
                    return 275 + (1 - exp((h - 53) * 0.06)) * 20
                elif h >= 80 and h < 100:
                    return 175
                else:
                    print("无法计算,因为高度大于100km")

            # 高纬度[大于45°]
            else:
                print("采用[高纬度]计算方案")
        else:# 冬季[10-12,1-3]
            print("采用[冬]时令[温度]计算方案")

            # 低纬度 [小于22°]
            if longitude < 22:
                print("采用[低纬度]计算方案")

            # 中纬度[22~45°]
            elif longitude >= 22 and longitude <= 45:
                print("采用[中纬度]计算方案")
                if h >= 0 and h < 10:
                    return 272.7241 - 3.6217 * h - 0.1759 * h ** 2
                elif h >= 10 and h < 33:
                    return 218
                elif h >= 33 and h < 47:
                    return 218 + (h - 33) * 3.3571
                elif h >= 47 and h < 53:
                    return 265
                elif h >= 53 and h < 80:
                    return 265 - (h - 53) * 2.0370
                elif h >= 80 and h < 100:
                    return 210
                else:
                    print("无法计算,因为高度大于100km")

            # 高纬度[大于45°]
            else:
                print("采用[高纬度]计算方案")

    def press(self,season,longitude,h):
        """
        压力[hPa]随高度[km]的变化函数
        """
        if season == "Summer": # 夏季[4-9]
            print("采用[夏]时令[压力]计算方案")
            # 低纬度[小于22°]
            if longitude < 22:
                print("采用[低纬度]计算方案")
            # 中纬度[22~45°]
            elif longitude >= 22 and longitude <= 45:
                print("采用[中纬度]计算方案")
                if h >= 0 and h < 10:
                    return 1012.8186 - 111.5569 * h + 3.8646 * h ** 2
                elif h >= 10 and h < 72:
                    # p_10 = 1012.8186-111.5569*10+3.8646*10**2
                    p_10: float = 283.7096
                    return p_10 * exp(-0.147 * (h - 10))
                elif h >= 72 and h < 100:
                    # p_72 = p_10*exp(-0.147*(72-10))
                    p_72: float = 0.03124
                    return p_72 * exp(-0.165 * (h - 72))
                else:
                    print("无法计算,因为高度大于100km")
            # 高纬度 [大于45°]
            else:
                print("采用[高纬度]计算方案")
        else: # 冬季[10-12,1-3]
            print("采用[冬]时令[压力]计算方案")

            # 低纬度[小于22°]
            if longitude < 22:
                print("采用[低纬度]计算方案")

            # 中纬度[22~45°]
            elif longitude >= 22 and longitude <= 45:
                print("采用[中纬度]计算方案")
                if h >= 0 and h < 10:
                    return 1018.8627 - 124.2954 * h + 4.8307 * h ** 2
                elif h >= 10 and h < 72:
                    # p_10 = 1018.8627-124.2954*10+4.8307*10**2
                    p_10: float = 258.9787
                    return p_10 * exp(-0.147 * (h - 10))
                elif h >= 72 and h < 100:
                    # p_72 = p_10*exp(-0.147*(72-10))
                    p_72: float = 0.028517
                    return p_72 * exp(-0.155 * (h - 72))
                else:
                    print("无法计算,因为高度大于100km")

            # 高纬度[大于45°]
            else:
                print("采用[高纬度]计算方案")

    def beta_mol(self,wavelength,season,longitude,h):
        # wavelength = 355 # [nm]
        numbda = str(wavelength)
        t0 = 25.7 # [℃]
        p0 = 1006.6 # [hPa]
        n_s = {"355":1.000285745,"387":1.000283480,"532":1.000278235,"1064":1.000273943}
        c_numb = {"355":3.62854e-08,"387":3.57125e-08,"532":3.44032e-08,"1064":3.33502e-08}
        Fk= {"355":1.05289,"387":1.05166,"532":1.04899,"1064":1.04721}

        n_r = n_s[numbda]*(t0/p0)*(self.press(season,longitude,h)/self.temp(season,longitude,h))

        c1 = 9*pow(pi,2)
        c2 = pow(wavelength,4)*pow(n_s[numbda],2)
        beta_mol = n_r*(c1/c2)*c_numb[numbda]*Fk[numbda]
        return beta_mol
        # print(beta_mol)

    def alpha_mol(self, r, LR_mol):
        '''
        大气分子消光系数与高度r的函数关系式
        '''
        c4 = LR_mol
        beta_mol = self.beta_mol(r)
        alpha_mol = c4 * beta_mol
        return alpha_mol

    def inter_beta_mol(self, r, r_ref):
        '''
        求解beta_mol在[r,ref]区间的定积分
        '''
        v, err = integrate.quad(self.beta_mol, r, r_ref)
        print(v)

    def beta_aer(self, r):
        '''
        计算气溶胶后向散射系数
        '''
        pass

    def beta_r(self, rcs, r, r_ref=3, LR_aer=40, c=1):
        '''
        计算总后向散射系数
        rcs为list类型
        '''
        # 激光雷达比
        LR_mol = 8 * pi / 3
        rcs_ref = rcs[r_ref]

        # print(rcs_ref)

        # 计算参考高度ref处的后向散射系数beta_mol_ref
        def beta_mol(r):
            '''
            大气分子后向散射系数与高度r的函数关系式
            '''
            c1 = 2.3850e-8
            c2 = 1012.8186 - 111.5569 * r + 3.8645 * (r ** 2)
            c3 = 294.9838 - 5.2159 * r - 0.0711 * (r ** 2)
            return c1 * (c2 / c3)

        beta_mol_ref = beta_mol(r_ref)

        c4, err = integrate.quad(beta_mol, r, r_ref)

        # beta_r计算公式的分子部分
        c5 = rcs[r] * exp(2 * (LR_aer - LR_mol) * c4)

        # beta_r计算公式的分母部分

        c6 = rcs_ref / (c * beta_mol_ref)
        # print(rcs_ref,c4,c5,c6)

        c7 = rcs[r] * exp(2 * (LR_aer - LR_mol) * c4)
        print(c7)

        # c7, err = integrate.quad(c6, r, r_ref)

        # beta_r = c5 / (c6 + 2*LR_aer*c7)

        # return beta_r


if __name__ == "__main__":
    print('开始读取RCS数据')
    data_path = "Results/plotdata/Klett.xlsx"
    h = np.array(pd.read_excel(data_path, sheet_name="H", header=0, index_col=0))
    rcs = np.array(pd.read_excel(data_path, sheet_name="RCS", header=0, index_col=0))
    print('使用Kletthod方法反演雷达方程')
    RM = Klett(height=h, rcs=rcs)
    # alpha = RM.alpha_z()
    # alpha = RM.beta_z()
    # alpha = RM.to_csv()
    Fernald = Fernald()
    # p = Fernald.press(season='Summer', longitude=22.02, h=8)
    # t = Fernald.temp(season='Summer', longitude=22.02, h=8)
    # print(t)
    Fernald.beta_mol(355,"Summer",22.02,8)
    print('程序运行结束')
