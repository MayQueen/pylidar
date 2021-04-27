# -*- encoding: utf-8 -*-
'''
@NAME      :bata_mol.py
@TIME      :2021/03/24 14:44:49
@AUTHOR     :MERI
@Email     :mayqueen2016@163.com
'''
# Inbuild package list
from math import exp

# Site-package list
from scipy import integrate  # 积分y运算
from scipy.constants import pi  # 圆周率pi

# Userdefined package list


# CODING CONTENT
def f(x):
    return x + 1
# v, err = integrate.quad(f, 1, 2)
# print(v)


def bata_mol(r):
    '''
    大气分子后向散射系数与高度r的函数关系式
    '''
    c1 = 2.3850e-8
    c2 = 1012.8186-111.5569*r+3.8645*(r**2)
    c3 = 294.9838-5.2159*r-0.0711*(r**2)
    bata_mol = c1*(c2/c3)
    return bata_mol


def alpha_mol(r, LR_mol):
    '''
    大气分子消光系数与高度r的函数关系式
    '''
    c4 = LR_mol
    bata_mol = bata_mol(r)
    alpha_mol = c4 * bata_mol
    return alpha_mol


def inter_bata_mol(r, r_ref):
    '''
    求解bata_mol在[r,ref]区间的定积分
    '''
    def _bata_mol(r):
        '''
        大气分子后向散射系数与高度r的函数关系式
        '''
        c1 = 2.3850e-8
        c2 = 1012.8186-111.5569*r+3.8645*(r**2)
        c3 = 294.9838-5.2159*r-0.0711*(r**2)
        return c1*(c2/c3)
    v, err = integrate.quad(bata_mol, r, r_ref)
    print(v)


def bata_aer(r):
    '''
    计算气溶胶后向散射系数
    '''
    pass


def bata_r(rcs, r, r_ref=3, LR_aer=40, c=1):
    '''
    计算总后向散射系数
    rcs为list类型
    '''
    # 激光雷达比
    LR_mol = 8*pi / 3
    rcs_ref = rcs[r_ref]
    # print(rcs_ref)

    # 计算参考高度ref处的后向散射系数bata_mol_ref
    def bata_mol(r):
        '''
        大气分子后向散射系数与高度r的函数关系式
        '''
        c1 = 2.3850e-8
        c2 = 1012.8186-111.5569*r+3.8645*(r**2)
        c3 = 294.9838-5.2159*r-0.0711*(r**2)
        return c1*(c2/c3)

    bata_mol_ref = bata_mol(r_ref)

    c4,err = integrate.quad(bata_mol,r,r_ref)

    # bata_r计算公式的分子部分
    c5 = rcs[r]*exp(2*(LR_aer - LR_mol)*c4)


    # bata_r计算公式的分母部分

    c6 = rcs_ref / (c * bata_mol_ref)
    # print(rcs_ref,c4,c5,c6)

    c7 = rcs[r]*exp(2*(LR_aer - LR_mol)*c4)  
    print(c7)
    
    # c7, err = integrate.quad(c6, r, r_ref)

    # bata_r = c5 / (c6 + 2*LR_aer*c7)

    # return bata_r

rcs = [5.241926131,3.26489019,3.795168048,2.361559713,2.075695638,1.972212843]
bata_r(rcs,r=1)