import os
import datetime
import pandas as pd

from lidarSRC.LicelFileIO import read_batch_files,gen_dataframe,save_all_result
from lidarSRC.LidarProcess import fernaldAlgorithm,calcMolecularBeta,overlopCorrect,noiseCorrect,disCorrect
from lidarSRC.Inversion import calcPBL,calcCloud,calcAOD,calcVIS
from lidarSRC.plotLidar import plotLidar


# /*-----------------------定制函数----------------------------*\
def main_process(fn_dir:str,chose_date:str=None,
                                lidar_ratio:int=30,boundary_height:int=6,
                                vertical_resolution:float = 7.5,
                                bg_noise_range:tuple=(30, 40), pblhMaxH:int=1000,
                                pblhLowerH:int=80)->list:
    """
    定制函數，相应参数可根据不同设备参数修改。

    Args:
        fn_dir:RMFile数据文件定级目录
        chose_date:选定日期
        lidar_ratio:雷达比
        boundary_height:截取数据高度
        bg_noise_range:背景噪声范围
        pblhMaxH:最大边界层高度
        pblhLowerH:最小边界层高度

    Returns:
        mback_cor_data: 背景修正信号
        mdr_data:退偏比
        mtotal_raw:修正后的回波信号
        mpbl: 边界层高度
        mcloud: 云相关参数
        mbeta_aero: 气溶胶后项散射系数
        mextinc_aero: 气溶胶消光系数
        maod: 气溶胶光学厚度
        mvis: 垂直能见度
    """
    tt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 检查文件并读取
    if chose_date is None:
        time_batch, data_batch = read_batch_files(fileDir=fn_dir, filter_str='RM')  # 獲取日期命名文件夾內的以'RM'的文件，
        # del height_batch
    else:
        fp = os.path.join(fn_dir, chose_date)
        time_batch, data_batch = read_batch_files(fileDir=fp, filter_str='RM')

    # 通道数据处理
    p_data = gen_dataframe((data_batch['BT0'][:, :]), time_batch, ver_resolution=vertical_resolution)
    s_data = gen_dataframe((data_batch['BT1'][:, :]), time_batch, ver_resolution=vertical_resolution)

    # 根据日期拼接通道数据
    file_d = os.path.split(fn_dir)[(-1)]

    if '20210101' <= file_d <= '20210403':
        factor = 0.68
    else:
        factor = 0.965
    print("[%s]" % tt, "\t", "[RAW] Calculating process")
    mtotal_raw = factor * p_data + s_data # pd.Dataframe
    # print(mtotal_raw.index)

    # 反演pbl高度
    print("[%s]" % tt, "\t", "[PBL] Calculating process")
    mpbl = mtotal_raw.apply(calcPBL, args=(pblhMaxH, pblhLowerH))

    # 构造pbl数据结构以便保存
    mpbl.index = ['PBL', 'hIndex']
    mpbl.columns=mtotal_raw.columns
    # print(mpbl)

    # 反演云相关参数
    print("[%s]" % tt, "\t", "[Cloud Height] Calculating process")
    _mch = calcCloud(mtotal_raw, min_level_height=None, start_level_height=None)   # np.array
    mcloud = pd.DataFrame(_mch,columns=mtotal_raw.columns,index=["CloudBase", "hIndexCloudBase","CloudTop","hIndexCloudTop"])

    # 计算退偏比参数
    print("[%s]" % tt, "\t", "[Depolarization Ratio] Calculating process")
    mdr_data = s_data / p_data / factor
    # print(type(mdr_data))
    
    # 计算分子后向散射系数
    print("[%s]" % tt, "\t", "[Beta Mole] Calculating process")
    _beta_mol = calcMolecularBeta((mtotal_raw.index * 0.001), lamda_length=532)
    # print(type(beta_mol))     # pandas.core.indexes.numeric.Float64Index

    print("[%s]" % tt, "\t", "[Singal Correcting] process")
    # 几何因子修正
    # ol_data = overlopCorrect(olData=None,lidarData=mtotal_raw)     #   oldata未知，因而未能進行幾何因子修正
    _ol_correct_data = mtotal_raw

    # 背景噪声修正
    mback_cor_data = noiseCorrect(_ol_correct_data,
                                                  height_resolution=(vertical_resolution / 1000.0),
                                                  noise_altitude=(bg_noise_range[0], bg_noise_range[1]))
    # print(mback_cor_data[mback_cor_data.columns[0]])   # type ->DataFrame

    # 数据平滑
    # smooth_corr_data = np.apply_along_axis(smooth, 0, _mback_cor_data, window_len=30)
    # print(smooth_corr_data)  # type -> numpy.ndarray
    _smooth_correct_data = mback_cor_data

    # 距离修正
    _dis_correct_data = disCorrect(_smooth_correct_data)  # 输入：type -> dataframe 或 series
    # print(dis_correct_data)


    # 反演气溶胶后向散射系数
    print("[%s]" % tt, "\t", "[Beta Aerosol] Calculating process")
    # 使用fernald算法求解雷达方程，输入dis_correct_data和bete_mol，得到mbeta_aero
    mbeta_aero = fernaldAlgorithm(_dis_correct_data,
                                    s_aer=lidar_ratio,
                                    height_resolution=(vertical_resolution / 1000.0),
                                    beta_mol=_beta_mol,
                                    inver_height=boundary_height)
    mbeta_aero[mbeta_aero < 0] = 0
    # print(mbeta_aero)

    # 计算气溶胶消光系数
    print("[%s]" % tt, "\t", "[Extinc Aerosol] Calculating process")
    mextinc_aero = mbeta_aero * lidar_ratio

    # 构造数据结构以便保存
    high_index = int(1000*boundary_height/vertical_resolution)      # 地面至指定高度範圍
    mbeta_aero = pd.DataFrame(mbeta_aero, index=mtotal_raw.index[:high_index], columns=mtotal_raw.columns)
    mextinc_aero = pd.DataFrame(mextinc_aero, index=mtotal_raw.index[:high_index], columns=mtotal_raw.columns)
    
    # 反演气溶胶光学厚度
    print("[%s]" % tt, "\t", "[AOD] Calculating process")
    maod = calcAOD(mextinc_aero)
    maod = pd.DataFrame(maod,columns=['AOD'],index=mtotal_raw.columns).T

    # 反演垂直能见度
    print("[%s]" % tt, "\t", "[VIS] Calculating process")
    mvis = calcVIS(mextinc_aero)
    mvis = pd.DataFrame(mvis,columns=['VIS'],index=mtotal_raw.columns).T
    print("[%s]" % tt, "\t", "Processing Complete!")
    
    return mback_cor_data, mdr_data, mtotal_raw,mpbl,mcloud, mbeta_aero, mextinc_aero,maod, mvis


if __name__ == '__main__':
    fn_dir = './inputRMFileData/202107'
    chose_date = '20210701'
    out_csv = './outputCSV'
    out_PNG = './outputPNG'
    back_cor_data, dr_data, total_rcs,pbl,cloud, beta_aero, extinc_aero,aod, vis \
    = main_process(fn_dir=fn_dir,chose_date=chose_date)

    # df_data_list = [back_cor_data, dr_data, total_rcs,pbl,cloud, beta_aero, extinc_aero,aod, vis]
    # save_all_result(chose_date=chose_date, out_csv=out_csv,df_data_list=df_data_list)

    pl = plotLidar(aod,chose_date,saveStr=False)
    pl.plotAOD()