import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class plotLidar():
    def __init__(self,dfPlot:"pd.DataFrame",date:'str',rangeIndex:'tuple' = (0, 8), outPNG:"str"='./outputPNG',saveStr=True):
        self.plotData = dfPlot
        self.date = date

        # 数据显示范围
        self.startIndex = int(rangeIndex[0] * 100)
        self.endIndex = int(rangeIndex[-1] * 100)
        self.timeindex = self.plotData.columns
        # 文件保存路径
        self.outPNG = outPNG
        self.saveStr = saveStr
        self.savepath = './' + self.outPNG + '/%s' % self.date[:6]
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.tt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def plotRCS(self):
        """
        可视化RCS信号

        Args:

        Returns:

        """
        _rcs = self.plotData.iloc[self.startIndex:self.endIndex, :] # 根据范围截取数

        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')
        plt.imshow(_rcs, cmap='rainbow', norm=LogNorm(vmin=450000.0, vmax=1000000.0),origin='lower', aspect='auto')
        clb = plt.colorbar() # 显示色卡图例
        clb.set_label('RCS')

        plt.title("RCS Singal plot %s" % self.date, fontsize=14)
        plt.ylabel('Height (km)',fontsize=14)

        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)
        ytickStep = np.arange(self.startIndex, self.endIndex+100, 100)
        plt.yticks(ytickStep,ytickStep*7.5/1000, fontsize=12)

        figname =  self.savepath + '/'+ self.date+ 'RCS' + '.png'
        saveNotes = "RCS singal plot saved"
        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()

    def plotDR(self):
        """
        可视化退偏比

        Args:

        Returns:

        """
        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')
        _drData = self.plotData.iloc[self.startIndex:self.endIndex, :]

        plt.imshow(_drData, cmap='rainbow', norm=LogNorm(vmin=0.5, vmax=20),
                origin='lower', aspect='auto')
        clb = plt.colorbar() # 显示色卡图例
        clb.set_label('DepolarizationRatio')
        plt.title("DepolarizationRatio Singal plot %s" % self.date, fontsize=14)
        figname =  self.savepath + '/'+ str(self.date)+ 'DepolarizationRatio' + '.png'
        saveNotes = "DepolarizationRatio singal plot saved"

        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)
        ytickStep = np.arange(self.startIndex, self.endIndex+100, 100)
        plt.yticks(ytickStep,ytickStep*7.5/1000, fontsize=12)
        
        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()

    def plotPBL(self):
        """
        可视化边界层高度

        Args:

        Returns:

        """

        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')
        _pbl = self.plotData.iloc[0, :] # 根据范围截取数据
        # 绘制PBL曲线
        plt.plot(_pbl, 'deepskyblue', linewidth=2)

        plt.title("PBL Height plot %s" % self.date, fontsize=14)
        plt.ylabel('PBL Height (m) ', fontsize=14)
        figname =  self.savepath + '/'+ self.date+ 'PBL' + '.png'
        saveNotes =  "PBL Height plot saved"

        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)
           
        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()

    def plotCloud(self,rcsdf):
        """
        可视化rcs、云底和云高

        Args:

        Returns:

        """
        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')

        _cloud_base = self.plotData.iloc[1, :]  # index
        _cloud_top = self.plotData.iloc[3, :]
        _rcs = rcsdf.iloc[self.startIndex:self.endIndex, :]

        # 绘制RCS背景图
        plt.imshow(_rcs, cmap='rainbow', norm=LogNorm(vmin=450000.0, vmax=1000000.0),
                origin='lower', aspect='auto')
        clb = plt.colorbar()
        clb.set_label('RCS')
        
        # 绘制云底高度
        plt.plot(_cloud_base, 'k*', linewidth=4)
        # 绘制云顶高度
        plt.plot(_cloud_top, 'k*', linewidth=4)

        plt.title("Cloud height plot %s" % self.date, fontsize=14)
        figname =  self.savepath + '/'+ str(self.date)+ 'Cloud' + '.png'
        saveNotes = "Cloud height plot saved"
        plt.ylabel('Cloud Base and Top (km)', fontsize=14)

        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)
        ytickStep = np.arange(self.startIndex, self.endIndex+100, 100)
        plt.yticks(ytickStep,ytickStep*7.5/1000, fontsize=12)
        
        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()

    def plotBackCoe(self):
        """
        可视化后项散射系数
        Args:

        Returns:

        """
        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')
        _backCoe = self.plotData.iloc[self.startIndex:self.endIndex, :]

        plt.imshow(_backCoe, cmap='rainbow', norm=LogNorm(vmin=1e-5, vmax=1e+2),
                origin='lower', aspect='auto')
        clb = plt.colorbar() # 显示色卡图例
        clb.set_label('BackScattering')
        saveNotes = "BackScattering singal plot saved"
        figname =  self.savepath + '/'+ str(self.date)+ 'BackScattering' + '.png'
        plt.title("BackScattering Singal plot %s" % self.date, fontsize=14)
        plt.ylabel('Height (km)', fontsize=14)

        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)
        ytickStep = np.arange(self.startIndex, self.endIndex+100, 100)
        plt.yticks(ytickStep,ytickStep*7.5/1000, fontsize=12)
        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()

    def plotExtin(self):
        """
        可视化消光系数

        Args:

        Returns:

        """

        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')
        _ext = self.plotData.iloc[self.startIndex:self.endIndex, :]

        plt.imshow(_ext, cmap='rainbow', norm=LogNorm(vmin=1e-5, vmax=1e+2),
                origin='lower', aspect='auto')
        clb = plt.colorbar() # 显示色卡图例
        clb.set_label('Extinction')
        figname =  self.savepath + '/'+ str(self.date)+ 'Extinction' + '.png'
        saveNotes = "Extinction singal plot saved"
        plt.title("Extinction Singal plot %s" % self.date, fontsize=14)
        plt.ylabel('Height (km)', fontsize=14)

        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)
        ytickStep = np.arange(self.startIndex, self.endIndex+100, 100)
        plt.yticks(ytickStep,ytickStep*7.5/1000, fontsize=12)
        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()
            
    def plotAOD(self):
        """
        可视化光学厚度

        Args:

        Returns:

        """

        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')
        _aod = self.plotData.iloc[0, :].values/1000

        plt.plot(_aod, 'deepskyblue', linewidth=2)
        plt.title("AOD plot %s" % self.date, fontsize=14)
        figname =  self.savepath + '/'+ str(self.date)+ 'AOD' + '.png'
        saveNotes = "AOD plot saved"

        plt.tick_params(axis='y',direction='out',which='both',left=True,right=False,labelbottom=True,labelsize=15)

        plt.ylabel('AOD [-]', fontsize=14)
        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)

        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()

    def plotVIS(self):
        """
        可视化能见度

        Args:

        Returns:

        """
        _vis = self.plotData.iloc[0, :].values/1000
        figname =  self.savepath + '/'+ str(self.date)+ 'VIS' + '.png'
        saveNotes = "VIS Height plot saved"
        yLabel = 'Vis Height (km) '
        pltTitle = "VIS plot %s" % self.date

        fig = plt.figure(figsize=(12, 8), dpi=100,facecolor='white')
        # 绘制曲线
        plt.plot(_vis)
        plt.ylabel(yLabel, fontsize=14)
        plt.title(pltTitle, fontsize=14)

        plt.tick_params(axis='y',direction='out',which='both',left=True,right=False,labelbottom=True,labelsize=15)
        if len(self.timeindex) < 60:
            xTickStep = 1
        else:
            xTickStep = 60
        plt.xticks((np.arange(0, len(self.timeindex), xTickStep)),
                [timestr for timestr in self.timeindex[np.arange(0, len(self.timeindex), xTickStep)]], fontsize=12, rotation=30)

        if self.saveStr:
            plt.savefig(figname)
            plt.close()
            print("[%s]" % self.tt, "\t", saveNotes)
        else:
            plt.show()

if __name__ == '__main__':
    out_csv = 'outputCSV/20210701_RCS.csv'
    # out_csv = 'outputCSV/20210701_DepolarizationRatio.csv'
    # out_csv = 'outputCSV/20210701_pblHeight.csv'
    # out_csv = 'outputCSV/20210701_Cloud.csv'
    # out_csv2 = 'outputCSV/20210701_RCS.csv'
    # out_csv = 'outputCSV/20210701_BackScattering.csv'
    # out_csv = 'outputCSV/20210701_Extinction.csv'
    # out_csv = 'outputCSV/20210701_AerosolOpticalDepth.csv'
    # out_csv = 'outputCSV/20210701_Visibility.csv'


    out_PNG = './outputPNG'
    pltData = pd.read_csv(out_csv,index_col=0)
    # rcsdf = pd.read_csv(out_csv2,index_col=0)
    chose_date = '20210701'
    pl = plotLidar(dfPlot=pltData,date=chose_date,saveStr=True)
    pl.plotRCS()
    # pl.plotDR()
    # pl.plotPBL()
    # pl.plotCloud(rcsdf=rcsdf)
    # pl.plotBackCoe()
    # pl.plotExtin()
    # pl.plotAOD()
    # pl.plotVIS()