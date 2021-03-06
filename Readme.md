#  介绍
激光雷达解析、反演、可视化程序

# 激光雷达设备及数据格式
- 设备说明： Raymetrics LR111-D300型激光雷达是一种主动式激光遥感仪器，旨在提供丰富的大气信息，包括气溶胶负荷、PBL混合高度、火山灰的明确识别和灰层高度。该系统还可以升级到探测水蒸气，允许进行远程湿度剖析（仅在夜间）。为气象和航空应用而设计，其规格是根据英国气象局和EARLINET（欧洲激光雷达网络）的要求确定的，使LR111-D300可能成为商业上最强大的眼球安全气溶胶激光雷达。

- 产品参考链接：[引用日期2021-05-06](https://www.environmental-expert.com/products/raymetrics-model-lr111-d300-raman-depolarization-lidar-for-meteorological-applications-442436#)
  
- 数据格式
  产品数据格式为标准Licel格式即ascii格式文件头binary数据块。

	```
	RM2173122.131155                                                             
	full     31/07/2021 22:12:11 31/07/2021 22:13:11 0110 113.5696 22.1657 -90.0 0.0 29.8 986.7
	0001201 0020 0001201 0000 06                                                 
	1 0 1 16230 1 0820 7.50 00355.p 0 0 09 000 12 001201 0.100 BT0               
	1 1 1 16230 1 0820 7.50 00355.p 0 0 00 000 00 001201 3.9683 BC0              
	1 0 1 16230 1 0900 7.50 00355.s 0 0 09 000 12 001201 0.100 BT1               
	1 1 1 16230 1 0900 7.50 00355.s 0 0 00 000 00 001201 3.9683 BC1              
	1 0 2 16230 1 0800 7.50 00387.o 0 0 09 000 12 001201 0.100 BT2               
	1 1 2 16230 1 0800 7.50 00387.o 0 0 00 000 00 001201 3.9683 BC2              
	# 空行
	#以下开始是二进制数据块
	```

	```
	第1行（文件名）：RMyyMddhh.mmssmsms

	第2行（站点信息）：
	站点名 开始时间 结束时间 海拔(m) 经度 纬度 天顶角 方位角 地表温度（℃）地表气压（hPa）

	第3行（发射器信息）：激光器1发射次数 激光器1重复频率 激光器2发射次数 激光器2重复频率 数据集个数

	第4~6行：各数据集信息采集信息
	[1 0 1] 			数据是1/否0存在 模拟0/光子1计数模式 激光器1/2
	[16230]				数据bins
	[1]					固定值
	[0820 7.50 00355.p] PMT高电压值 bin宽度(m) 激光波长和极化方式(o-无极化；s-垂直；p-平行)
	[0 0 09 000]		向后兼容性
	[12 001201]			模拟数据集ADC位数，否则为0 拍摄次数
	[0.100] 			模拟数据集，以mVolt为单位的输入范围；光子数据集，鉴别器级别
	[BT0]				数据集描述和瞬时记录器编号，BT=模拟数据集，BC=光子数据集，
	```
	数据集描述后跟一个额外的 CRLF。数据集是 32 位整数值。数据集由 CRLF 分隔。最后一个数据集后面是 CRLF。这些 CRLF 用作标记，可用作文件完整性的检查点。
- 设备通道信息

	| 通道数 | 波长 | 极化方式 | 信号记录方式 |
	| ------ | ---- | -------- | ------------ |
	| 通道1  | 355  | p        | BT           |
	| 通道2  | 355  | p        | BC           |
	| 通道3  | 355  | s        | BT           |
	| 通道4  | 355  | s        | BC           |
	| 通道5  | 387  | o        | BT           |
	| 通道6  | 387  | o        | BC           |

# 激光雷达产品分类
## 廓线产品包括：
- 1级：
	距离修正后的回波信号
- 2级：
	气溶胶后向散射系数廓线、气溶胶消光系数廓线、退偏比廓线、~色比廓线~
- 3级：
	浓度等其他参数
	
## 非廓线产品包括：
- 大气边界层高度
- 气溶胶光学厚度
- 云底、云高
- 其他

# 产品反演原理概述
- 距离修正后的回波信号：
    对环境监测激光雷达探测得到的数据进行二进制读取(bin_data)、通道拼接(channel_data)、雷达信号计算(raw_data)、
	几何因子修正(overlap_correct)、背景噪声去除(background_correct)、平滑(smoothing)和距离平方修正(dis_correct)后得到的信号(rcs)。

- 气溶胶后向散射系数廓线/气溶胶消光系数廓线：
	根据估计的分子后向散射系数(bate_mol),基于Fernald方法进行雷达方程求解得到气溶胶后向散射系数(beta_aero)。根据雷达比(lidar_radio)和beta_aero得到气溶胶消光系数(extin_aero)

- 退偏比廓线：
    取532mn的垂直后向散射强度和平行后向散射强度之比，结果即为退偏比(dr)。

- ~色比廓线：~
    不同波长的后向散射强度之比：

- 颗粒物浓度廓线：
	通过气溶胶消光系数廓线，以退偏比廓线为辅助，通过poliphon1阶和2阶算法反演得到。

- 边界层高度：
	重力波梯度演算法（专利号 CN 103135113 B）
	对每一距离订正回波信号，开三次方后求其梯度，梯度最小值所对应的散射目标高度即为大气边界层高度(pbl)，具体如以下公式所示：
	$$h_{CRGM}=min⁡[(\frac{∆(RCS)^{(1/3)}}{∆R}]$$
    其中，RCS为距离订正回波信号；R为该回波信号对应的散射目标高度

- 气溶胶光学厚度：
	将消光系数随高度进行积分，就能够得到气溶胶的光学厚度（AOD）。

- 云底、云高：
	阈值法判断当前时刻是否有云进而计算云底(cloud_bottom)、云高(cloud_top)。

*其中，大气边界层高度和云高等产品主要是基于定性处理，糈度高：气溶胶后向散射系数廓线、气溶胶消光系数廓线、颗粒物浓度廓线、气溶胶光学厚度等产品主要是基于定量处理，糈度较低：

## 不确定性分析
激光雷达反演的误差来源，主要在于一个方程（激光雷达信号方程）解两个未知数（大气气溶胶的消光系数和后向散射系数均为未知）。在反演时，一般假设大气气溶胶的消光系数和后向散射系数两者比值固定（称为激光雷达比）。
应用Fernald法后向反演激光雷达方程时，需要先确定3个参数：激光雷达比、定标高度（后向反演的起始高度）以及该高度处的大气气溶胶消光系数。
- 激光雷达比
气溶胶的激光雷达比与气溶胶粒于的折射率、尺寸、形态和组成等诸多因素有关，而实际气溶胶葙子的尺寸、形态、组成、折射率等参数的差异很大，因此难以确定气溶胶激光雷达比。实际反演消光系数时，人们通常依据不同参考条件来确定大气气溶胶的激光雷达比值，例如有文献指出，火山爆发后大量气溶胶进
人平流层，此时大气气溶胶的激光雷达比分别取40（6 ~ 15km)，22（15 ~ 20km），40（20 ~ 25km）和43（25 ~ 30km），对处于背景期的平流层气溶胶和对流层气溶胶，激光雷达比的参考值可以设置为50。
