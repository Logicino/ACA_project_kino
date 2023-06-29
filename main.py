import numpy as np
import pandas as pd
import csv
from scipy.interpolate import interp1d

def read_csv(csv_dir, row_num = 0):
    # 逐行读取 CSV 文件，并且只输出第一列
    with open(csv_dir) as csvfile:
        reader = csv.reader(csvfile)
        lst = []
        for row in reader:
            lst.append(row[row_num])  # 获取第一列的数据
    return lst

def read_txt_file(file_name, col):
    # 打开txt文件
    with open(file_name, 'r') as f:
        # 读取txt数据
        lines = f.readlines()

    # 解析每一行，并且提取第col列数据
    data = []
    for line in lines:
        # 将一行数据按照tab分割成多个字段
        fields = line.strip().split('\t')
        # 取出第col列数据，并转换为浮点型数值
        value = float(fields[col])
        data.append(value)

    return data

def count_error(predicted_lst, ground_truth_lst):
    err = 0
    lst_len = len(predicted_lst)
    for i in range(lst_len):
        if ground_truth_lst[i] * 0.9715 <= predicted_lst[i] <= ground_truth_lst[i] * 1.0293:
            err = err + 0
        else:
            err = err + 1

    return err, lst_len

def main(csv_dir, txt_dir, method_row):
    # csv_dir = "E:\\testdata\\URMP\\vn\\AuSep_1_vn_01_Jupiter.csv"  # target x
    # txt_dir = "E:\\testdata\\URMP\\vn\\F0s_1_vn_01_Jupiter.txt" # origin x & y

    # csv只用读target的time值
    time_lst_target = read_csv(csv_dir=csv_dir, row_num=0)
    time_lst_target = time_lst_target[1:]  # 去掉time
    time_lst_target = [float(i) for i in time_lst_target]
    time_lst_target = time_lst_target[:-2]

    # txt是源值
    time_lst_old = read_txt_file(txt_dir, 0)
    val_lst_old = read_txt_file(txt_dir, 1)

    # 做插值
    '''
    ‘linear’：线性插值（默认值），对应于传统的拉格朗日插值方法
    ‘nearest’：最近邻插值，即取离要求插值点最近的已知数据点的函数值作为插值结果
    ‘zero’：零阶插值，即在两个已知数据点之间取函数值为0的水平线段作为插值结果
    ‘slinear’：一次样条插值，即所谓的“分段线性插值”
    ‘quadratic’：二次插值
    ‘cubic’：三次插值
    ‘previous’：向前差值（piecewise constant function）
    ‘next’：向后差值（piecewise constant function）
    '''
    f = interp1d(time_lst_old, val_lst_old, kind='nearest')

    # 按照计算结果的x轴，重新采样的groundtruth
    ground_truth_lst_resampled = f(time_lst_target)

    # 算法算出来的结果
    '''
    1 SpectralAcf	
    2 SpectralHps	
    3 TimeAcf	
    4 TimeAmdf	
    5 TimeAuditory	
    6 TimeZeroCrossings	
    7 YIN	
    8 PYIN
    '''
    val_lst_target = read_csv(csv_dir=csv_dir, row_num=method_row)
    val_lst_target = val_lst_target[1:]  # 去掉time
    val_lst_target = [float(i) for i in val_lst_target]
    val_lst_target = val_lst_target[:-2]
    # 进行对比
    err = count_error(predicted_lst=val_lst_target, ground_truth_lst=ground_truth_lst_resampled)

    return err


import os
root_dir = "D:\\课程\\ACA-Slides-2nd_edition\\URMP"
instrument_list = os.listdir(root_dir)

# 新建一个全0的二维数组，行数：乐器数=13，列数：方法数=3
dim_1 = [0 for index in range(3)]
data_matrix = [list(dim_1) for index in range(13)]

# 遍历乐器
for instrument_num in range(len(instrument_list)): # 外循环：instrument
    instrument_lst_path = os.path.join(root_dir, instrument_list[instrument_num]) # "D:\\课程\\ACA-Slides-2nd_edition\\URMP\\bn"
    instrument_dir = os.listdir(instrument_lst_path)
    # 只留下结尾为register.csv和txt的groundtruth
    instrument_dir = [file_name for file_name in instrument_dir if file_name.endswith('_register.csv') or file_name.endswith('.txt') ]

    instrument_dir_len = len(instrument_dir)

    total_err = 0
    total_lst_len = 0

    for method_row in range(1, 4):  # 方法数目，这里是
        for i in range(instrument_dir_len//2): # 组数
            csv_dir = instrument_dir[i]
            csv_dir = os.path.join(instrument_lst_path, csv_dir)
            txt_dir = instrument_dir[i+instrument_dir_len//2]
            txt_dir = os.path.join(instrument_lst_path, txt_dir)

            err, lst_len = main(csv_dir=csv_dir, txt_dir=txt_dir, method_row=method_row)  # 内循环：method

            total_err = total_err + err
            total_lst_len = total_lst_len + lst_len

        err_rate = total_err / total_lst_len

        data_matrix[instrument_num][method_row-1] = err_rate
        print("total error is", total_err/total_lst_len)

    np.save("data_matrix_register.npy", data_matrix)

