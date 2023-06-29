import numpy as np
import matplotlib.pyplot as plt

data_matrix = np.load("data_matrix.npy")
data_matrix_register = np.load("data_matrix_register.npy")

instrument_name_list = ['Basson', 'Clarinet', 'DoubleBass', 'Flute', 'Horn', 'Obeo', 'Saxphone', 'Tuba', 'Trombone','Trumpet', 'Viola', 'Cello', 'Violin']

def plot_different_method_error_rate():
    for i in range(13):
        instrument_data = data_matrix[i]
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
        methods = ['SpectralAcf', 'SpectralHps', 'TimeAcf', 'TimeAmdf', 'TimeAuditory', 'TimeZeroCrossings', 'YIN', 'PYIN']

        fig, ax = plt.subplots()
        ax.barh(methods, instrument_data, height=0.5)

        # 调整间距
        plt.subplots_adjust(left=0.25)
        plt.xlabel('Error rate')

        plt.ylabel('Methods')

        instrument_name = instrument_name_list[i]

        plt.title(f'Error rate of different pitch detection methods of {instrument_name}')

        plt.show()

def plot_different_method_error_register():
    # 遍历方法：TimeAmdf; YIN; PYIN
    methods_name = ['TimeAmdf', 'YIN', 'PYIN']
    instrument_data = []

    instrument_data_register = []
    for i in range(3):
        if i == 0:
            instrument_data = data_matrix[:, 3]
        if i == 1:
            instrument_data = data_matrix[:, 6]
        if i == 2:
            instrument_data = data_matrix[:, 7]

        instrument_data_register = data_matrix_register[:, i]

        attribute1 = instrument_data
        attribute2 = instrument_data_register

        x_positions = np.arange(len(instrument_name_list))

        # 绘制柱状图
        plt.bar(x_positions - 0.2, attribute1, width=0.4, label='original')
        plt.bar(x_positions + 0.2, attribute2, width=0.4, label='refined_register')

        # 添加标签
        plt.xlabel('Instruments')
        plt.ylabel('Error rate')
        methods_name_tmp = methods_name[i]
        plt.title(f'{methods_name_tmp} Comparison')

        plt.xticks(x_positions, instrument_name_list,rotation=45)

        # plt.xticks(x_positions, instrument_names, )
        plt.subplots_adjust(left=0.05)

        # 显示图例
        plt.legend()

        # 展示图形
        plt.show()


plot_different_method_error_register()


