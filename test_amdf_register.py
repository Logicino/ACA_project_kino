'''
test the accuracy when given the resigter (f_min and f_max) of instrument
use TimeAmdf method
'''


import os

import pyACA.PitchTimeAmdf
from pyACA.PitchTimeAmdf import PitchTimeAmdf
import pandas as pd
import librosa

instrument_register = {
    "bn": [58.270, 587.33],
    "cl": [146.83, 1568.0],
    "db": [32.703, 246.94],
    "fl": [261.63, 2093.0],
    "hn": [65.406, 783.99],
    "ob": [233.08, 1568.0],
    "sax": [103.83, 830.61],
    "tba": [36.708, 233.08],
    "tbn": [41.203, 523.25],
    "tpt": [164.81, 1318.5],
    "va": [130.81, 1568.0],
    "vc": [65.406, 1568.0],
    "vn": [196.00, 2637.0]
}
FRAME_LENGTH = 4096
HOP_LENGTH = 2048




if __name__ == '__main__':
    urmp_path = '.\\testdata\\URMP'
    urmp_folder_list = [os.path.join(urmp_path, dir) for dir in os.listdir(urmp_path)]

    for urmp_folder in urmp_folder_list:
        instrument = urmp_folder.split("\\")[-1]
        f_min = instrument_register[instrument][0]
        f_max = instrument_register[instrument][1]
        file_list = [os.path.join(urmp_folder, file) for file in os.listdir(urmp_folder)]
        for file_path in file_list:
            if file_path.endswith('.wav'):  # 是wav文件，扔进去算基频
                print('Dealing with ' + file_path)
                signal, sample_rate = librosa.load(file_path, sr=None)

                # timeAmdf
                f0_amdf, t = PitchTimeAmdf(signal, FRAME_LENGTH, HOP_LENGTH, sample_rate, f_min=f_min, f_max=f_max)
                f0_amdf = pd.DataFrame(f0_amdf, columns=['TimeAmdf'])
                t = pd.DataFrame(t, columns=['time'])

                # YIN
                signal, sample_rate = librosa.load(file_path, sr=None)
                f0_yin = librosa.yin(signal, fmin=f_min, fmax=f_max, sr=sample_rate, frame_length=FRAME_LENGTH,
                                     hop_length=HOP_LENGTH)
                f0_yin = pd.DataFrame(f0_yin, columns=['YIN'])

                # PYIN
                f0_pyin, _, _ = librosa.pyin(signal, fmin=f_min, fmax=f_max, sr=sample_rate, frame_length=FRAME_LENGTH,
                                             hop_length=HOP_LENGTH, fill_na=0)
                f0_pyin = pd.DataFrame(f0_pyin, columns=['PYIN'])

                df = pd.concat([t, f0_amdf, f0_yin, f0_pyin], axis=1)

                df.to_csv(file_path[:-4] + '_register.csv', index=False)