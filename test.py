import os
from pyACA import computePitchCl
import pandas as pd
import librosa
import numpy as np

PitchTrackerNameList = ['SpectralAcf',
                        'SpectralHps',
                        'TimeAcf',
                        'TimeAmdf',
                        'TimeAuditory',
                        'TimeZeroCrossings',]
FRAME_LENGTH = 4096
HOP_LENGTH = 2048


def testfolder(file_list):
    for file_path in file_list:
        if file_path.endswith('.wav'):  # 是wav文件，扔进去算基频
            df = pd.DataFrame()
            print('Dealing with ' + file_path)
            for PitchTrackName in PitchTrackerNameList:
                print('Using ' + PitchTrackName)
                f0, t = computePitchCl(file_path, PitchTrackName, False)
                f0 = pd.DataFrame(f0, columns=[PitchTrackName])
                if PitchTrackName == 'SpectralAcf':
                    t = pd.DataFrame(t, columns=['time'])
                    df = pd.concat([df, t, f0], axis=1)
                else:
                    df = pd.concat([df, f0], axis=1)

            # YIN
            signal, sample_rate = librosa.load(file_path, sr=None)
            f0_yin = librosa.yin(signal, fmin=50, fmax=2000, sr=sample_rate, frame_length=FRAME_LENGTH,
                                 hop_length=HOP_LENGTH)

            f0_yin = pd.DataFrame(f0_yin, columns=['YIN'])

            # PYIN
            f0_pyin, _, _ = librosa.pyin(signal, fmin=50, fmax=2000, sr=sample_rate, frame_length=FRAME_LENGTH,
                                         hop_length=HOP_LENGTH, fill_na=0)
            f0_pyin = pd.DataFrame(f0_pyin, columns=['PYIN'])
            df = pd.concat([df, f0_yin, f0_pyin], axis=1)

            df.to_csv(file_path[:-4] + '.csv', index=False)


def revisefolder(file_list):
    for file_path in file_list:
        if file_path.endswith('.wav'):  # 是wav文件，扔进去算基频
            df = pd.read_csv(file_path[:-4] + '.csv', usecols=[0, 1, 2, 3, 4, 5, 6])
            print('Dealing with ' + file_path)

            # YIN
            signal, sample_rate = librosa.load(file_path, sr=None)
            f0_yin = librosa.yin(signal, fmin=50, fmax=2000, sr=sample_rate, frame_length=FRAME_LENGTH,
                                 hop_length=HOP_LENGTH)

            f0_yin = pd.DataFrame(f0_yin, columns=['YIN'])

            # PYIN
            f0_pyin, _, _ = librosa.pyin(signal, fmin=50, fmax=2000, sr=sample_rate, frame_length=FRAME_LENGTH,
                                         hop_length=HOP_LENGTH, fill_na=0)
            f0_pyin = pd.DataFrame(f0_pyin, columns=['PYIN'])
            df = pd.concat([df, f0_yin, f0_pyin], axis=1)

            df.to_csv(file_path[:-4] + '.csv', index=False)


def testURMP():
    urmp_path = '.\\testdata\\URMP'
    urmp_folder_list = [os.path.join(urmp_path, dir) for dir in os.listdir(urmp_path)]

    for urmp_folder in urmp_folder_list:
        file_list = [os.path.join(urmp_folder, file) for file in os.listdir(urmp_folder)]
        testfolder(file_list)


def testVocadito():
    vocadito_path = '.\\testdata\\vocadito\\Audio'
    vocadito_file_list = [os.path.join(vocadito_path, dir) for dir in os.listdir(vocadito_path)]

    testfolder(vocadito_file_list)




def testFlute():
    flute_path = '.\\testdata\\traditional-flute-dataset\\audio'
    flute_file_list = [os.path.join(flute_path, dir) for dir in os.listdir(flute_path)]

    testfolder(flute_file_list)


def revise_yin():
    flute_path = '.\\testdata\\traditional-flute-dataset\\audio'
    flute_file_list = [os.path.join(flute_path, dir) for dir in os.listdir(flute_path)]
    revisefolder(flute_file_list)

    vocadito_path = '.\\testdata\\vocadito\\Audio'
    vocadito_file_list = [os.path.join(vocadito_path, dir) for dir in os.listdir(vocadito_path)]
    revisefolder(vocadito_file_list)

    urmp_path = '.\\testdata\\URMP'
    urmp_folder_list = [os.path.join(urmp_path, dir) for dir in os.listdir(urmp_path)]

    for urmp_folder in urmp_folder_list:
        file_list = [os.path.join(urmp_folder, file) for file in os.listdir(urmp_folder)]
        revisefolder(file_list)



def getduration():
    urmp_path = '.\\testdata\\URMP'
    urmp_folder_list = [os.path.join(urmp_path, dir) for dir in os.listdir(urmp_path)]

    for urmp_folder in urmp_folder_list:
        file_list = [os.path.join(urmp_folder, file) for file in os.listdir(urmp_folder)]
        duration = get_folder_duration(file_list)
        print(urmp_folder, duration)



def get_folder_duration(file_list):
    total_duration = 0
    for file_path in file_list:
        if file_path.endswith('.wav'):  # 是wav文件，统计时长
            duration = librosa.get_duration(path=file_path)
            total_duration += duration
    return total_duration




if __name__ == '__main__':
    result = np.load('./data_matrix.npy')
    print(result)


