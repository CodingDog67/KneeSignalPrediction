
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from vaghelpers import vag2float
from workstationfile import return_data_locs


# 1 for großhadern, 2 for home, 3 for laptop, 4 for server
filePath, labelPath = return_data_locs(1)

# read in all labels
Knee_label_list = pd.read_excel(labelPath + 'Übersicht Probanden_Patienten_Upload.xlsx')

# read in all data location
session_list = os.listdir(filePath)


for session in range(len(session_list)):
    patient_files = os.listdir(filePath + session_list[0])
    approved = ["wav"]


    # filter out all the wav files and sort by sensor (2 runs/2 files)
    patient_files[:] = [w for w in patient_files if any(sub in w for sub in approved)]
    patella_data = [s for s in patient_files if 'Patella' in s]
    tibiaMedial_data = [s for s in patient_files if 'Medial' in s]
    tibiaLateral_data = [s for s in patient_files if 'Lateral' in s]

    samplerate, knee_data = wavfile.read(filePath + session_list[0] + '/' + patient_files[0])

    knee_data = vag2float(knee_data, np.float32)

    # test plotting
    length = knee_data.shape[0] / samplerate
    time = np.linspace(0., length, knee_data.shape[0])

    fig, axs = plt.subplots(2)
    axs[0].plot(time, knee_data[:, 1], label="right channel")
    axs[0].legend()
    plt.xlabel("Time in seconds")
    plt.ylabel('Amplitude')
    axs[1].plot(time, knee_data[:, 0], label="left channel")
    axs[1].legend()
    plt.xlabel("Time in seconds")
    plt.ylabel('Amplitude')
    plt.show()
    breaking = 0

    # knee_data = preprocess(knee_data)  # return divided section from a single recording session

    # maybe save as single sequences for faster read in

    # preprocess (read in all data, save in structure, split in extension flexion cycle)

