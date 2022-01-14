
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pathlib import Path

filePath = '/media/ari/Harddisk/knee data/knee signal/Patienten/'
labelPath = '/media/ari/Harddisk/knee data/knee signal/'


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
    # test plotting
    length = knee_data.shape[0] / samplerate
    time = np.linspace(0., length, knee_data.shape[0])
    plt.plot(time, knee_data[:, 1], label="left channel")
    plt.legend()
    plt.xlabel("Time in seconds")
    plt.ylabel('Amplitude')
    plt.show()
    breaking = 0


    # knee_data = preprocess(knee_data)  # return divided section from a single recording session


    # maybe save as single sequences for faster read in


    # preprocess (read in all data, save in structure, split in extension flexion cycle)

