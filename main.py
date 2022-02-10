# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from batch_segmentation import *

from pathlib import Path


filePath = '/media/ari/Harddisk/knee data/knee signal/Patienten/'
labelPath = '/media/ari/Harddisk/knee data/knee signal/'
# general testing if file exists
directory = Path('/media/ari/Harddisk/knee data/knee signal/')
isit = directory.is_dir()
file_there = os.path.isfile(labelPath + 'Übersicht Probanden_Patienten_Upload.xlsx')

# read in labels
Knee_label_list = pd.read_excel(labelPath + 'Übersicht Probanden_Patienten_Upload.xlsx')


# read in data
session_list = os.listdir(filePath)
patient_files = os.listdir(filePath + session_list[0])
approved = ["wav"]

# filter out all the wav files and sort by sensor (2 runs/2 files)
patient_files[:] = [w for w in patient_files if any(sub in w for sub in approved)]
patella_data = [s for s in patient_files if 'Patella' in s]
tibiaMedial_data = [s for s in patient_files if 'Medial' in s]
tibiaLateral_data = [s for s in patient_files if 'Lateral' in s]

samplerate, bone_music  = wavfile.read(filePath + session_list[0] + '/' + patient_files[0])
# test plotting
bone_sound = bone_music .shape[0]
angles = bone_music [:, 1]

length = bone_sound / samplerate
time = np.linspace(0., length, bone_music .shape[0])
plt.plot(time, angles, label="left channel")
plt.legend()
plt.xlabel("Time in seconds")
plt.ylabel('Amplitude')
plt.show()
breaking = 0

sections = segmentation_jhu(samplerate, )

# preprocess (read in all data, save in structure, split in extension flexion cycle)

