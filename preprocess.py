
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from scipy.io import wavfile
from vaghelpers import vag2float
from workstationfile import return_data_locs
from batch_segmentation import *
from pathlib import Path

# 1 for großhadern, 2 for home, 3 for laptop, 4 for server
filePath, labelPath, savepath = return_data_locs(2)

# read in all labels
Knee_label_list = pd.read_excel(labelPath + 'Übersicht Probanden_Patienten_Upload.xlsx')

# read in all data location
session_list = os.listdir(filePath)


for session in range(len(session_list)):
    patient_files = os.listdir(filePath + session_list[session])
    approved = ["wav"]

    # filter out all the wav files and sort by sensor (2 runs/2 files)
    patient_files[:] = [w for w in patient_files if any(sub in w for sub in approved)]
    patella_data = [s for s in patient_files if 'Patella' in s]
    tibiaMedial_data = [s for s in patient_files if 'Medial' in s]
    tibiaLateral_data = [s for s in patient_files if 'Lateral' in s]

    samplerate, bone_music = wavfile.read(filePath + session_list[session] + '\\' + patient_files[session])

    bone_music  = vag2float(bone_music , np.float32)

    signal = bone_music [:, 0]
    angles = bone_music [:, 1]

    # test plotting
    length = bone_music .shape[0] / samplerate
    time = np.linspace(0., length, bone_music .shape[0])

    fig, axs = plt.subplots(2)
    axs[0].plot(time, angles, label="right channel")
    axs[0].legend()
    plt.xlabel("Time in seconds")
    plt.ylabel('Amplitude')
    axs[1].plot(time, signal, label="left channel")
    axs[1].legend()
    plt.xlabel("Time in seconds")
    plt.ylabel('Amplitude')
    plt.show()
    breaking = 0

    # todo write automation script and seperation script, plot visuals, pre-process audio data, send patrick data and paper, read some stuff on instrument or speech distinction, learn
    # todo how to do transfer learning and try with pre-trained networks, try the matlab code to distringuish between healthy and sick patients, look and weights and biases
    # run the entire code from nima and walther on their data again, try on ours

    # bone_music  = preprocess(bone_music )  # return divided section from a single recording session
    sections = segmentation_jhu(samplerate, angles)

    # maybe save as single sequences for faster read in

    XSEGMENTS = numpy.sort(sections)
    XSEGMENTS = numpy.reshape(XSEGMENTS, (-1, 2))
    output_filepath = askdirectory(parent=root)
    output_filepath = os.path.join(os.path.normcase(output_filepath), realname)
    for i in range(0, len(XSEGMENTS)):
        wavfile.write(output_filepath + "_segment_" + str(i + 1) + ".wav", samplerate, bone_music [XSEGMENTS[i][0]:XSEGMENTS[i][1]])
    print(u"ExportToFiles ... Finish!")


    #create path if it does not exist yet
    Path("/my/directory").mkdir(parents=True, exist_ok=True)
    
    # preprocess (read in all data, save in structure, split in extension flexion cycle)

