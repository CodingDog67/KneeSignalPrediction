
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from scipy.io import wavfile
from vaghelpers import vag2float
from batch_segmentation import *
from pathlib import Path
import numpy as np
import matplotlib.lines as mlines


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    line = mlines.Line2D([xmin, xmax], [ymin, ymax])
    ax.add_line(line)
    return line


def preprocessing(file_path, save_path):

    # read in all data location
    session_list = sorted(os.listdir(file_path))


    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(f"{save_path} 'full file images\\").mkdir(parents=True, exist_ok=True)  # path for overall images

    for session in range(len(session_list)):
        patient_files = os.listdir(file_path + session_list[session])
        approved = ["wav"]

        # filter out all the wav files and sort by sensor (2 runs/2 files)
        patient_files[:] = [w for w in patient_files if any(sub in w for sub in approved)]

        for soundfile in range(len(patient_files)):
            samplerate, bone_music = wavfile.read(file_path + session_list[session] + '\\' + patient_files[soundfile])
            realname = patient_files[soundfile].replace(".wav", "")

            if os.path.isfile(save_path + realname + '.png'):
                continue

            bone_music = vag2float(bone_music, np.float32)

            signals = bone_music[:, 0]
            angles = bone_music[:, 1]

            #  todo
            #   read some stuff on instrument or speech distinction, learn
            #  todo how to do transfer learning and try with pre-trained networks, try the matlab code to distinguish
            #   between healthy and sick patients, look and weights and biases
            #   run the entire code from nima and walther on their data again, try on ours

            # bone_music  = preprocess(bone_music )  # return divided section from a single recording session
            sections = segmentation_jhu(samplerate, angles)

            # create path if it does not exist yet
            if session == 0:
                Path(save_path).mkdir(parents=True, exist_ok=True)

            # test plotting
            length = bone_music.shape[0] / samplerate
            time = np.linspace(0., length, bone_music.shape[0])

            even = tuple(time[sections[0:len(sections):2]])
            uneven = tuple(time[sections[1:len(sections):2]])

            fig, axs = plt.subplots(2)
            axs[0].plot(time, angles)
            axs[0].vlines(even, min(angles), max(angles), linestyles='dotted', colors='green')  # plot start of segment
            axs[0].vlines(uneven, min(angles), max(angles), linestyles='dotted', colors='red')  # plot end of segment
            # plot start of segment
            plt.xlabel("Time in seconds")
            plt.ylabel('Amplitude')
            axs[1].plot(time, signals)
            plt.xlabel("Time in seconds")
            plt.ylabel('Amplitude')
            plt.savefig(f"{save_path}full file images\\{realname}.png")
            plt.close()
            # plt.show()

            # maybe save as single sequences for faster read in
            x_segments = np.reshape(np.sort(sections), (-1, 2))

            for i in range(0, len(x_segments)):
                # extract the single movement and identify which sensor it came from
                single_movement = bone_music[:, 0]
                single_movement = single_movement[x_segments[i][0]:x_segments[i][1]]

                if 'Medial' in realname:
                    sensortype = 'Medial'
                elif 'Lateral' in realname:
                    sensortype = 'Lateral'
                else:
                    sensortype = 'Patella'

                # test plotting
                length = single_movement.shape[0] / samplerate
                time = np.linspace(0., length, single_movement.shape[0])

                plt.title(sensortype + " sensor")
                plt.xlabel("amplitude")
                plt.ylabel("time in seconds")
                plt.plot(time, single_movement)
                plt.savefig(f"{save_path}individual movements\\{realname}_segment_{str(i + 1)}.png")
                plt.close()
                # plt.show()
                wavfile.write(f"{save_path}individual movements\\{realname}_segment_{str(i + 1)}.wav",
                              samplerate, single_movement)


        print(u"Session" + str(session) + "ExportToFiles ... Finish!")


def read_save_labels(label_path):
    # read in all labels

    knee_label_list = pd.read_excel(label_path + 'labels.xlsx')
    retropatellar = knee_label_list['Retropatellar'].to_numpy()
    lateral = knee_label_list['lateral'].to_numpy()
    medial = knee_label_list['medial'].to_numpy()
    innenmeniskus = knee_label_list['Innenmeniskus'].to_numpy()
    aussenmeniskus = knee_label_list['Außenmeniskus'].to_numpy()
    session = knee_label_list['Nummer'].to_list()

    return retropatellar, lateral, medial, innenmeniskus, aussenmeniskus, session
