# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# todo read The new descriptor in processing of vibroacoustic signal of knee joint
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7506694/#B14-sensors-20-05015

import librosa.display
import torch
from sklearn.model_selection import train_test_split

import preprocess
from helpers import *
from preprocess import *
from workstationfile import return_data_locs  # consider putting all of this into a yaml file + hyperparameters

preprocess_data = False
filter_data = True
sorting = False

# 1 for großhadern, 2 for home, 3 for laptop, 4 for server
paths = return_data_locs(2)

# look at thsi : https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/blob/master/VoiceAssistant/speechrecognition/neuralnet/train.py

# look at this : https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/blob/master/VoiceAssistant/speechrecognition/neuralnet/train.py


def prepare_data():
    # preprocess data and labels, splitting into segments and saving, or simple loading from split location save labels
    # in the same way
    if preprocess_data:
        preprocessing(paths['filePath'], paths['savePath'])

    # read in data into datastructure
    if sorting:
        sortdata(paths['savePath'])


def main(alpha=0.95):
    # read in labels and data
    retropatellar, lateral, medial, innenmeniskus, aussenmeniskus, session = \
        read_save_labels(paths['labelPath'])

    prepare_data()

    # read in names of data + create a label list based on that ( labels need extending due to segmentation of segments)
    name_list_patella, name_list_medial, name_list_lateral = read_data_names_in(paths)

    label_patella = extend_labels(retropatellar, name_list_patella, session)
    label_medial = extend_labels(medial, name_list_medial, session)
    label_lateral = extend_labels(lateral, name_list_lateral, session)



    data_patella, sr_pat = read_final_data(paths['patella_data'])
    data_patella, sr_pat = read_final_data(paths['patella_data'])
    data_patella, sr_pat = read_final_data(paths['patella_data'])

    # filter data to get rid of outliers (consider different filters too)
    if filter_data:
        if os.path.isfile(paths['patella_data_smooth'] + "smoothed_cursed.npy"):
            data_pat_smooth = np.load(paths['patella_data_smooth'] + "smoothed.npy", allow_pickle=True)
            data_med_smooth = np.load(paths['medial_data_smooth'] + "smoothed.npy", allow_pickle=True)
            data_lat_smooth = np.load(paths['lateral_data_smooth'] + "smoothed.npy", allow_pickle=True)

        else:

            data_med_smooth = preprocess.smooth_data(data_patella, alpha, sr_pat, name_list_patella,
                                                              savepath=paths['medial_data_smooth'])
            data_lat_smooth = preprocess.smooth_data(data_patella, alpha, sr_pat, name_list_patella,
                                                              savepath=paths['lateral_data_smooth'])
            data_pat_smooth = preprocess.smooth_data(data_patella, alpha, sr_pat, name_list_patella,
                                                     savepath=paths['patella_data_smooth'])

    # todo
    # split data
    # write network pure cnn, rnn, lstm and mix
    # write training framework

    train_pat, test_pat, train_label_pat, test_label_pat = train_test_split(data_pat_smooth, label_patella, test_size=0.3,
                                                            random_state=42, stratify=label_patella)



    #feature extraction
    #todo
    # calculate, chroma, mfccs, simple sftf

    D = librosa.stft(data_pat_smooth[0])
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    mfccs= librosa.feature.mfcc(y=data_pat_smooth[0], sr=sr_pat[0], n_mfcc=40)  # play around with amount of features


    # plot normal data
    plt.figure()
    librosa.display.waveshow(data_pat_smooth[0], sr=sr_pat[0], marker='.', label="full_signal")
    plt.title(name_list_patella[0])
    plt.show()

    # spectogram librosa
    # But if you try to compute a 512-point FFT over a sequence of length 1000, MATLAB will take only the first 512 points and truncate the rest. If you try to compare between a 1024 point FFT and a 2056-point FFT over a [1:1000], you will get a similar plot.
    # So the moral: choose your N to be greater than or equal to the length of the sequence.
    fig, ax = plt.subplots()
    D = librosa.stft(data_pat_smooth[0])
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="spectrogram" + name_list_patella[0])
    plt.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

    # chroma form
    chroma = librosa.feature.chroma_stft(y=data_pat_smooth[0], sr=sr_pat[0])
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title="chroma features")
    plt.show()

    # stft
    amp = 0.25
    f, t, Zxx = signal.stft(data_pat_smooth[0], sr_pat[0], nperseg=5)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

if __name__ == '__main__':
    main()

######################### graveyard

# quick check for sessions in file_list und official session list
# elements = [element[0:11] for element in file_list_patella]
# myset = list(set(elements))
# myset.sort()
# diff = list(set(session) - set(myset))
# diff.sort()


# #mel_spec makes no sense, this is not human speech
# spec = librosa.feature.melspectrogram(y=file_data_patella_smooth[0], sr=samplerate_data[0])
# fig, ax = plt.subplots()
# img = librosa.display.specshow(spec, y_axis='mel', x_axis='time', ax=ax)
# ax.set(title="spectrogram display")
# fig.colorbar(img, ax=ax)
# plt.show()
# plt.close()
