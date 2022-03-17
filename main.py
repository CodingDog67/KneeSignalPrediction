# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# todo read The new descriptor in processing of vibroacoustic signal of knee joint
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7506694/#B14-sensors-20-05015
import numpy as np

import preprocess
from workstationfile import return_data_locs
from preprocess import *
from helpers import *
import librosa
import librosa.display
import torch

preprocess_data = False
filter_data = True
sorting = False

# 1 for großhadern, 2 for home, 3 for laptop, 4 for server
paths = return_data_locs(2)

# look at thsi : https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/blob/master/VoiceAssistant/speechrecognition/neuralnet/train.py


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


    # filter data to get rid of outliers
    file_data_patella, samplerate_data = read_final_data(paths['patella_data'])

    if filter_data:
        if os.path.isfile(paths['patella_data_smooth']+"smoothed.npy"):
            file_data_patella_smooth = np.load(paths['patella_data_smooth']+"smoothed.npy", allow_pickle=True)
        else:
            file_data_patella_smooth = preprocess.smooth_data(file_data_patella, alpha, samplerate_data, name_list_patella,
                                   savepath=paths['patella_data_smooth'])

    #todo
    # split data
    # load all data
    # write network pure cnn, rnn, lstm and mix 
    # write training framework

    torch.utils.data.random_split


    # plot normal data
    plt.figure()
    librosa.display.waveshow(file_data_patella_smooth[0], sr=samplerate_data[0], marker='.', label="full_signal")
    plt.title(name_list_patella[0])
    plt.show()

    # spectogram librosa
    #But if you try to compute a 512-point FFT over a sequence of length 1000, MATLAB will take only the first 512 points and truncate the rest. If you try to compare between a 1024 point FFT and a 2056-point FFT over a [1:1000], you will get a similar plot.
    #So the moral: choose your N to be greater than or equal to the length of the sequence.
    fig, ax = plt.subplots()
    D = librosa.stft(file_data_patella_smooth[0])
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="Higher time and frequency resolution")
    plt.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

    # chroma form
    chroma = librosa.feature.chroma_stft(y=file_data_patella_smooth[0], sr=samplerate_data[0])
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    plt.show()

    #mel_spec makes no sense, this is not human speech
    spec = librosa.feature.melspectrogram(y=file_data_patella_smooth[0], sr=samplerate_data[0])
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title="spectrogram display")
    fig.colorbar(img, ax=ax)
    plt.show()
    plt.close()

    #stft
    amp = 0.25
    f, t, Zxx = signal.stft(file_data_patella_smooth[0], samplerate_data[0], nperseg=5)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    #mel_spec
    spec = librosa.feature.melspectrogram(y=file_data_patella_smooth[0], sr=samplerate_data[0], S=None, n_fft=2048,
                                           hop_length=50, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0)



if __name__ == '__main__':
    main()



######################### graveyard

    # quick check for sessions in file_list und official session list
    # elements = [element[0:11] for element in file_list_patella]
    # myset = list(set(elements))
    # myset.sort()
    # diff = list(set(session) - set(myset))
    # diff.sort()