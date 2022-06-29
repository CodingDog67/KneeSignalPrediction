import pandas as pd
import os.path
import matplotlib.pyplot as plt
from scipy.io import wavfile
from vaghelpers import vag2float
from batch_segmentation import *
from pathlib import Path
import librosa
import numpy as np
import matplotlib.lines as mlines
import os.path as path


class Data():

    def __init__(self, filepath:str, savepath:str, plotting: False ):
        self.filepath = filepath
        self.savepath = savepath
        self.plotting = plotting
        self.sessionlist = sorted(os.listdir(self.file_path))