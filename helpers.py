# misc functions to help filter, sort, process data
from workstationfile import *
import os.path
import shutil
from pathlib import Path
import numpy as np


def sortdata(savePath):

    full_file_path = savePath+"individual movements/"

    file_list = os.listdir(full_file_path)

    approved = ["wav"]
    #file_list[:] = [w for w in file_list if any(sub in w for sub in approved)]
    patella_data = [s for s in file_list if 'Patella' in s]
    tibiaMedial_data = [s for s in file_list if 'Medial' in s]
    tibiaLateral_data = [s for s in file_list if 'Lateral' in s]

    target_patella = savePath + "patella_individual_movements/"
    target_medial = savePath + "medial_individual_movements/"
    target_lateral = savePath + "lateral_individual_movements/"

    Path(target_patella).mkdir(parents=True, exist_ok=True)
    Path(target_medial).mkdir(parents=True, exist_ok=True)
    Path(target_lateral).mkdir(parents=True, exist_ok=True)

    for i in patella_data:
        shutil.copy2(full_file_path + i, target_patella)

    for i in tibiaMedial_data:
        shutil.copy2(full_file_path + i, target_medial)

    for i in tibiaLateral_data:
        shutil.copy2(full_file_path + i, target_lateral)


def extend_labels(label_list, file_list, session):
    position = 1
    session.sort()
    file_list.sort()
    for sesh in session:
        print(sesh)
        segment_number = len([s for s in file_list if sesh in s])
        extending = segment_number -1 # label for 1 already exists
        active_label = label_list[position]
        label_list = np.insert(label_list, position, np.full(extending, active_label))
        position += extending + 1

    return label_list


def exponential_smoothing(data, alpha = 0.99):
    data_smooth = np.zeros(data.shape())

    for k in range(2,data.shape(0)):
        data_smooth[k] = alpha * data_smooth[k-1] + (1-alpha)*data[k]
