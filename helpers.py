# misc functions to help filter, sort, process data
import os.path
import shutil
from pathlib import Path
import numpy as np


def sortdata(savePath):

    full_file_path = savePath+"individual movements/"

    file_list = os.listdir(full_file_path)

    patella_data = [s for s in file_list if 'Patella' in s]
    tibiaMedial_data = [s for s in file_list if 'Medial' in s]
    tibiaLateral_data = [s for s in file_list if 'Lateral' in s]

    target_patella = savePath + "patella_individual_movements/"
    target_medial = savePath + "medial_individual_movements/"
    target_lateral = savePath + "lateral_individual_movements/"

    Path(target_patella).mkdir(parents=True, exist_ok=False)
    Path(target_medial).mkdir(parents=True, exist_ok=False)
    Path(target_lateral).mkdir(parents=True, exist_ok=False)

    for i in patella_data:
        shutil.copy2(full_file_path + i, target_patella)

    for i in tibiaMedial_data:
        shutil.copy2(full_file_path + i, target_medial)

    for i in tibiaLateral_data:
        shutil.copy2(full_file_path + i, target_lateral)


def extend_labels(label_list, file_list, session):

    session.sort()
    file_list.sort()
    new_label_list = []
    check_segment = []
    check_sesh = []
    for counter, sesh in enumerate(session):

        active_label = label_list[counter]
        print(counter, sesh)
        if any(sesh in file_list for file_list in file_list):
            segment_number = len([s for s in file_list if sesh in s])
            new_label_list = np.append(new_label_list, np.full(segment_number, active_label))

    return new_label_list.astype(int)





def read_data_names_in(paths):
    file_list_patella = os.listdir(paths["patella_data"])
    file_list_medial = os.listdir(paths["medial_data"])
    file_list_lateral = os.listdir(paths["lateral_data"])

    approved = ["wav"]
    # filter out all the wav files and sort by sensor (2 runs/2 files)
    file_list_patella[:] = [w for w in file_list_patella if any(sub in w for sub in approved)]
    file_list_patella.sort()

    file_list_medial[:] = [w for w in file_list_medial if any(sub in w for sub in approved)]
    file_list_medial.sort()

    file_list_lateral[:] = [w for w in file_list_lateral if any(sub in w for sub in approved)]
    file_list_lateral.sort()

    return file_list_patella, file_list_medial, file_list_lateral


