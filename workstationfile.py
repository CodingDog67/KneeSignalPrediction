# 1 f2lab
# 2 home
# 3 laptop
# 4 server

filePath = ''
labelPath = ''


def return_data_locs(location):

    # data locations f2lab
    if location == 1:
        paths = {
            "filePath": '/media/ari/Harddisk/knee data/knee signal/Patienten/',
            "labelPath": '/media/ari/Harddisk/knee data/knee signal/',
            "savePath": '/media/ari/Harddisk/knee data/knee signal/results/',
            "patella_data":  '/media/ari/Harddisk/knee data/knee signal/results/patella_individual_movements/',
            "medial_data": '/media/ari/Harddisk/knee data/knee signal/results/medial_individual_movements/',
            "lateral_data": '/media/ari/Harddisk/knee data/knee signal/results/lateral_individual_movements/',
        }

    # data locations home
    if location == 2:
        paths = {
            "filePath": 'D:\\knee signal project\\daten\\Patienten\\',
            "labelPath": 'D:\\knee signal project\\',
            "savePath": 'D:\\knee signal project\\daten\\results\\',
            "patella_data":  'D:\\knee signal project\\daten\\results\\patella_individual_movements\\',
            "medial_data":  'D:\\knee signal project\\daten\\results\\medial_individual_movements\\',
            "lateral_data": 'D:\\knee signal project\\daten\\results\\lateral_individual_movements\\',
        }

    # data locations laptop

    # data locations server

    return paths