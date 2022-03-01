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
            "savePath": '/media/ari/Harddisk/knee data/knee signal/processed/',
            "savePath_processed": '/media/ari/Harddisk/knee data/knee signal/processed/',

            "patella_data":  '/media/ari/Harddisk/knee data/knee signal/processed/patella_individual_movements/',
            "medial_data": '/media/ari/Harddisk/knee data/knee signal/processed/medial_individual_movements/',
            "lateral_data": '/media/ari/Harddisk/knee data/knee signal/processed/lateral_individual_movements/',

            "patella_data_smooth": '/media/ari/Harddisk/knee data/knee signal/processed/smoothed/patella_individual_movements/',
            "medial_data_smooth": '/media/ari/Harddisk/knee data/knee signal/processed/smoothed/medial_individual_movements/',
            "lateral_data_smooth": '/media/ari/Harddisk/knee data/knee signal/processed/smoothed/lateral_individual_movements/',
        }



    # data locations home
    if location == 2:
        paths = {
            "filePath": 'D:\\knee signal project\\daten\\Patienten\\',
            "labelPath": 'D:\\knee signal project\\',
            "savePath_results": 'D:\\knee signal project\\daten\\results\\',
            "savePath_processed": 'D:\\knee signal project\\daten\\processed\\',

            "patella_data":  'D:\\knee signal project\\daten\\processed\\patella_individual_movements\\',
            "medial_data":  'D:\\knee signal project\\daten\\processed\\medial_individual_movements\\',
            "lateral_data": 'D:\\knee signal project\\daten\\processed\\lateral_individual_movements\\',

            "patella_data_smooth": 'D:\\knee signal project\\daten\\processed\\smoothed\\patella_individual_movements\\',
            "medial_data_smooth": 'D:\\knee signal project\\daten\\processed\\smoothed\\medial_individual_movements\\',
            "lateral_data_smooth": 'D:\\knee signal project\\daten\\processed\\smoothed\\lateral_individual_movements\\',
        }

    # data locations laptop

    # data locations server

    return paths