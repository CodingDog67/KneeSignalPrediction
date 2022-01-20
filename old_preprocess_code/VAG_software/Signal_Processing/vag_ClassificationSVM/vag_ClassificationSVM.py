#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------
Vibroarthrography (VAG) Project
-------------------------------
Perform my signal analysis

Author: Tuan Nam Le
Last modified: 02/06/2015 Walther Schulze
"""
from os import path
from PySide.QtGui import QFileDialog
import matplotlib.pyplot as plt
import vaghelpers###
from vaghelpers import *
reload(vaghelpers)###
from vaghelpers import *###
#from svmutil import svm_read_problem, svm_train
#from svm import svm_problem, svm_parameter, toPyModel
import numpy

from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from AllFeaturesCombinations_waltherTEST import *
from sklearn import preprocessing
from time import strftime


if __name__ == '__main__':

    absolutePathOutput = "/Users/admin/Desktop/temp/reproduce_DKOU/DKOU/VAG_DKOU_2015/Result/Result_DKOU_FLEXION_waltherTEST"
    resultFilenameTXT = "_Protocol_DKOU_FLEXION.txt"

    ########################################
    ##### CLASSIFICATION USING SKLEARN #####
    ########################################

    ##### SELECT CSV FILE #####

    PatientsResultTimeDomainFilename = "/Users/admin/Desktop/temp/reproduce_DKOU/DKOU/VAG_DKOU_2015/Result/_hacked/DKOU_Patients_TimeDomainFeatures_FLEXION.csv"
    PatientsResultFrequencyDomainFilename = "/Users/admin/Desktop/temp/reproduce_DKOU/DKOU/VAG_DKOU_2015/Result/_hacked/DKOU_Patients_FrequencyDomainFeatures_FLEXION.csv"
    HealthyResultTimeDomainFilename = "/Users/admin/Desktop/temp/reproduce_DKOU/DKOU/VAG_DKOU_2015/Result/_hacked/DKOU_Healthy_TimeDomainFeatures_FLEXION.csv"
    HealthyResultFrequencyDomainFilename = "/Users/admin/Desktop/temp/reproduce_DKOU/DKOU/VAG_DKOU_2015/Result/_hacked/DKOU_Healthy_FrequencyDomainFeatures_FLEXION.csv"

    FeaturesSelectionTD = [1,2,3,4,5,6] # 1=mean, 2=variance, 3=std, 4=rms, 5=zcr_0, 6=zcr_std
    FeaturesSelectionFD = [1,2,3,4,5,6,7,8,9] # 1=r_10_50, 2=r_25_320, 3=r_40_140, 4=r_40_140, 5=r_300_600, 6=r_500_8k, 7=r_10_100, 8=r_3k_5k, 9=r_6k_8k
    FeaturesSelectionNames = ['mean','variance','std','rms','zcr','zcr_std',
                            'r_10_50','r_25_320','r_40_140','r_50_500','r_300_600','r_500_8k','r_10_100','r_3k_5k','r_6k_8k']

    PatientsFeaturesTD = export_certain_columns_to_dataset(PatientsResultTimeDomainFilename, FeaturesSelectionTD)
    PatientsFeaturesFD = export_certain_columns_to_dataset(PatientsResultFrequencyDomainFilename, FeaturesSelectionFD)
    PatientsFeaturesALL = numpy.hstack([PatientsFeaturesTD, PatientsFeaturesFD])
    HealthyFeaturesTD = export_certain_columns_to_dataset(HealthyResultTimeDomainFilename, FeaturesSelectionTD)
    HealthyFeaturesFD = export_certain_columns_to_dataset(HealthyResultFrequencyDomainFilename, FeaturesSelectionFD)
    HealthyFeaturesALL = numpy.hstack([HealthyFeaturesTD, HealthyFeaturesFD])

    PatientsSession, PatientsRL, PatientsSensorpos = identify_session_rl_sensorpos_in_dataset(PatientsResultFrequencyDomainFilename)
    HealthySession, HealthyRL, HealthySensorpos = identify_session_rl_sensorpos_in_dataset(HealthyResultFrequencyDomainFilename)
    dummy, dummy1, dummy2 = identify_session_rl_sensorpos_in_dataset(PatientsResultTimeDomainFilename) #run on TD as well to identify bad file names
    dummy, dummy1, dummy2 = identify_session_rl_sensorpos_in_dataset(HealthyResultTimeDomainFilename) #run on TD as well to identify bad file names
    
    PatientsTargetALL = create_target_vector_with_label(PatientsFeaturesALL.shape[0], 1)
    HealthyTargetALL = create_target_vector_with_label(HealthyFeaturesALL.shape[0], 0)

    vagDataALL = numpy.vstack((PatientsFeaturesALL, HealthyFeaturesALL))
    vagTarget = numpy.concatenate((PatientsTargetALL, HealthyTargetALL))
    SessionALL = numpy.concatenate((PatientsSession, HealthySession))
    RLALL = numpy.concatenate((PatientsRL, HealthyRL))
    SensorposALL = numpy.concatenate((PatientsSensorpos, HealthySensorpos))

    # Selection of features combination
    FeaturesDict = dict(zip(FeaturesSelectionNames, numpy.arange(15)))
    FeaturesCombinationToSelect = myFeaturesSelection
    scoring = "accuracy"
    
    # Select the sensor positions
    SensorPosCombination = [0,1,2] # Patella:0, TibiaplateauMedial:1, TibiaplateauLateral:2

    # Prepare text file to write
    fout = open(os.path.join(absolutePathOutput, resultFilenameTXT), 'w+')
    fout.write("EVALUATION DKOU - Nam Le\n")
    fout.write(strftime("%Y-%m-%d %H:%M:%S") + "\n")
    fout.write("TWO CLASSES CLASSIFICATION HEALTHY vs. PATIENTS\n")
    fout.write("Classificator = SVC\n")

    counter = 1
    Top10ROCAUC = numpy.zeros(10) # numpy array
    Top10FeaturesCombination = [0] * 10 # list

    for combination in FeaturesCombinationToSelect:
        itemsCount = len(FeaturesCombinationToSelect)
        percent = 100.0 * counter/float(itemsCount)
        print str(percent)+"% - processing: combination of ("+' '.join(combination)+")"
        TOWRITE = "------------------------------------------------------------\n"
        TOWRITE += "Features combination ("+' '.join(combination)+")\n"
        FeaturesCombinationIndex = [FeaturesDict[x] for x in combination]
        vagData = vagDataALL[:, FeaturesCombinationIndex]
        
        vagDataMEASUREMENTS = []
        vagTargetMEASUREMENTS = vagTarget[0:vagDataALL.shape[0]:len(SensorPosCombination)]
        #group sensor signal features as measurement features
        for vagDataALLit in xrange(0, vagDataALL.shape[0], len(SensorPosCombination)): #step:len(SensorPosCombination)
            SensorposALLcheck = SensorposALL[vagDataALLit:vagDataALLit+len(SensorPosCombination),:]
            #all required SensorPosCombination elements in there?
            for SensorPosCombinationEL in SensorPosCombination:
                if not (SensorPosCombinationEL in SensorposALLcheck):
                    raise Exception('Not all elements of SensorPosCombination in the series of sensor measurements, Session: ' + str(SessionALL[vagDataALLit]) + ' RL: ' + str(RLALL[vagDataALLit]) )
            #If no exception, combine!
            makecombination = []
            for makecombinationIT in range(vagDataALLit,vagDataALLit+len(SensorPosCombination)):
                makecombination = numpy.hstack((makecombination, vagData[makecombinationIT,:]))
            if (len(vagDataMEASUREMENTS)==0):
                vagDataMEASUREMENTS = makecombination
            else:
                vagDataMEASUREMENTS = numpy.vstack([vagDataMEASUREMENTS, makecombination])
        print "Shape of Data Vector:"
        print vagDataMEASUREMENTS.shape
        vagDataMEASUREMENTS = preprocessing.scale(vagDataMEASUREMENTS) # test scaling http://stackoverflow.com/questions/17455302/gridsearchcv-extremely-slow-on-small-dataset-in-scikit-learn # why it should be scaled: http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
        resulttext, sensitivity, specificity, mean_auc = svm_classification_gridsearch_roc(vagDataMEASUREMENTS, vagTargetMEASUREMENTS, svm.SVC(), scoring, os.path.join(absolutePathOutput,' '.join(combination)))

        TOWRITE += resulttext + "\n"
        TOWRITE += "Sensitivity = " + str(sensitivity) + "\n"
        TOWRITE += "Specificity = " + str(specificity) + "\n"
        TOWRITE += "ROC_AUC = " + str(mean_auc) + "\n"
        TOWRITE += "- END -\n"
        if mean_auc > Top10ROCAUC[0]:
            Top10ROCAUC = numpy.roll(Top10ROCAUC,1) # Roll array elements backwards
            Top10ROCAUC[0] = mean_auc # Set first element
            Top10FeaturesCombination = shift_list(Top10FeaturesCombination,1) # Shift element of Top10FeaturesCombination
            Top10FeaturesCombination[0] = ' '.join(combination) # Set first element
        print Top10ROCAUC
        print Top10FeaturesCombination
        fout.write(TOWRITE)
        fout.flush()
        counter+=1

    fout.write("------------------------------------------------------------\n")
    fout.write("Best ROC_AUC = " + str(Top10ROCAUC[0]) + "\n")
    fout.write("Best Features Combination = (" + Top10FeaturesCombination[0] + ")\n")
    fout.close()
