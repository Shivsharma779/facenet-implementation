
import os
import random
import pickle
import math
import torch
import numpy as np


def getLabelsFromFilesVgg(rootFolderPath):
    classes = os.listdir(rootFolderPath)
    fileLabels = {}
    for _class in classes:
        files = os.listdir(os.path.join(rootFolderPath, _class))
        fileLabels[_class] = np.array([os.path.join(_class, f) for f in files])
    return fileLabels


def getMappingVGG(rootFolderPath):
    with open("classList.pickle", "rb") as f:
        _classes = pickle.load(f)
    _classes = _classes[:150]
    labelToImageMapping = getLabelsFromFilesVgg(rootFolderPath)
    labelToImageMappingFinal = {}

    for i in _classes:
        labelToImageMappingFinal[i] = labelToImageMapping[i]

    return labelToImageMappingFinal

def getMappingLFW(rootFolderPath):
    labelToImageMappingLFW = getLabelsFromFilesVgg(rootFolderPath)
    labelToImageMappingLFW = {key: labelToImageMappingLFW[key] for key in labelToImageMappingLFW if len(labelToImageMappingLFW[key]) >= 3}
    return labelToImageMappingLFW


def splitIntoTrainValid(labelToImageMapping, splitRatio):
    trainSetMapping = labelToImageMapping.copy()
    validSetMapping = {}
    for i in trainSetMapping:
        setLen = trainSetMapping[i].shape[0]
        validRange = math.floor(setLen * splitRatio)
        np.random.shuffle(trainSetMapping[i])
        validSetMapping[i] = trainSetMapping[i][validRange:]
        trainSetMapping[i] = trainSetMapping[i][:validRange]
    
    return trainSetMapping, validSetMapping