#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:31:58 2019

@author: carsault
"""
#%%
import argparse
import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data.dataset import *
from torch.utils import data
import pickle
from sklearn.model_selection import train_test_split
import random
from utilities import chordUtil
from utilities.chordVocab import *
from utilities import utils
from utilities import ACEdataImport

parser = argparse.ArgumentParser(description='ACE dataset')
parser.add_argument('--dataFolder',   type=str,   default='okcomputer',    help='name of the data folder')
parser.add_argument('--alpha',      type=str,   default='a0', help='type of alphabet')
parser.add_argument('--random_state',   type=int,   default=123456,    help='seed for the random train/test split')
args = parser.parse_args()
print(args)

args.dataFolder = args.alpha + "_ACE_" + str(args.random_state)

dictChord, listChord = chordUtil.getDictChord(eval(args.alpha))

# List files
filenames = os.listdir("datas/processed_CQT_data")
filenames.remove(".DS_Store")
filenames = [ x for x in filenames if "_lab.npy" not in x ]
filenames = [ x.replace('.npy','')for x in filenames]
filenames.sort()

sizeOfPartTrain = []
sizeOfPartValid = []
sizeOfPartTest = []


# Create datasets
files_train ,files_test = train_test_split(filenames,test_size=0.2,random_state=args.random_state)
files_test ,files_valid = train_test_split(files_test,test_size=0.5,random_state=args.random_state)

dictFilenameTrain = {}
for i in files_train:
    listOfFrame  = []
    cqt = np.load("datas/processed_CQT_data/"+ i + ".npy")
    listOfFrame = list(range(len(cqt)-15))
    random.shuffle(listOfFrame)
    dictFilenameTrain[i] = listOfFrame
    for j in range(len(listOfFrame)):
        sizeOfPartTrain.append(i)
    #print(list(range(len(cqt)-15)))
#%%    
dictFilenameValid = {}
for i in files_valid:
    listOfFrame  = []
    cqt = np.load("datas/processed_CQT_data/"+ i + ".npy")
    listOfFrame = list(range(len(cqt)-15))
    random.shuffle(listOfFrame)
    dictFilenameValid[i] = listOfFrame
    for j in range(len(listOfFrame)):
        sizeOfPartValid.append(i)
    #print(list(range(len(cqt)-15)))    
    
dictFilenameTest = {}
for i in files_test:
    listOfFrame  = []
    cqt = np.load("datas/processed_CQT_data/"+ i + ".npy")
    listOfFrame = list(range(len(cqt)-15))
    random.shuffle(listOfFrame)
    dictFilenameTest[i] = listOfFrame
    for j in range(len(listOfFrame)):
        sizeOfPartTest.append(i)
    #print(list(range(len(cqt)-15)))
#%%
 

random.shuffle(sizeOfPartTrain)
random.shuffle(sizeOfPartValid)
random.shuffle(sizeOfPartTest)
randTransp = ACEdataImport.randomTranspose(dictChord, listChord, args) 
#randTransp = None
print("Process train dataset")
ACEdataImport.datasetSaved(files_train, "train", sizeOfPartTrain, dictFilenameTrain, args, dictChord, listChord, randTransp)
print("Process valid dataset")
ACEdataImport.datasetSaved(files_valid, "valid", sizeOfPartValid, dictFilenameValid, args, dictChord, listChord)
print("Process test dataset")
ACEdataImport.datasetSaved(files_test, "test", sizeOfPartTest, dictFilenameTest, args, dictChord, listChord)