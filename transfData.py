#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:31:58 2019

@author: carsault
"""
#%%
import argparse
import numpy as np
from librosa.core import cqt,load,note_to_hz
from librosa.util import find_files
from utilities.utils import LoadLabelArr
from utilities import chordUtil
from utilities.chordVocab import *
#import const as C
import os

parser = argparse.ArgumentParser(description='ACE dataset')
parser.add_argument('--alpha',      type=str,   default='a0', help='type of alphabet')
args = parser.parse_args()
print(args)
dictChord, listChord = chordUtil.getDictChord(eval(args.alpha))
#%%

audiolist = find_files("datas/audio_data")
sr = 44100
path_hcqt = "datas/processed_CQT_data"
itemcnt = len(audiolist)
i = 0
#%%
for audiofile in audiolist:
    labelfile = audiofile.replace('.wav','.lab')
    labelfile = labelfile.replace('audio_data','labels_data')
    labelfile = labelfile.replace(' ','_')
    labelfile = labelfile.replace(',','')
    #labelfile = labelfile.replace('.','._')
    i += 1
    print("Processing %d/%d" % (i,itemcnt))    
    wav,sr = load(audiofile,sr=sr)
    print("Sampling rate : " + str(sr))
    #fmin = note_to_hz("C2")
    fmin = 65
    #spec = np.stack([np.abs(cqt(wav,sr=C.SR,hop_length=512,n_bins=C.BIN_CNT,bins_per_octave=C.OCT_BIN,fmin=fmin*(h+1),filter_scale=2,tuning=None)).T.astype(np.float32) for h in range(C.CQT_H)])
    spec = np.abs(cqt(wav,sr=sr,hop_length=512*4,n_bins=105,bins_per_octave=24,fmin=fmin,filter_scale=2,tuning=None)).T.astype(np.float32)
    print("Number of frames : "+ str(len(spec)))
    print("CQT bins : " + str(len(spec[0])))
    lab = LoadLabelArr(labelfile,dictChord,args,512*4)
    filename = audiofile.split('/')[-1].split(".")[0]
    savepath = os.path.join(path_hcqt,filename+".npy")
    np.save(savepath,spec)
    savepath = os.path.join(path_hcqt,filename+"_lab.npy")
    np.save(savepath,lab)

#%%