#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:12:15 2019

@author: carsault
"""

#%%
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
import os, errno
from progress.bar import Bar
import copy

def createDatasetFull(name):
    with open(name, 'rb') as pickle_file:
        test = pickle.load(pickle_file)
    return test

#%%    
class randomTranspose(object):
    """OneHot the sample.
    """
    def __init__(self, dictChord, listChord, args):
        self.dictChord = dictChord
        self.listChord = listChord
        self.args = args
    def __call__(self, X, y):
        y = self.listChord[y]
        #print(y)
        #t = np.random.randint(25)-12
        t = np.random.randint(9)-4 #maximum of 4 semi-tone of difference
        #print(chordUtil.reduChord(y,self.args.alpha, t))
        X = utils.transpCQTFrame(X, self.args.bin_semitone, t)
        y = torch.tensor(self.dictChord[chordUtil.reduChord(y,self.args.alpha, t)])
        #print(t)
        return (X, y, t)
    
def datasetSaved(filenames,nameFold,sizeOfPart,dictFilename, args, dictChord, listChord,transf = None, debug=False, padding = True):
    listX = []
    listy = []
    part = 0
    lenSubset = 0
    #from tqdm import tqdm --> change for tqdm ?
    #for i in tqdm(range(10000)):
    initdict = copy.deepcopy(dictFilename)
    with Bar('Processing', max=len(sizeOfPart)) as bar:
        for i in sizeOfPart:
            bar.next()
            size = 0
            name = i
            size = len(dictFilename[name])
            if debug :
                print(size)
            cqt = np.load("datas/processed_CQT_data/"+ name + ".npy")
            lab = np.load("datas/processed_CQT_data/"+ name + "_lab.npy")
            randElement = np.random.randint(len(dictFilename[name]))
            start = dictFilename[name][randElement]
            dictFilename[name].pop(randElement)
            #start = np.random.randint(len(cqt)-15)
            print(name)
            if debug :
                print(name)

            if padding:
                x = cqt[start:start+15]
                y = lab[start+6]
                listX.append(x)
                listy.append(y)

            else:
                if start > 16 and start < (len(initdict[name])-16):
                    x = cqt[start:start+15]
                    y = lab[start+6]
                    listX.append(x)
                    listy.append(y)
                else:
                    print("bad padding")
                    print("start :" + str(start))
                    print("len dict :" + str(len(initdict[name])))



            #if lenSubset%10000 == 9999: #set a value to dertime size of "big batch"
            if lenSubset%5000 == 4999:
                Xfull = torch.tensor(listX)
                yfull = torch.tensor(listy)
                
                try:
                    os.mkdir("datas/" + args.dataFolder )
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
                    pass
                try:
                    os.mkdir("datas/" + args.dataFolder +'/' + nameFold)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
                    pass
                
                sauv = open("datas/" + args.dataFolder +'/' + nameFold +"/"+ str(part) +".pkl","wb")  
                pickle.dump(ACETensorDataset(Xfull, yfull, dictChord, listChord, transf),sauv)
                sauv.close()
                part += 1
                listX = []
                listy = []
            lenSubset += 1
            
        
    Xfull = torch.tensor(listX)
    yfull = torch.tensor(listy)
    
    try:
        os.mkdir("datas/" + args.dataFolder )
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    sauv = open("datas/" + args.dataFolder +'/' + nameFold +"/"+str(part) +".pkl","wb")  
    pickle.dump(ACETensorDataset(Xfull, yfull, dictChord, listChord, transf),sauv)
    sauv.close()
    
class ACETensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, X, y, dictChord, listChord, transform = None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        #self.tensors = tensors
        self.X = X
        self.y = y
        self.dictChord = dictChord
        self.listChord = listChord
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            #X, y = tuple(tensor[index] for tensor in self.tensors)
            
            X, y, t = self.transform(self.X[index], self.y[index])
            return (X, y, t)
        else:
           # return tuple(tensor[index] for tensor in self.tensors)
           return (self.X[index], self.y[index].long())

    def __len__(self):
        return self.X.size(0) 