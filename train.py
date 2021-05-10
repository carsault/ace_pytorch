#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:48:17 2019

@author: carsault
"""

#%%
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data.dataset import *
from torch.utils import data
import pickle
from sklearn.model_selection import train_test_split
import random
from utilities import chordUtil
from utilities import ACEdataImport
from utilities.chordVocab import *
from utilities import utils
from utilities import ACEmodels
from progress.bar import Bar
import os, errno
import glob

#%%
"""
###################

Argument parsing

###################
"""
parser = argparse.ArgumentParser(description='Hierarchical Latent Space')
# General
parser.add_argument('--dataFolder',   type=str,   default='a0_124_123456',    help='name of the data folder')
parser.add_argument('--batch_size',      type=int,   default=200,                                help='batch size (default: 50)')
parser.add_argument('--alpha',      type=str,   default='a0',                            help='type of alphabet')
parser.add_argument('--latent',     type=int,   default=50,                                 help='size of the latent space (default: 50)')
parser.add_argument('--hidden',     type=int,   default=500,                                 help='size of the hidden layer (default: 500)')
parser.add_argument('--modelType',      type=str,   default='cnn',                            help='type of model to evaluate')
parser.add_argument('--layer',     type=int,   default=1,                                 help='number of the hidden layer - 2 (default: 1)')
parser.add_argument('--dropRatio',     type=float,   default=0.5,                                 help='drop Out ratio (default: 0.5)')
parser.add_argument('--device',     type=str,   default="cuda",                              help='set the device (cpu or cuda, default: cpu)')
parser.add_argument('--epochs',     type=int,   default=20000,                                help='number of epochs (default: 15000)')
parser.add_argument('--lr',         type=float, default=2e-5,                               help='learning rate for Adam optimizer (default: 2e-4)')
parser.add_argument('--random_state',   type=int,   default=123456,    help='seed for the random train/test split')
# Save file
parser.add_argument('--foldName',      type=str,   default='modelSave190515',                            help='name of the folder containing the models')
parser.add_argument('--modelName',      type=str,   default='bqwlbq',                            help='name of model to evaluate')
parser.add_argument('--dist',      type=str,   default='euclidian',                            help='distance to compare predicted sequence (default : euclidian')
args = parser.parse_args()
print(args)

dictChord, listChord = chordUtil.getDictChord(eval(args.alpha))
n_categories = len(listChord)

args.dataFolder = args.alpha + "_ACE_" + str(args.random_state)
#args.dataFolder = "test"

args.modelName = args.dataFolder + "_" + args.modelType

# Create save folder
try:
    os.mkdir(args.foldName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir(args.foldName + '/' + args.modelName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# CUDA for PyTorch
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor

#if args.device is not torch.device("cpu"):
if torch.cuda.is_available():
    print(args.device)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark=True


#%%
# Create dataset

#dataset_valid = ACEdataImport.createDatasetFull("datasets/" + args.dataFolder + "/valid.pkl")
#dataset_test = ACEdataImport.createDatasetFull("datasets/" + args.dataFolder + "/test.pkl")

# Create generators
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 6}


#validating_generator = data.DataLoader(dataset_valid, pin_memory = True, **params)
#testing_generator = data.DataLoader(dataset_test, pin_memory = True, **params)

#%%
if args.modelType == "mlp":
    net = ACEmodels.MLP(15, 105, 50, 25, 1, 1)
elif args.modelType == "cnn":
    net = ACEmodels.ConvNet(args)
else:
    print("Not known model")

# Print model 
print(net)
#f.write(print(net))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(net))

if args.device is not "cpu":
    net.to(args.device)




# Compute weight_vector
weight_vector = torch.zeros(len(listChord))

trainFiles = glob.glob("datas/" + args.dataFolder + "/train/*.pkl")
part = 0
print("Compute weight_vector")
for testF in trainFiles:
    dataset_train = ACEdataImport.createDatasetFull(testF)
    training_generator = data.DataLoader(dataset_train, pin_memory = True, **params)
    part += 1
    #print("part : " + str(part) + " over " + str(len(trainFiles)))
    for local_batch, local_labels, local_transp in training_generator:
        #compute the weight vector -> do this before to train the network on a specific dataset then copy paste the obtained wieight_vector
        for i in range(len(local_labels)):
            weight_vector[local_labels[i]] += 1
print(weight_vector)
# Do a "fake" training to compute this weight_vector before to run the training (see in the training loop)
#weight_vector = [189545.,  51882., 154889.,  45571., 182577.,  42433., 196305.,  49910.,
#        187021.,  49019., 183968.,  49045., 151922.,  47066., 165120.,  42766.,
#        197705.,  46390., 197010.,  51775., 197793.,  48258., 173593.,  47643.,
#        148356.]

weight_vector = [1/i for i in weight_vector]
num = sum(weight_vector)
weight_vector = [i/num for i in weight_vector]
class_weights = torch.FloatTensor(weight_vector).to(args.device)
criterion = nn.CrossEntropyLoss(class_weights)

    
# choose optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)

# Begin training
for epoch in range(args.epochs):
    print('Epoch number {} '.format(epoch))
    trainFiles = glob.glob("datas/" + args.dataFolder + "/train/*.pkl")
    train_total_loss = 0
    part = 0
    print("Training phase")
    with Bar('Processing', max=len(trainFiles)) as bar:
        for testF in trainFiles:
            bar.next()  
            dataset_train = ACEdataImport.createDatasetFull(testF)
            training_generator = data.DataLoader(dataset_train, pin_memory = True, **params)
            part += 1
            #print("part : " + str(part) + " over " + str(len(trainFiles)))
            for local_batch, local_labels, local_transp in training_generator:
                if args.modelType == "cnn":
                    local_batch, local_labels = local_batch.transpose(1,2).view(len(local_batch),1,105,15).to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                local_batch, local_labels = local_batch.to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True) 
                net.train() 
                net.zero_grad()
                output = net(local_batch)
                #print(output)
                loss = criterion(output, local_labels)
                loss.backward()
                optimizer.step()
                train_total_loss += loss
    print(train_total_loss)
    print(weight_vector)
    
    validFiles = glob.glob("datas/" + args.dataFolder + "/valid/*.pkl")
    correct = 0
    total = 0
    print("Validating phase")
    with Bar('Processing', max=len(validFiles)) as bar: 
        for testF in validFiles:
            bar.next()
            dataset_valid = ACEdataImport.createDatasetFull(testF)
            validating_generator = data.DataLoader(dataset_valid, pin_memory = True, **params)
            for local_batch, local_labels in validating_generator:
                if args.modelType == "cnn":
                    local_batch, local_labels = local_batch.transpose(1,2).view(len(local_batch),1,105,15).to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                local_batch, local_labels = local_batch.to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                with torch.no_grad():
                    net.eval() 
                    net.zero_grad()
                batchcorrect, batchtotal = chordUtil.accuracy_quick(net, local_batch, local_labels)
                correct += batchcorrect.item()
                total += batchtotal
    accurValid = (correct * 100.0 /total)
    print("valid acc = " + str(accurValid))
            