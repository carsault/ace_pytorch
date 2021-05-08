# ace_pytorch
Pytorch implementation for training of Automatic Chord Extraction models

TRAINING STEPS:

1 - Place .wav and .lab in Datas/audio_data and Datas/labels_data accordingly
2 - Run "transfData.py -alpha" in oder to process file to CQT and associated label (choose the chord alphabet -> alpha = [a0,a2,a5])
3 - Run createDataset.py in order to create the datasplit (train/valid/test), random transposition is set for train dataset.
4 - Run train.py with corresponding arguments

