import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import feature_extraction as fe
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from svmutil import *

# W:anger, L:boredom, E:disgust, A:fear, F:happiness, T:sadness, N:neutral
emotion_label = {'W': 0, 'L': 1, 'E': 2, 'A': 3, 'F': 4, 'T': 5, 'N': 6}
sex_label = {'03': 0, '08': 1, '09': 1, '10': 0, '11': 0, '12': 0, '13': 1, '14': 1, '15': 0, '16': 1}
dbpath = '../db/wav/'
training = []
testing = []

def emo_dataset(output, ext='lsvm', pick=(0, 0, 0, 1, 1), speaker='00', 
                win_ms=0.025, overlap=0.015, mode=2, preemp=0.97):
    #TODO
    num_tr = len(training)
    num_te = len(testing)

    if ext == 'lsvm':
        train = ''
        test = ''
        for fn in range(num_files):
            frames = fe.Signal(dbpath + files[fn], win_ms, overlap, mode, preemp)
            features = frames.pickFeatures(pick)
            num_fea = len(features)
            label = emotion_label[filter(lambda x: x.isupper(), files[fn])]
            if files[fn][:2] == speaker:
                test += str(label) + ' '
                for i in range(num_fea):
                    test += str(i+1) + ':' + str(features[i]) + ' '
                test += '\n'
            else:
                train += str(label) + ' '
                for i in range(num_fea):
                    train += str(i+1) + ':' + str(features[i]) + ' '
                train += '\n'
        
        training = output.split('.')[0] + '_' + speaker +'_training'
        testing = output.split('.')[0] + '_' + speaker +'_testing'
        with open(training, 'w') as f:
                f.write(train)
        if speaker in ('03','08','09','10','11','12','13','14','15','16'):
            with open(testing, 'w') as f:
                f.write(test)
        else:
            splitDataset(training)

    elif ext == 'csv':
        labels = fe.allLabels(pick = pick)
        file_name = []
        emo_labels = []
        result = []
        for fn in range(num_files):
            frames = fe.Signal(dbpath + files[fn], win_ms, overlap, mode, preemp)
            features = frames.pickFeatures(pick)
            result.append(features)
            emo_labels.append(emotion_label[filter(lambda x: x.isupper(), files[fn])])
            file_name.append(files[fn])
      
        result = np.asarray(result)
        result = pd.DataFrame(result, columns = labels)
        result['emotion'] = emo_labels
        result['file'] = file_name

        result.to_csv(output, sep = ',', encoding = 'utf-8')
      

def sex_dataset(output, pick=(0, 0, 0, 1, 1), win_ms=0.025, overlap=0.015, mode=2, preemp=0.97):
    #TODO
    num_tr = len(training)
    num_te = len(testing)

    train = ''
    test = ''
    for fn in range(num_tr):
        frames = fe.Signal(dbpath + training[fn], win_ms, overlap, mode, preemp)
        features = frames.pickFeatures(pick)
        num_fea = len(features)
        label = sex_label[training[fn][:2]]
        train += str(label) + ' '
        for i in range(num_fea):
            train += str(i+1) + ':' + str(features[i]) + ' '
        train += '\n'
    for fn in range(num_te):
        frames = fe.Signal(dbpath + testing[fn], win_ms, overlap, mode, preemp)
        features = frames.pickFeatures(pick)
        num_fea = len(features)
        label = sex_label[testing[fn][:2]]
        test += str(label) + ' '
        for i in range(num_fea):
            test += str(i+1) + ':' + str(features[i]) + ' '
        test += '\n'     
      
    with open(output, 'w') as f:
        f.write(train)
    splitDataset(output)   
    

def splitDataset(dataset):
    label, raw_data = svm_read_problem(dataset)
    train_data, test_data, train_label, test_label = train_test_split(raw_data, label, test_size=0.2)

    train_str = ''
    test_str = ''
    for i in range(len(train_label)):
        value = str(train_data[i])[1:-1].replace(',', '').replace(': ',':')
        train_str += str(int(train_label[i])) + ' ' + value + ' \n'
    for i in range(len(test_label)):
        value = str(test_data[i])[1:-1].replace(',', '').replace(': ',':')
        test_str += str(int(test_label[i])) + ' ' + value + ' \n'
    with open(dataset.split('.')[0] + '_trainingset', 'w') as f:       
        f.write(train_str)
    with open(dataset.split('.')[0] + '_testingset', 'w') as f:       
        f.write(test_str)


def splitData(dbpath, emotion_num=7):
    global training, testing
    files = os.listdir(dbpath)
    if emotion_num == 4:
        files = [f for f in files if f.find('W') != -1 or f.find('F') != -1 or f.find('T') != -1 or f.find('N') != -1]
    elif emotion_num == 5:
        files = [f for f in files if f.find('W') != -1 or f.find('A') != -1 or f.find('F') != -1 or f.find('T') != -1 or f.find('N') != -1]
    elif len(sys.argv) < 3:
       pass
    
    testing = random.sample(files, int(0.2*len(files)))
    training = [f for f in files if f not in testing]
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('-e', '--emotion', help='number of emotons')
    parser.add_argument('-f', '--features', help='feature selection')
    parser.add_argument('-t', '--speaker', help='speaker selection')
    parser.add_argument('-s', '--sex', help='sex separation', action='store_true')
    args =vars(parser.parse_args())
    emotion = 0
    pick = (0, 0, 0, 1, 1)
    speaker = '00'
    if args['emotion'] != None:
        emotion = args['emotion']
    if args['features'] != None:
        pick = tuple([int(i) for i in list(args['features'])])
    if args['speaker'] != None:
        speaker = str(args['speaker'])
    if args['sex'] == None:
        emo_dataset(args['output'], args['output'].split('.')[1], pick, speaker)
    else:
        sex_dataset(args['output'], pick)
   
