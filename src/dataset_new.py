import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import feature_extraction as fe
from sklearn.model_selection import train_test_split

# W:anger, L:boredom, E:disgust, A:fear, F:happiness, T:sadness, N:neutral
emo_label_ber = {'W': 0, 'L': 1, 'E': 2, 'A': 3, 'F': 4, 'T': 5, 'N': 6}
gender_label = {'03': 0, '08': 1, '09': 1, '10': 0, '11': 0, '12': 0, '13': 1, '14': 1, '15': 0, '16': 1}


def csv_dataset(dataset, output, pick=(0, 0, 0, 1, 1), 
                win_ms=0.025, overlap=0.015, mode=0, preemp=0.97):
    dbpath = ''
    if dataset == 'Berlin':
        dbpath = '../db/berdb/'
    else:
        dbpath = '../db/ravdb_normal/'
    files = os.listdir(dbpath)
    labels = fe.allLabels(pick = pick)
    file_name = []
    emo_labels = []
    gen_labels = []
    result = []
    for fn in range(len(files)):
        frames = fe.Signal(dbpath + files[fn], win_ms, overlap, mode, preemp)
        features = frames.pickFeatures(pick)
        result.append(features)
        if dataset == 'Berlin':
            emo_labels.append(emo_label_ber[filter(lambda x: x.isupper(), files[fn])])
            gen_labels.append(gender_label[files[fn][:2]])
        else:
            emo_labels.append(int(files[fn].split('.')[0].split('-')[2]))
            gen_labels.append(0 if int(files[fn].split('.')[0].split('-')[-1]) & 1 else 1)
        file_name.append(files[fn])
      
    result = np.asarray(result)
    result = pd.DataFrame(result, columns = labels)
    result['emotion'] = emo_labels
    result['gender'] = gen_labels
    result['file'] = file_name

    result.to_csv(output, sep = ',', encoding = 'utf-8')    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, help='input dataset(Berlin or RAV)')
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('-f', '--features', help='feature selection')
    args =vars(parser.parse_args())
    dataset = str(args['dataset'])
    if dataset not in ('Berlin', 'RAV'):
        Exception('Wrong dataset! Please input Berlin or RAV!')
    output_name = str(args['output'])
    pick = (0, 0, 0, 1, 1)
    if args['features'] != None:
            pick = tuple([int(i) for i in list(args['features'])])
    csv_dataset(dataset, output_name + '.csv', pick) 

