from __future__ import print_function
import os
import sys
import cPickle
import argparse
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#emotion_label = {0: 'Anger', 1: 'Boredom', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Sadness', 6: 'Neutral'}


def eval_model(train_std, train_label, test_std, test_label):
   
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(decision_function_shape = 'ovr'), tuned_parameters, cv = 10,
                       scoring = '%s_macro' % score)
        model = clf.fit(train_std, train_label)
        print(model)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Accuracy: %f" % clf.score(test_std, test_label))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        test_label, pred_label = test_label, clf.predict(test_std)
        print(classification_report(test_label, pred_label))
        print()


def random_split(dataframe, ratio=0.8):
    train = dataframe.sample(frac=ratio)
    test = dataframe.loc[~dataframe.index.isin(train.index), :]
    return train, test


def independent_split(dataframe, language, speaker):
    test = None
    if language == 1:
        test = dataframe.loc[dataframe['file'].str[:2] == speaker]
    else:
        test = dataframe.loc[dataframe['file'].str[-6:-4] == speaker]
    train = dataframe.loc[~dataframe.index.isin(test.index), :]
    return train, test


def plot_confusion_matrix(language, y_true, y_pred):
    cnm = confusion_matrix(y_true,y_pred)
    row_sum = cnm.sum(axis=1)
    total = sum(row_sum)
    correct = 0
    emotion = []
    if language == 1:
        emotion = ["Anger","Boredom","Disgust","Fear","Happiness","Sadness","Neutral"]
    else:
        emotion = ["Neutral","Calm","Happy","Sad","Angry","Fearful","Disgust","Surprised"]
    for i in range(cnm.shape[0]):
        if row_sum[i] == 0:
            print('No data')
        else:
            print(emotion[i],"accuracy rate:",cnm[i][i]*100.0/row_sum[i],"%")
        correct += cnm[i][i]*1.0
    acc = correct / total
    print("Accuracy: ", acc)
    print("\nConfusion matrix\n=================")
    print(cnm)
    return acc, cnm


def final_model(dataframe, output):
    output_path = '../model'
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    train_label = dataframe.loc[:,'emotion']
    train_data = dataframe.iloc[:,0:-3]
    
    sc1 = MinMaxScaler(feature_range=(-1, 1))
    sc1.fit(train_data)
    train_std = sc1.transform(train_data)
    
    model = GridSearchCV(SVC(decision_function_shape='ovr'), tuned_parameters, cv=10)
    model.fit(train_std, train_label)
    
    cPickle.dump(model, open(os.path.join(output_path, output+'_model.pkl'), 'wb'))
    cPickle.dump(sc1, open(os.path.join(output_path, output+'_range.pkl'), 'wb'))
    

def independent_train(dataframe, language):
    #best k = 165
    global final_result, final_pred
    opt = 0
    opt_acc = 0.0
    #for topk in range(1,205):
    #print('k = ', topk)
    speakers = None
    if language == 1:
        speakers = ('03', '08', '09', '10', '11', '12', '13', '14', '15', '16') 
    else: 
        speakers = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24')
        
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                       ] 
    
    train_time = np.zeros(2)
    for para in range(len(tuned_parameters)):
        final_result = []
        final_pred = []
        print('kernel')
        for speaker in speakers:
            print(speaker+':')    
            train, test = independent_split(dataframe, language, speaker)
            feature_selector = SelectPercentile(score_func=f_classif, percentile=80)
            train_data, emo_train_label = train.iloc[:, 0:-3], train.loc[:,'emotion']
            
            test_data, emo_test_label = test.iloc[:, 0:-3], test.loc[:,'emotion']
            
            sc1 = MinMaxScaler(feature_range=(-1, 1))
            sc1.fit(train_data)
            
            train_std = sc1.transform(train_data)
            feature_selector.fit(train_std, emo_train_label)
            features = feature_selector.transform(train_std)
            
            start = timer()       
            model = GridSearchCV(SVC(decision_function_shape = 'ovr'), tuned_parameters[para], cv=10)
            model.fit(features, emo_train_label)
            end = timer()
            
            test_std = sc1.transform(test_data)
            test_std = test_std[:,feature_selector.get_support()]
            pred_label = model.predict(test_std)
            #print(classification_report(emo_test_label, pred_label))
            for i in emo_test_label:
                final_result.append(i)
            for i in pred_label:
                final_pred.append(i)
            plot_confusion_matrix(language, emo_test_label, pred_label)
            train_time[para] += end-start
            
        final_result = np.asarray(final_result)
        final_pred = np.asarray(final_pred)
        acc, cnm = plot_confusion_matrix(language, final_result, final_pred)
        print(acc)
        print(cnm * 100.0/(cnm.sum(axis=1)[:,None]*1.0))
    train_time /= len(speakers)
    print(train_time)        

def random_train(dataframe, language, ratio, time, output):
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    train_time = np.zeros(2)
    avg_acc = np.zeros(2)
    max_acc = 0.0
    opt_model = None
    opt_range = None
    test_file = []
    emo_num = 7 if language == 1 else 8
    matrix = np.zeros((2, emo_num, emo_num))
    for i in range(time):
        print('time',i+1,':')   
        train, test = random_split(dataframe, ratio)
        for para in range(len(tuned_parameters)):
            print('Kernel') 
            feature_selector = SelectPercentile(score_func=f_classif, percentile=100)
            train_data, emo_train_label = train.iloc[:, 0:-3], train.loc[:,'emotion']
            test_data, emo_test_label = test.iloc[:, 0:-3], test.loc[:,'emotion']
            
            sc1 = MinMaxScaler(feature_range=(-1, 1))
            sc1.fit(train_data)
            train_std = sc1.transform(train_data)
            feature_selector.fit(train_std, emo_train_label)
            features = feature_selector.transform(train_std)
            
            start = timer()
            model = GridSearchCV(SVC(decision_function_shape = 'ovr'), tuned_parameters[para], cv=10)
            model.fit(features, emo_train_label)
            end = timer()
            test_std = sc1.transform(test_data)
            test_std = test_std[:,feature_selector.get_support()]
            pred_label = model.predict(test_std)
            #print(classification_report(emo_test_label, pred_label))
            acc, cnm = plot_confusion_matrix(language, emo_test_label, pred_label)
            if acc > max_acc:
                max_acc = acc
                opt_model = model
                opt_range = sc1
                test_file = test.loc[:,'file']
            train_time[para] += end-start
            avg_acc[para] += acc
            matrix[para] = np.add(matrix[para], cnm)
     
    train_time /= time
    avg_acc /= time
    print(train_time)
    print(avg_acc)
    for i in range(2):
        print(matrix[i]*100.0/matrix[i].sum(axis=1)[:,None])
    cPickle.dump(opt_model, open(output+'_model.pkl', 'wb'))
    cPickle.dump(opt_range, open(output+'_range.pkl', 'wb'))
    with open('test_file.txt', 'w') as f:
        f.write(str(test_file))

"""
def gender_identification():

    gender_model = clf1.fit(train_std, gender_train_label)
    #male model
    sc2.fit(male_train_data)
    train_std = sc2.transform(male_train_data)
    male_model = clf2.fit(train_std, male_train_label)
    #female model
    sc3.fit(female_train_data)
    train_std = sc3.transform(female_train_data)
    female_model = clf3.fit(train_std, female_train_label)
    #predict gender
    pred_label = clf1.predict(test_std)
    pred_emo_label = []    
    for i in range(pred_label.shape[0]):
        if pred_label[i] == 0:
            t_std = sc2.transform(test_data.values[i].reshape(1, -1))
            pred_emo_label.append(clf2.predict(t_std))
        else:
            t_std = sc3.transform(test_data.values[i].reshape(1, -1))
            pred_emo_label.append(clf3.predict(t_std))
    pred_emo_label = np.asarray(pred_emo_label)
    plot_confusion_matrix(pred_emo_label,emo_test_label)
    #male
    
    
    test_std = sc2.transform(male_test_data)
    
    pred_label = clf2.predict(test_std)
    print(classification_report(male_test_label, pred_label))
    plot_confusion_matrix(pred_label,male_test_label)
    
    #female
    
    test_std = sc3.transform(female_test_data)
    
    pred_label = clf3.predict(test_std)
    print(classification_report(female_test_label, pred_label))
    plot_confusion_matrix(pred_label,female_test_label)
    #eval_model(train_std, emo_train_label, test_std, emo_test_label)
    #eval_model(train_std, gender_train_label, test_std, gender_test_label)
"""
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ind', help='speaker independent', action='store_true')
    parser.add_argument('--fin', help='final model', action='store_true')
    parser.add_argument('-t', '--time', required=False, help='training time')
    parser.add_argument('-i', '--input', required=True, help='input dataset')
    parser.add_argument('-o', '--output', required=False, help='output model')
    parser.add_argument('-l', '--language', required=False, help='language of input dataset(1 for German 2 for English)')
    args =vars(parser.parse_args())
    df = pd.read_csv(args['input'], index_col = 0)
    if args['language'] != None:
        language = int(args['language'])
        if language not in (1, 2):
            raise Exception('Please input 1 or 2!')
        if args['ind'] == True:
            independent_train(df, language)
        else:
            random_train(df, language, 0.8, int(args['time']), args['output'] or args['input'].split('.')[0])
    else:
        final_model(df, os.path.basename(args['output'] or args['input']).split('.')[0])
       
    
