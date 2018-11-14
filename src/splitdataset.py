import sys
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from svmutil import *

FILE = sys.argv[1]

""""
def get_data():
    data = load_svmlight_file(FILE)
    return data[0], data[1]
"""

def splitDataset():
    label, raw_data = svm_read_problem(FILE)
    train_data, test_data, train_label, test_label = train_test_split(raw_data, label, test_size=0.2)
    print(type(train_data),type(train_label))
    train_str = ''
    test_str = ''
    for i in range(len(train_label)):
       value = str(train_data[i])[1:-1].replace(',', '').replace(': ',':')
       train_str += str(int(train_label[i])) + ' ' + value + ' \n'
    for i in range(len(test_label)):
       value = str(test_data[i])[1:-1].replace(',', '').replace(': ',':')
       test_str += str(int(test_label[i])) + ' ' + value + ' \n'
    with open('trainingset', 'w') as f:       
       f.write(train_str)
    with open('testingset', 'w') as f:       
       f.write(test_str)


if __name__ == '__main__':

   splitDataset()
# CV
"""
cross_validation.train_test_split(data, label, test_size=0.3, random_state=0)
print((train_data.shape, test_data.shape, train_label.shape, test_label.shape))

#CV score

clf = svm.SVC(kernel='linear', C=8)
scores = cross_val_score(clf, data, label, cv=10)
print(scores)

"""
