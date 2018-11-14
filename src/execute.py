from __future__ import print_function
import warnings
import sys
import feature_extraction as fe
from svm import *
from svmutil import *
from subprocess import *
from sklearn.metrics import confusion_matrix

FILE = sys.argv[1]
MODEL = sys.argv[2]
outfile = FILE.split('.')[0]
scaled_test_file = outfile + '.scale'
#PATH = sys.argv[3]
svmscale_exe = './svm-scale'
range_file = './trainingset.range'
EMOTION = ["Anger","Boredom","Disgust","Fear","Happiness","Sadness","Neutral"]
EMOTION_TE = ["Anger", "Fear", "Happiness", "Sadness", "Neutral"]

fe.getTestFeat(FILE,outfile)

cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, outfile, scaled_test_file)
print('Scaling testing data...')
Popen(cmd, shell = True, stdout = PIPE).communicate()	


test_label, test_data = svm_read_problem(scaled_test_file)
train_model = svm_load_model(MODEL)
pred_label, pred_acc, pred_val = svm_predict(test_label, test_data, train_model, options = '-q')
print(pred_label)





