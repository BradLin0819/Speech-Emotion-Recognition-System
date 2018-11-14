import sys
import feature_extraction as fe
from svm import *
from svmutil import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

FILE = sys.argv[1]
MODEL = sys.argv[2]
GENDER_MODEL = sys.argv[3]
MALE_MODEL = sys.argv[4]
FEMALE_MODEL = sys.argv[5]
#PATH = sys.argv[3]
EMOTION = ["Anger","Boredom","Disgust","Fear","Happiness","Sadness","Neutral"]
EMOTION_TE = ["Anger", "Fear", "Happiness", "Sadness", "Neutral"]
scaled_test_file = 'teee'
cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, 'teee_training.range', 'teee_testing', scaled_test_file + '.scale')
print('Scaling testing data...')
Popen(cmd, shell = True, stdout = PIPE).communicate()
	
test_label, test_data = svm_read_problem(scaled_test_file+'.scale')
model = svm_load_model(MODEL)
pred_label, pred_acc, pred_val = svm_predict(test_label, test_data, model)

cnm1 = confusion_matrix(pred_label,test_label)
col_sum1 = cnm1.sum(axis= 0)

for i in range(len(EMOTION)):
    print(EMOTION[i],"accuracy rate:",cnm[i][i]*100.0/col_sum1[i],"%")

print(classification_report(test_label, pred_label, target_names=EMOTION))
print("\nConfusion matrix1\n=================")
print(cnm1)
print("==========================")

cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, 'teee_training_gender.range', 'teee_testing', scaled_test_file + '_gender.scale')
print('Scaling testing data...')
Popen(cmd, shell = True, stdout = PIPE).communicate()
test_label, test_data = svm_read_problem(scaled_test_file+'_gender.scale')
gen_model = svm_load_model(GENDER_MODEL)
pred_label, pred_acc, pred_val = svm_predict(test_label, test_data, gen_model)
male_model = svm_load_model(MALE_MODEL)
female_model = svm_load_model(FEMALE_MODEL)




