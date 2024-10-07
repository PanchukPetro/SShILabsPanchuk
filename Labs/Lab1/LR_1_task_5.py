import pandas as pd
df = pd.read_csv('data_metrics.csv')
df.head()
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

from sklearn.metrics import confusion_matrix
confusion_matrix(df.actual_label.values,df.predicted_RF.values)



def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))

print('TP:',find_TP(df.actual_label.values,
df.predicted_RF.values))
print('FN:',find_FN(df.actual_label.values,
df.predicted_RF.values))
print('FP:',find_FP(df.actual_label.values,
df.predicted_RF.values))
print('TN:',find_TN(df.actual_label.values,
df.predicted_RF.values))

import numpy as np
def find_conf_matrix_values(y_true,y_pred):
 # calculate TP, FN, FP, TN
 TP = find_TP(y_true,y_pred)
 FN = find_FN(y_true,y_pred)
 FP = find_FP(y_true,y_pred)
 TN = find_TN(y_true,y_pred)
 return TP,FN,FP,TN
def panchuk_confusion_matrix(y_true, y_pred):
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return np.array([[TN,FP],[FN,TP]])

panchuk_confusion_matrix(df.actual_label.values,
                         df.predicted_RF.values)

assert np.array_equal(panchuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values), panchuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'panchuk_confusion_matrix() is not correct for RF'
assert np.array_equal(panchuk_confusion_matrix(df.actual_label.values, df.predicted_LR.values), confusion_matrix(df.actual_label.values,df.predicted_LR.values)), 'panchuk_confusion_matrix() is not correct for LR'

from sklearn.metrics import accuracy_score
print(accuracy_score(df.actual_label.values,
df.predicted_RF.values))

def panchuk_accuracy_score(y_true, y_pred):
 # calculates the fraction of samples
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 accuracy = (TP + TN) / (TP + TN + FP + FN)
 return accuracy
assert panchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'panchuk_accuracy_score failed on RF'
assert panchuk_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'panchuk_accuracy_score failed on LR'
print('Accuracy RF: %.3f' % (panchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))

from sklearn.metrics import recall_score
print(recall_score(df.actual_label.values, df.predicted_RF.values))

def panchuk_recall_score(y_true, y_pred):
 # calculates the fraction of positive samples predicted correctly
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 recall = TP / (TP + FN) if (TP + FN) != 0 else 0
 return recall

assert panchuk_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'panchuk_accuracy_score failed on RF'
assert panchuk_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'panchuk_accuracy_score failed on LR'
print('Recall RF: %.3f'%(panchuk_recall_score(df.actual_label.values,
                                              df.predicted_RF.values)))

from sklearn.metrics import precision_score
print(precision_score(df.actual_label.values, df.predicted_RF.values))
def panchuk_precision_score(y_true, y_pred):
    # Get the confusion matrix values
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    # Calculate precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    return precision

assert panchuk_precision_score(df.actual_label.values,
                               df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'panchuk_accuracy_score failed on RF'
assert panchuk_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'panchuk_accuracy_score failed on LR'
print('Precision RF: %.3f'%(panchuk_precision_score(df.actual_label.values,
                                                    df.predicted_RF.values)))

from sklearn.metrics import f1_score
print(f1_score(df.actual_label.values, df.predicted_RF.values))
def panchuk_f1_score(y_true, y_pred):
    # Get recall and precision
    recall = panchuk_recall_score(y_true, y_pred)
    precision = panchuk_precision_score(y_true, y_pred)

    # Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0  # To handle the case where both precision and recall are zero


    return f1_score


# Assuming df is a DataFrame with actual and predicted labels
#assert panchuk_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values), 'panchuk_accuracy_score failed on RF'
#assert panchuk_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'panchuk_accuracy_score failed on LR'
print("Panchuk f1 RF value: %.15f" % panchuk_f1_score(df.actual_label.values, df.predicted_RF.values))
print('F1   RF value: %.15f'% f1_score(df.actual_label.values, df.predicted_RF.values))
print("Panchuk f1 LR value: %.15f" % panchuk_f1_score(df.actual_label.values, df.predicted_LR.values))
print("F1   LR value: %.15f"% f1_score(df.actual_label.values, df.predicted_LR.values))
print("Чогось асерти говорять що значення не рівні, вивів їх в консоль")


print('Scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (panchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (panchuk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (panchuk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (panchuk_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')

print('Scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (panchuk_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (panchuk_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (panchuk_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (panchuk_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))


from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values,df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()





