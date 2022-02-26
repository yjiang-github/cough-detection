# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 14:53:28 2018

@author: yjian
"""


#%%
import time
import pandas as pd
from datetime import date, datetime
import csv
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.preprocessing import Normalizer
import math

###Random forest

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import  AdaBoostClassifier
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
#from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold


path = r''

#%%
createVar = locals()

#%%
# with open(os.path.join(path,'New Labeled Data v2.csv')) as Data:
#     reader = csv.reader(Data)
#     header = next(reader)
#     l_rawdata = [r for r in reader]

with open(os.path.join(path,'New Labeled Data v2.csv')) as Data:
    df_rawdata = pd.read_csv(Data)

l_index = ['Participantid','Hand','Status','Context']
df_rawdata = df_rawdata.set_index(l_index)
df_rawdata = df_rawdata.replace({'Activity':{'Non-cough':0,'Cough':1}})

#%%
ol = 0.9 # overlap rate = 90%

#tw = 2 # timewindow = 2s

twn = 200 # number of rows in one timewindow = 200

twn10 = math.floor(twn*(1-ol))

#%%

l_samples = [] # samples (feature vectors)
l_gts = [] # groudtruth
l_indexes = [] # indexes of pid hand, status and context

for index,df_vals in tqdm(df_rawdata.groupby(level = l_index)):
    
    s_index = 0 # starting index
    np_vals = df_vals[['X','Y','Z','Activity']].values
    
    while s_index + twn < len(np_vals):
        # x[0:200]+y[0:200]+z[0:200] as a vector sample
        l_samples.append(np.hstack((np_vals[s_index:s_index+twn,0],np_vals[s_index:s_index+twn,1],np_vals[s_index:s_index+twn,2])))
        l_gts.append(int(np_vals[s_index+twn,3]))
        l_indexes.append(index)
        
        s_index += twn10

#%%
# double cough samples and sampling the same size of non-cough samples
l_cough, l_noncough, l_facts_c, l_facts_n = [], [], [], []

for i in range(len(l_samples)):
    if l_gts[i] == 1:
        l_cough.append(l_samples[i])
        l_facts_c.append(l_indexes[i])
    else:
        l_noncough.append(l_samples[i])
        l_facts_n.append(l_indexes[i])

# double size
l_cough += l_cough
l_facts_c += l_facts_c

# sampling from l_noncough
l_sampling = random.sample(range(len(l_noncough)), len(l_cough))
l_noncough = [l_noncough[i] for i in l_sampling]
l_facts_n = [l_facts_n[i] for i in l_sampling]


l_x = l_cough + l_noncough
l_y = [1 for i in range(len(l_cough))] + [0 for i in range(len(l_noncough))]
l_facts = l_facts_c + l_facts_n

np_x = np.vstack(l_x)
np_y = np.array(l_y)


#%%
# define function:
# split the dataset by selecting a participantid (PID) where its samples will be set as test set 
# while the others are train set

def split_dataset(pid):
    l_pid = [i for i in range(len(l_facts)) if l_facts[i][0] == pid]
    l_nonpid = [i for i in range(len(l_facts)) if i not in l_pid]
    
    return np_x[l_nonpid], np_x[l_pid], np_y[l_nonpid], np_y[l_pid]

#%%
""" training preparation """


D = 3*twn # sample dimension
K = 2 # number of class


# Parameter initialization
h = 100 # size of hidden layer (number of neurons)
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
 
step_size = 1e-0 #step size
reg = 1e-3 # normalization parameter

Loss = []  #store loss for each time

#confusion matrix for neural networks
FN_NN = 0; #false negative
TN_NN = 0; #true negative
TP_NN = 0; #true positive
FP_NN = 0; #false negative

#confusion matrix for random forest
TN_RF = 0
TP_RF = 0
FP_RF = 0
FN_RF = 0

#list of length:
TN_NNN = []
FN_NNN = []
FP_NNN = []
TP_NNN = []

TN_RFN = []
FN_RFN = []
FP_RFN = []
TP_RFN = []


#%%
"""
######################################################
### --- NN & RF LOSOCV --- ###
######################################################
"""

# lists of prediction result
y_predNN = []
y_predRF = []

result = []

l_pid = sorted(list(df_rawdata.index.unique(level=0)))

for PID in l_pid:
    #leave one subject out cross validation: split training and test set
    X_train,X_test,y_train,y_test = split_dataset(PID)
    y_train = y_train.flatten()
    y_train = y_train.astype('int')
    y_test = y_test.flatten()
    y_test = y_test.astype('int')
    
    
    ### gradient iteration and cycle
    num_examples = X_train.shape[0]
    
    for i in range(50):
        
        hidden_layer = np.maximum(0, np.dot(X_train, W) + b) # using ReLU 
        scores = np.dot(hidden_layer, W2) + b2
 
        # Calculate classification probability
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
 
        # Calculate Cross Entropy Loss and regulization Terms
        corect_logprobs = -np.log(probs[range(num_examples),y_train])
        data_loss = np.sum(corect_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        #if i % 10 == 0:
        #    print ("iteration %d: loss %f" % (i, loss))
               
        # calculate gradient
        dscores = probs
        dscores[range(num_examples),y_train] -= 1
        dscores /= num_examples
        # gradient descent
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
    
        dhidden = np.dot(dscores, W2.T)
    
        dhidden[hidden_layer <= 0] = 0
    
        # back to gradient on w and b
        dW = np.dot(X_train.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)
 
        # add gradient in normalization
        dW2 += reg * W2
        dW += reg * W
    
        # parameter iteration and update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2
    
    ### Evaluate the result: Loss calculation
    hidden_layer = np.maximum(0, np.dot(X_test, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    # Calculate classification probability
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    #calculate cross entropy loss
    num_testsamples = X_test.shape[0]
    corect_logprobs = -np.log(probs[range(num_testsamples),y_test])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    Loss.append(loss)
    #print(loss)

    #Test accuracy: Neural network
    hidden_layer = np.maximum(0, np.dot(X_test, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print('test set accuracy Neural Network: %.8f' % (np.mean(predicted_class == y_test)))
    
    for l in range(0,len(predicted_class)):
        result.append(predicted_class[l])
        
    
    #Test accuracy: Random forest
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #result.append(clf.score(X_test, y_test))
    print('test set accuracy Random Forest: %.8f' % (clf.score(X_test, y_test)))
    
    
    #confusion matrix: Neural network
    TNNN0 = TN_NN
    FNNN0 = FN_NN
    FPNN0 = FP_NN
    TPNN0 = TP_NN
    
    
    for j in range(len(predicted_class)):
        if predicted_class[j] == 0 and y_test[j] == 0:
            TN_NN = TN_NN + 1
        elif predicted_class[j] == 0 and y_test[j] == 1:
            FN_NN = FN_NN + 1
        elif predicted_class[j] == 1 and y_test[j] == 0:
            FP_NN = FP_NN + 1
        elif predicted_class[j] == 1 and y_test[j] == 1:
            TP_NN = TP_NN + 1
        y_predNN.append(predicted_class[j]) #save predicted result
    
    #result for each subject
    TNNN1 = TN_NN
    FNNN1 = FN_NN
    FPNN1 = FP_NN
    TPNN1 = TP_NN
    
    TN_NNN.append(TNNN1-TNNN0)
    FN_NNN.append(FNNN1-FNNN0)
    FP_NNN.append(FPNN1-FPNN0)
    TP_NNN.append(TPNN1-TPNN0)
    
       
    #confusion matrix: Random forest
    TNRF0 = TN_RF
    FNRF0 = FN_RF
    FPRF0 = FP_RF
    TPRF0 = TP_RF

    for k in range(len(y_test)):
        if clf.predict(X_test)[k] == 1 and y_test[k] == 1:
            TP_RF = TP_RF + 1
        elif clf.predict(X_test)[k] == 1 and y_test[k] == 0:
            FP_RF = FP_RF + 1
        elif clf.predict(X_test)[k] == 0 and y_test[k] == 1:
            FN_RF = FN_RF + 1
        elif clf.predict(X_test)[k] == 0 and y_test[k] == 0:
            TN_RF = TN_RF + 1
        y_predRF.append(clf.predict(X_test)[k]) #save predicted result
    
    #result for each subject
    TNRF1 = TN_RF
    FNRF1 = FN_RF
    FPRF1 = FP_RF
    TPRF1 = TP_RF
    
    TN_RFN.append(TNRF1-TNRF0)
    FN_RFN.append(FNRF1-FNRF0)
    FP_RFN.append(FPRF1-FPRF0)
    TP_RFN.append(TPRF1-TPRF0)
    
    
    print("confusion matrix")
    print(confusion_matrix(y_test,predicted_class,labels=[0,1]))  

    print("classification report")
    print(classification_report(y_test,predicted_class))  

    print("accuracy:", accuracy_score(y_test, predicted_class))  

    
    
#print confusion matrix    
print('Confusion Matrix_Neural Network:  True Positive: %d False Positive: %d False Negative: %d True Negative: %d' 
      % (TP_NN, FP_NN, FN_NN, TN_NN))
print('Confusion Matrix_Random Forest:  True Positive: %d False Positive: %d False Negative: %d True Negative: %d' 
      % (TP_RF, FP_RF, FN_RF, TN_RF))
#Calculate mean loss of implementing neural network
sum(Loss)/len(Loss)








