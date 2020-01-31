#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:01:26 2019

# 2. hyper_training_train
train the Train dataset from mlogit. Using two architectures.

@author: shenhao
"""

#cd /Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/9_ml_dnn_alt_spe_util/code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats as ss
import pickle
import util_nn_mlarch as util
from sklearn import preprocessing


#with open('mlogit_choice_data.pickle', 'rb') as data:
#    data_dic = pickle.load(data)
with open('data/mlogit_choice_data.pickle', 'rb') as data:
    data_dic = pickle.load(data)

# use Train dataset
df = data_dic['Train_wide']

# divide into training_validation vs. testing set.
np.random.seed(100) # replicable
n_index = df.shape[0]
n_index_shuffle = np.arange(n_index)
np.random.shuffle(n_index_shuffle)
data_shuffled = df 
data_shuffled = df.loc[n_index_shuffle, :]

# replace values
choice_name_dic = {}
choice_name_dic['choice1'] = 0
choice_name_dic['choice2'] = 1
data_shuffled['choice']=data_shuffled['choice'].replace(to_replace=choice_name_dic.keys(),value=choice_name_dic.values())

useful_vars = ['choice', 'price1', 'time1', 'change1', 'comfort1',
               'price2', 'time2', 'change2', 'comfort2']
data_shuffled_useful_vars=data_shuffled[useful_vars]
data_shuffled_useful_vars.dropna(axis = 0, how = 'any', inplace = True)
print(data_shuffled_useful_vars.columns)

# normalize values
X = preprocessing.scale(data_shuffled_useful_vars.iloc[:, 1:].values)
Y = data_shuffled_useful_vars.iloc[:, 0].values

def generate_cross_validation_set(data, validation_index, df = True):
    '''
    five_fold cross validation
    return training set and validation set
    df: True (is a dataframe); df: False: (is a matrix)
    '''
#    np.random.seed(100) # replicable
    n_index = data.shape[0]
#    n_index_shuffle = np.arange(n_index)
#    np.random.shuffle(n_index_shuffle)
    data_shuffled = data # may not need to shuffle the data...
#    data_shuffled = data.loc[n_index_shuffle, :]
    # use validation index to split; validation index: 0,1,2,3,4
    if df == True:
        if len(data.shape) > 1:
            validation_set = data_shuffled.iloc[np.int(n_index/5)*validation_index:np.int(n_index/5)*(validation_index+1), :]
            train_set = pd.concat([data_shuffled.iloc[: np.int(n_index/5)*validation_index, :], 
                                                         data_shuffled.iloc[np.int(n_index/5)*(validation_index+1):, :]]) 
        elif len(data.shape) == 1:
            validation_set = data_shuffled.iloc[np.int(n_index/5)*validation_index:np.int(n_index/5)*(validation_index+1)]
            train_set = pd.concat([data_shuffled.iloc[: np.int(n_index/5)*validation_index], 
                                                         data_shuffled.iloc[np.int(n_index/5)*(validation_index+1):]])    
    elif df == False:
        if len(data.shape) > 1:
            validation_set = data_shuffled[np.int(n_index/5)*validation_index:np.int(n_index/5)*(validation_index+1), :]
            train_set = np.concatenate([data_shuffled[: np.int(n_index/5)*validation_index, :], 
                                                         data_shuffled[np.int(n_index/5)*(validation_index+1):, :]]) 
        elif len(data.shape) == 1:
            validation_set = data_shuffled[np.int(n_index/5)*validation_index:np.int(n_index/5)*(validation_index+1)]
            train_set = np.concatenate([data_shuffled[: np.int(n_index/5)*validation_index], 
                                                         data_shuffled[np.int(n_index/5)*(validation_index+1):]])           
    
    return train_set,validation_set


# 
X_train_validation = X[:np.int(n_index*5/6), :]
X_test = X[np.int(n_index*5/6):, :]
Y_train_validation = Y[:np.int(n_index*5/6)]
Y_test = Y[np.int(n_index*5/6):]

############################################################
############################################################
### use other models as benchmarks
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

model_titles = ['MNL_with_l1_reg_medium',
                'MNL_with_l2_reg_medium',
                'Linear_SVM_medium',
                'RBF_SVM_medium',
                'Naive_Bayesian',
                'KNN_3_medium',
                'DT_medium',
                'AdaBoost',
                "QDA_medium"]

model_list = [LogisticRegression(penalty = 'l1', C = 1, multi_class = 'multinomial', solver = 'saga'),
              # larger C, weaker penalty

              LogisticRegression(penalty = 'l2', C = 1, multi_class = 'multinomial', solver = 'newton-cg'),
              # larger C, weaker penalty

              SVC(kernel = 'linear', C = 1),
              # larger C, weaker penalty

              SVC(kernel = 'rbf', gamma=2, C = 1),
              # larger C, weaker penalty

              GaussianNB(),
              # Naive Bayesian. Cannnot find regularization terms

              KNeighborsClassifier(n_neighbors = 3),
              # n_neighbors, larger n_neighbors, more bias, similar to stronger penalty

              DecisionTreeClassifier(max_depth=5),
              # smaller depth, more regularization

              AdaBoostClassifier(),
              # Boost classifer with Decision Tree as base. Cannot find regularization terms.

              QuadraticDiscriminantAnalysis(reg_param = 1),
              # reg_param, larger reg_param, stronger penalty
              ]

# 
training_accuracy_table = pd.DataFrame(np.zeros((len(model_list), 5)), index = model_titles)
validation_accuracy_table = pd.DataFrame(np.zeros((len(model_list), 5)), index = model_titles)
testing_accuracy_table = pd.DataFrame(np.zeros((len(model_list), 5)), index = model_titles)

for j in range(5):
    # five fold training with cross validation
    X_train,X_validation = generate_cross_validation_set(X_train_validation, j, df = False)
    Y_train,Y_validation = generate_cross_validation_set(Y_train_validation, j, df = False)

    ###
    for name, model in zip(model_titles, model_list):    
        print()
        print("Training model ", name, " ...")
        model.fit(X_train, Y_train)
        # compute accuracy
        training_accuracy = accuracy_score(model.predict(X_train), Y_train)
        validation_accuracy = accuracy_score(model.predict(X_validation), Y_validation)
        testing_accuracy = accuracy_score(model.predict(X_test), Y_test)
        print("Its training accuracy is:", training_accuracy)
        print("Its validation accuracy is:", validation_accuracy)
        print("Its testing accuracy is:", testing_accuracy)

        training_accuracy_table.loc[name,j]=training_accuracy
        validation_accuracy_table.loc[name,j]=validation_accuracy
        testing_accuracy_table.loc[name,j]=testing_accuracy
        
### 
classifiers_accuracy = {}
classifiers_accuracy['training']=training_accuracy_table
classifiers_accuracy['validation']=validation_accuracy_table
classifiers_accuracy['testing']=testing_accuracy_table

                    
############################################################
############################################################
# specify hyperparameter space for fully connected DNN
M_list = [1,2,3,4,5,6,7,8,9,10,11,12] # 5
n_hidden_list = [60, 120, 240, 360, 480, 600] # 6
l1_const_list = [1e-3, 1e-5, 1e-10, 1e-20]# 8
l2_const_list = [1e-3, 1e-5, 1e-10, 1e-20]# 8
dropout_rate_list = [1e-3, 1e-5] # 5
batch_normalization_list = [True, False] # 2
learning_rate_list = [0.01, 1e-3, 1e-4, 1e-5] # 5
n_iteration_list = [500, 1000, 5000, 10000, 20000] # 5
n_mini_batch_list = [50, 100, 200, 500, 1000] # 5

total_sample = 50 # could change...in total it has 250 training.
full_dnn_dic = {}

for i in range(total_sample):
    print("------------------------")
    print("Estimate full connected model ", str(i))
    M = np.random.choice(M_list, size = 1)[0]
    n_hidden = np.random.choice(n_hidden_list, size = 1)[0]
    l1_const = np.random.choice(l1_const_list, size = 1)[0]
    l2_const = np.random.choice(l2_const_list, size = 1)[0]
    dropout_rate = np.random.choice(dropout_rate_list, size = 1)[0]
    batch_normalization = np.random.choice(batch_normalization_list, size = 1)[0]
    learning_rate = np.random.choice(learning_rate_list, size = 1)[0]
    n_iteration = np.random.choice(n_iteration_list, size = 1)[0]
    n_mini_batch = np.random.choice(n_mini_batch_list, size = 1)[0]
    
    # store information
    full_dnn_dic[i] = {}
    full_dnn_dic[i]['M'] = M
    full_dnn_dic[i]['n_hidden'] = n_hidden
    full_dnn_dic[i]['l1_const'] = l1_const
    full_dnn_dic[i]['l2_const'] = l2_const
    full_dnn_dic[i]['dropout_rate'] = dropout_rate
    full_dnn_dic[i]['batch_normalization'] = batch_normalization
    full_dnn_dic[i]['learning_rate'] = learning_rate
    full_dnn_dic[i]['n_iteration'] = n_iteration
    full_dnn_dic[i]['n_mini_batch'] = n_mini_batch
    print(full_dnn_dic[i])
    
    for j in range(5):
        # five fold training with cross validation
        X_train,X_validation = generate_cross_validation_set(X_train_validation, j, df = False)
        Y_train,Y_validation = generate_cross_validation_set(Y_train_validation, j, df = False)

        # training
        train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt = \
                        util.dnn_estimation(X_train, Y_train, X_validation, Y_validation, X_test, Y_test,
                                       M, n_hidden, l1_const, l2_const, dropout_rate, batch_normalization, learning_rate, n_iteration, n_mini_batch, 
                                       K=2, Train = True)
        # 
        print("Training accuracy is ", train_accuracy)
        print("Validation accuracy is ", validation_accuracy)
        print("Testing accuracy is ", test_accuracy)

        # store information        
        full_dnn_dic[i]['train_accuracy_'+str(j)] = train_accuracy
        full_dnn_dic[i]['validation_accuracy_'+str(j)] = validation_accuracy            
        full_dnn_dic[i]['test_accuracy_'+str(j)] = test_accuracy            
        full_dnn_dic[i]['prob_cost_'+str(j)] = prob_cost
        full_dnn_dic[i]['prob_ivt_'+str(j)] = prob_ivt

############################################################
############################################################
# specify hyperparameter space for sparsely connected DNN
# define hyperparameter space
M_list = [1,2,3,4,5,6,7,8,9,10,11,12] # 5
n_hidden_list = [30, 60, 120, 180, 240, 300] # 6
l1_const_list = [1e-3, 1e-5, 1e-10, 1e-20] # 8
l2_const_list = [1e-3, 1e-5, 1e-10, 1e-20]# 8
dropout_rate_list = [0.5, 0.1, 0.01, 1e-3, 1e-5] # 5
batch_normalization_list = [True, False] # 2
learning_rate_list = [0.01, 1e-3, 1e-4, 1e-5] # 5
n_iteration_list = [500, 1000, 5000, 10000, 20000] # 5
n_mini_batch_list = [50, 100, 200, 500, 1000] # 5

# random draw...and HPO
total_sample = 50 # could change...in total it has 250 training.
sparse_dnn_dic = {}
for i in range(total_sample):
    print("------------------------")
    print("Estimate sparse connected model ", str(i))

    M = np.random.choice(M_list, size = 1)[0]
    n_hidden = np.random.choice(n_hidden_list, size = 1)[0]
    l1_const = np.random.choice(l1_const_list, size = 1)[0]
    l2_const = np.random.choice(l2_const_list, size = 1)[0]
    dropout_rate = np.random.choice(dropout_rate_list, size = 1)[0]
    batch_normalization = np.random.choice(batch_normalization_list, size = 1)[0]
    learning_rate = np.random.choice(learning_rate_list, size = 1)[0]
    n_iteration = np.random.choice(n_iteration_list, size = 1)[0]
    n_mini_batch = np.random.choice(n_mini_batch_list, size = 1)[0]

    # store information
    sparse_dnn_dic[i] = {}
    sparse_dnn_dic[i]['M'] = M
    sparse_dnn_dic[i]['n_hidden'] = n_hidden
    sparse_dnn_dic[i]['l1_const'] = l1_const
    sparse_dnn_dic[i]['l2_const'] = l2_const
    sparse_dnn_dic[i]['dropout_rate'] = dropout_rate
    sparse_dnn_dic[i]['batch_normalization'] = batch_normalization
    sparse_dnn_dic[i]['learning_rate'] = learning_rate
    sparse_dnn_dic[i]['n_iteration'] = n_iteration
    sparse_dnn_dic[i]['n_mini_batch'] = n_mini_batch
    print(sparse_dnn_dic[i])              

    for j in range(5):
        X_train,X_validation = generate_cross_validation_set(X_train_validation, j, df = False)
        Y_train,Y_validation = generate_cross_validation_set(Y_train_validation, j, df = False)

        X0_train=X_train[:, :4]
        X1_train=X_train[:, 4:]
        X0_validation=X_validation[:,:4]
        X1_validation=X_validation[:,4:]
        X0_test=X_test[:,:4]
        X1_test=X_test[:,4:]
        
        # one estimation here
        train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt = \
                    util.dnn_alt_spec_estimation_train(X0_train,X1_train,Y_train,
                                            X0_validation,X1_validation,Y_validation,
                                            X0_test,X1_test,Y_test,
                                            M,n_hidden,l1_const,l2_const,
                                            dropout_rate,batch_normalization,learning_rate,n_iteration,n_mini_batch,K=2)
                    
        print("Training accuracy is ", train_accuracy)
        print("Validation accuracy is ", validation_accuracy)
        print("Testing accuracy is ", test_accuracy)
    
        # store information
        sparse_dnn_dic[i]['train_accuracy'+str(j)] = train_accuracy
        sparse_dnn_dic[i]['validation_accuracy'+str(j)] = validation_accuracy
        sparse_dnn_dic[i]['test_accuracy'+str(j)] = test_accuracy
        sparse_dnn_dic[i]['prob_cost'+str(j)] = prob_cost
        sparse_dnn_dic[i]['prob_ivt'+str(j)] = prob_ivt

# outputs
import pickle
####
#with open('full_dnn_results_train.pickle', 'wb') as full_dnn_results_finer:
#    pickle.dump(full_dnn_dic, full_dnn_results_finer, protocol=pickle.HIGHEST_PROTOCOL)
#with open('sparse_dnn_results_train.pickle', 'wb') as sparse_dnn_results_finer:
#    pickle.dump(sparse_dnn_dic, sparse_dnn_results_finer, protocol=pickle.HIGHEST_PROTOCOL)
#with open('classifiers_accuracy_train.pickle', 'wb') as data:
#    pickle.dump(classifiers_accuracy, data, protocol=pickle.HIGHEST_PROTOCOL)

###
with open('output/full_dnn_results_train.pickle', 'wb') as full_dnn_results_finer:
    pickle.dump(full_dnn_dic, full_dnn_results_finer, protocol=pickle.HIGHEST_PROTOCOL)
with open('output/sparse_dnn_results_train.pickle', 'wb') as sparse_dnn_results_finer:
    pickle.dump(sparse_dnn_dic, sparse_dnn_results_finer, protocol=pickle.HIGHEST_PROTOCOL)
with open('output/classifiers_accuracy_train.pickle', 'wb') as data:
    pickle.dump(classifiers_accuracy, data, protocol=pickle.HIGHEST_PROTOCOL)





















