"""
Created on Wed Nov  7 21:18:30 2018

finer hyper paramter training with cross validation

@author: shenhao
"""

# cd /Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/9_ml_dnn_alt_spe_util/code
# cd D:\Dropbox (MIT)\Shenhao_Jinhua (1)\9_ml_dnn_alt_spe_util\code

# read datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pickle

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.expressions as bioexp
import os
import biogeme.models as biomodels

# starting time
start_time = time.time()
with open('data/mlogit_choice_data.pickle', 'rb') as data:
    data_dic = pickle.load(data)

# use Train dataset

model_titles = ['MNL']

df = data_dic['Train_wide']

# divide into training_validation vs. testing set.
np.random.seed(100)  # replicable
n_index = df.shape[0]
n_index_shuffle = np.arange(n_index)
np.random.shuffle(n_index_shuffle)
data_shuffled = df.loc[n_index_shuffle, :]

# replace values
choice_name_dic = {}
choice_name_dic['choice1'] = 0
choice_name_dic['choice2'] = 1
data_shuffled['choice'] = data_shuffled['choice'].replace(to_replace=choice_name_dic.keys(),
                                                          value=choice_name_dic.values())
key_choice_index = {'choice1':0,'choice2':1}
modes_list = ['choice1','choice2']

useful_vars = ['choice', 'price1', 'time1', 'change1', 'comfort1',
               'price2', 'time2', 'change2', 'comfort2']
data_shuffled_useful_vars = data_shuffled[useful_vars]
data_shuffled_useful_vars.dropna(axis=0, how='any', inplace=True)
print(data_shuffled_useful_vars.columns)

# normalize values
X = preprocessing.scale(data_shuffled_useful_vars.iloc[:, 1:].values)
Y = data_shuffled_useful_vars.iloc[:, 0].values


classifiers_accuracy = {}
classifiers_accuracy['training'] = pd.DataFrame()
classifiers_accuracy['validation'] = pd.DataFrame()
classifiers_accuracy['testing'] = pd.DataFrame()

input_vars = ['price1', 'time1', 'change1', 'comfort1',
               'price2', 'time2', 'change2', 'comfort2']
att = {'choice1':['price1', 'time1', 'change1', 'comfort1'],
       'choice2':['price2', 'time2', 'change2', 'comfort2']}
def generate_cross_validation_set(data, validation_index, df=True):
    '''
    five_fold cross validation
    return training set and validation set
    df: True (is a dataframe); df: False: (is a matrix)
    '''
    #    np.random.seed(100) # replicable
    n_index = data.shape[0]
    #    n_index_shuffle = np.arange(n_index)
    #    np.random.shuffle(n_index_shuffle)
    data_shuffled = data  # may not need to shuffle the data...
    #    data_shuffled = data.loc[n_index_shuffle, :]
    # use validation index to split; validation index: 0,1,2,3,4
    if df == True:
        if len(data.shape) > 1:
            validation_set = data_shuffled.iloc[
                             np.int(n_index / 5) * validation_index:np.int(n_index / 5) * (validation_index + 1), :]
            train_set = pd.concat([data_shuffled.iloc[: np.int(n_index / 5) * validation_index, :],
                                   data_shuffled.iloc[np.int(n_index / 5) * (validation_index + 1):, :]])
        elif len(data.shape) == 1:
            validation_set = data_shuffled.iloc[
                             np.int(n_index / 5) * validation_index:np.int(n_index / 5) * (validation_index + 1)]
            train_set = pd.concat([data_shuffled.iloc[: np.int(n_index / 5) * validation_index],
                                   data_shuffled.iloc[np.int(n_index / 5) * (validation_index + 1):]])
    elif df == False:
        if len(data.shape) > 1:
            validation_set = data_shuffled[
                             np.int(n_index / 5) * validation_index:np.int(n_index / 5) * (validation_index + 1), :]
            train_set = np.concatenate([data_shuffled[: np.int(n_index / 5) * validation_index, :],
                                        data_shuffled[np.int(n_index / 5) * (validation_index + 1):, :]])
        elif len(data.shape) == 1:
            validation_set = data_shuffled[
                             np.int(n_index / 5) * validation_index:np.int(n_index / 5) * (validation_index + 1)]
            train_set = np.concatenate([data_shuffled[: np.int(n_index / 5) * validation_index],
                                        data_shuffled[np.int(n_index / 5) * (validation_index + 1):]])

    return train_set, validation_set


#
X_train_validation = X[:np.int(n_index * 5 / 6), :]
X_test = X[np.int(n_index * 5 / 6):, :]
Y_train_validation = Y[:np.int(n_index * 5 / 6)]
Y_test = Y[np.int(n_index * 5 / 6):]

def train_MNL(data):
    for mode in modes_list:
        # availability
        data[mode+'_avail'] = 1
    database = db.Database("MNL_Train", data)
    beta_dic = {}
    variables = {}

    ASC_1 = bioexp.Beta('B___ASC___choice1',0,None,None,1) #fixed
    ASC_2 = bioexp.Beta('B___ASC___choice2',0,None,None,0)

    for key in att:
        beta_dic[key] = {}
        for var in att[key]:
            if var not in variables:
                variables[var] = bioexp.Variable(var)
            beta_name = 'B___' + var + '___' + key
            beta_dic[key][beta_name] = bioexp.Beta(beta_name, 0, None, None, 0)


    V = {key_choice_index['choice1']:ASC_1, key_choice_index['choice2']:ASC_2}
    AV = {}

    for key in att:
        AV[key_choice_index[key]] = bioexp.Variable(key+'_avail')
        for var in att[key]:
            beta_name = 'B___' + var + '___' + key
            V[key_choice_index[key]] += variables[var] * beta_dic[key][beta_name]
    CHOICE = bioexp.Variable('choice')
    logprob = bioexp.bioLogLogit(V, AV, CHOICE)
    formulas = {'loglike': logprob}
    biogeme = bio.BIOGEME(database, formulas,numberOfThreads = 4)
    biogeme.modelName = "MNL_Train"
    results = biogeme.estimate()
    os.remove("MNL_Train.html")
    os.remove("MNL_Train.pickle")
    # Print the estimated values
    betas = results.getBetaValues()
    beta={}
    for k, v in betas.items():
        beta[k] = v

    return beta

def predict_MNL(data_test,betas):

    for mode in modes_list:
        col_name = 'exp_U_' + mode
        data_test[col_name] = 0
    for k in betas:
        v = betas[k]
        mode = k.split('___')[2]
        col_name = 'exp_U_' + mode
        if 'ASC' in k:
            data_test[col_name] += 1 * v
        else:
            var_name = k.split('___')[1]
            data_test[col_name] += data_test[var_name] * v
    data_test['exp_U'] = 0
    for mode in modes_list:
        col_name = 'exp_U_' + mode
        data_test[col_name] = np.exp(data_test[col_name])
        data_test['exp_U'] += data_test[col_name]

    prob_list = []
    for mode in modes_list:
        col_nameprob = 'prob_' + mode
        prob_list.append(col_nameprob)
        col_name = 'exp_U_' + mode
        data_test[col_nameprob] = data_test[col_name]/data_test['exp_U']

    data_test['max_prob'] = data_test[prob_list].max(axis=1)
    data_test['CHOOSE'] = 0

    for mode in key_choice_index:
        col_nameprob = 'prob_' + mode
        data_test.loc[data_test[col_nameprob]==data_test['max_prob'],'CHOOSE'] = key_choice_index[mode]

    acc = len(data_test.loc[data_test['CHOOSE']==data_test['choice']])/len(data_test)
    return acc, data_test



training_accuracy_table = pd.DataFrame(np.zeros((len(model_titles), 5)), index=model_titles)
validation_accuracy_table = pd.DataFrame(np.zeros((len(model_titles), 5)), index=model_titles)
testing_accuracy_table = pd.DataFrame(np.zeros((len(model_titles), 5)), index=model_titles)

for j in range(5):
    # five fold training with cross validation
    X_train,X_validation = generate_cross_validation_set(X_train_validation, j, df = False)
    Y_train, Y_validation = generate_cross_validation_set(Y_train_validation, j, df=False)
    train_df = pd.DataFrame(X_train,columns = input_vars)
    train_df['choice'] = Y_train
    validation_df = pd.DataFrame(X_validation, columns=input_vars)
    validation_df['choice'] = Y_validation

    test_df = pd.DataFrame(X_test,columns = input_vars)
    test_df['choice'] = Y_test

    ###
    for name in model_titles:
        tic = time.time()
        print("Training model ", name, " ...")
        if name == 'MNL':
            beta = train_MNL(train_df)
            Training_time = round((time.time() - tic), 2)
            print('Training time', Training_time, 'seconds')
            training_accuracy,_ = predict_MNL(train_df, beta)
            validation_accuracy,_ = predict_MNL(validation_df, beta)
            testing_accuracy,df_sp_test_prob = predict_MNL(test_df, beta)


        # compute accuracy

        print("Its training accuracy is:", training_accuracy)
        print("Its validation accuracy is:", validation_accuracy)
        print("Its testing accuracy is:", testing_accuracy)

        training_accuracy_table.loc[name, j] = training_accuracy
        validation_accuracy_table.loc[name, j] = validation_accuracy
        testing_accuracy_table.loc[name, j] = testing_accuracy

        print (' ================== ')

classifiers_accuracy['training']=training_accuracy_table
classifiers_accuracy['validation']=validation_accuracy_table
classifiers_accuracy['testing']=testing_accuracy_table

import pickle
with open('output/classifiers_accuracy_MNL_NL_Train.pickle', 'wb') as data:
    pickle.dump(classifiers_accuracy, data, protocol=pickle.HIGHEST_PROTOCOL)


with open('output/classifiers_accuracy_MNL_NL_Train.pickle', 'rb') as source:
    classifiers_accuracy_MNL_NL_dic = pickle.load(source)


classifier_train_MNL_NL = classifiers_accuracy_MNL_NL_dic['training']
classifier_validation_MNL_NL = classifiers_accuracy_MNL_NL_dic['validation']
classifier_test_MNL_NL = classifiers_accuracy_MNL_NL_dic['testing']
classifier_train_MNL_NL=classifier_train_MNL_NL.mean(1)
classifier_validation_MNL_NL=classifier_validation_MNL_NL.mean(1)
classifier_test_MNL_NL=classifier_test_MNL_NL.mean(1)
save_mnl_nl_acc = pd.DataFrame({'Model':['MNL'],'Train_acc':list(classifier_train_MNL_NL.values),
                   'Val_acc':list(classifier_validation_MNL_NL.values),'Test_acc':list(classifier_test_MNL_NL.values)})
save_mnl_nl_acc.to_csv('output/MNL_NL_accuracy_Train.csv',index=False)














