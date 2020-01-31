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
import pickle
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# starting time
start_time = time.time()

# Singapore data
df_sp_train = pd.read_csv('data/data_AV_Singapore_v1_sp_train.csv')
df_sp_validation = pd.read_csv('data/data_AV_Singapore_v1_sp_validation.csv')
# here we combine train and validation set to recreate training and validation sets...
df_sp_combined_train = pd.concat([df_sp_train, df_sp_validation], axis=0)
df_sp_combined_train.index = np.arange(df_sp_combined_train.shape[0])
df_sp_test = pd.read_csv('data/data_AV_Singapore_v1_sp_test.csv')


# TRAIN data

with open('data/mlogit_choice_data.pickle', 'rb') as data:
    data_dic = pickle.load(data)

# use Train dataset
df = data_dic['Train_wide']

# divide into training_validation vs. testing set.
np.random.seed(100) # replicable
n_index = df.shape[0]
n_index_shuffle = np.arange(n_index)
np.random.shuffle(n_index_shuffle)
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

#======================

data_SGP_standard = pd.concat([df_sp_combined_train, df_sp_test])
data_SGP = pd.read_csv('data/data_AV_Singapore_v1_sp_full_nonstand.csv')
data_SGP = data_SGP.loc[:,data_SGP_standard.columns]
data_Train = data_shuffled_useful_vars.copy()

data_SGP_statistics = data_SGP.describe()
data_Train_statistics = data_Train.describe()

data_SGP_statistics.to_csv('output/data_SGP_statistics.csv')
data_Train_statistics.to_csv('output/data_Train_statistics.csv')

for i in range(5):
    num = len(data_SGP.loc[data_SGP['choice']==i])
    print('SGP choice',i, num, 'proportion:', num/len(data_SGP))

for i in range(2):
    num = len(data_Train.loc[data_Train['choice'] == i])
    print('Train choice',i, num, 'proportion:', num/len(data_Train))