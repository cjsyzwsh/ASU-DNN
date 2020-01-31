"""
Created on Wed Nov  7 21:18:30 2018

finer hyper paramter training with cross validation

@author: shenhao
"""

#cd /Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/9_ml_dnn_alt_spe_util/code
#cd D:\Dropbox (MIT)\Shenhao_Jinhua (1)\9_ml_dnn_alt_spe_util\code

# read datasets 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import util_nn_mlarch as util
import pickle
import copy
#starting time
start_time = time.time()

#%matplotlib inline
df_sp_train = pd.read_csv('data/data_AV_Singapore_v1_sp_train.csv')
df_sp_validation = pd.read_csv('data/data_AV_Singapore_v1_sp_validation.csv')
# here we combine train and validation set to recreate training and validation sets...
df_sp_combined_train = pd.concat([df_sp_train, df_sp_validation], axis = 0)
df_sp_combined_train.index = np.arange(df_sp_combined_train.shape[0])
df_sp_test = pd.read_csv('data/data_AV_Singapore_v1_sp_test.csv')
df_sp_test_nonstand = pd.read_csv('data/data_AV_Singapore_v1_sp_test_nonstand.csv') # for elasticity

def generate_cross_validation_set(data, validation_index):
    '''
    five_fold cross validation
    return training set and validation set
    '''
#    np.random.seed(100) # replicable
    n_index = data.shape[0]
#    n_index_shuffle = np.arange(n_index)
#    np.random.shuffle(n_index_shuffle)
    data_shuffled = data # may not need to shuffle the data...
#    data_shuffled = data.loc[n_index_shuffle, :]
    # use validation index to split; validation index: 0,1,2,3,4
    validation_set = data_shuffled.iloc[np.int(n_index/5)*validation_index:np.int(n_index/5)*(validation_index+1), :]
    train_set = pd.concat([data_shuffled.iloc[: np.int(n_index/5)*validation_index, :],
                                                 data_shuffled.iloc[np.int(n_index/5)*(validation_index+1):, :]])
    return train_set,validation_set

## test
#train_set, validation_set = generate_cross_validation_set(df_sp_combined_train, 3)

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

#

#############################################################
with open('output/full_dnn_results.pickle', 'rb') as full_dnn_results:
    full_dnn_dic = pickle.load(full_dnn_results)

with open('output/sparse_dnn_results.pickle', 'rb') as sparse_dnn_results:
    sparse_dnn_dic = pickle.load(sparse_dnn_results)

with open('output/full_dnn_results_finer.pickle', 'rb') as full_dnn_results_finer:
    full_dnn_finer_dic = pickle.load(full_dnn_results_finer)

with open('output/sparse_dnn_results_finer.pickle', 'rb') as sparse_dnn_results_finer:
    sparse_dnn_finer_dic = pickle.load(sparse_dnn_results_finer)

with open('output/classifiers_accuracy.pickle', 'rb') as source:
    classifier_accuracy_dic = pickle.load(source)

with open('output/full_dnn_results_train.pickle', 'rb') as source:
    full_dnn_train_dic = pickle.load(source)

with open('output/sparse_dnn_results_train.pickle', 'rb') as source:
    sparse_dnn_train_dic = pickle.load(source)

with open('output/classifiers_accuracy_train.pickle', 'rb') as source:
    classifier_accuracy_train_dic = pickle.load(source)

with open('output/classifiers_accuracy_MNL_NL.pickle', 'rb') as source:
    classifiers_accuracy_MNL_NL_dic = pickle.load(source)

#
sp_full_nonstand_df = pd.read_csv('data/data_AV_Singapore_v1_sp_full_nonstand.csv')

# create a dataframe for full dnn
columns_ = list(full_dnn_dic[1].keys())
columns_.remove('prob_cost')
columns_.remove('prob_ivt')
n_rows = len(full_dnn_dic.keys())
full_dnn_df = pd.DataFrame(np.zeros((n_rows, len(columns_))), columns=columns_)
for i in range(n_rows):
    for j in range(len(columns_)):
        full_dnn_df.iloc[i, j] = full_dnn_dic[i][columns_[j]]

# create a dataframe for full dnn (finer search)
columns_ = list(full_dnn_finer_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i]
n_rows = len(full_dnn_finer_dic.keys())
full_dnn_finer_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns=columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        full_dnn_finer_df.loc[i, columns_remove_prob[j]] = full_dnn_finer_dic[i][columns_remove_prob[j]]

# create a dataframe for sparse dnn
columns_sparse = list(sparse_dnn_dic[1].keys())
columns_sparse.remove('prob_cost')
columns_sparse.remove('prob_ivt')
n_rows = len(sparse_dnn_dic.keys())
sparse_dnn_df = pd.DataFrame(np.zeros((n_rows, len(columns_sparse))), columns=columns_sparse)
for i in range(n_rows):
    for j in range(len(columns_sparse)):
        sparse_dnn_df.iloc[i, j] = sparse_dnn_dic[i][columns_sparse[j]]

# create a dataframe for sparse dnn (finer search)
columns_ = list(sparse_dnn_finer_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i]
n_rows = len(sparse_dnn_finer_dic.keys())
sparse_dnn_finer_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns=columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        sparse_dnn_finer_df.loc[i, columns_remove_prob[j]] = sparse_dnn_finer_dic[i][columns_remove_prob[j]]
# rename sparse_dnn_finer_df due to some stupid mistake...(add "_" )
part_name = list(sparse_dnn_finer_df.columns[:-15])
part_name.extend([name[:-1] + '_' + name[-1:] for name in sparse_dnn_finer_df.columns[-15:]])
sparse_dnn_finer_df.columns = part_name

# create a dataframe for classifier accuracy
classifier_train = classifier_accuracy_dic['training']
classifier_validation = classifier_accuracy_dic['validation']
classifier_test = classifier_accuracy_dic['testing']
classifier_train = classifier_train.mean(1)
classifier_validation = classifier_validation.mean(1)
classifier_test = classifier_test.mean(1)

# create a dataframe for classifier accuracy (mlogit data)
classifier_train_mlogit = classifier_accuracy_train_dic['training']
classifier_validation_mlogit = classifier_accuracy_train_dic['validation']
classifier_test_mlogit = classifier_accuracy_train_dic['testing']
classifier_train_mlogit = classifier_train_mlogit.mean(1)
classifier_validation_mlogit = classifier_validation_mlogit.mean(1)
classifier_test_mlogit = classifier_test_mlogit.mean(1)

# create a dataframe for sparse dnn (mlogit data)
columns_ = list(sparse_dnn_train_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i]
n_rows = len(sparse_dnn_train_dic.keys())
sparse_dnn_train_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns=columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        sparse_dnn_train_df.loc[i, columns_remove_prob[j]] = sparse_dnn_train_dic[i][columns_remove_prob[j]]
# rename sparse_dnn_finer_df due to some stupid mistake...(add "_" )
part_name_1 = list(sparse_dnn_train_df.columns[:-15])
part_name_1.extend([name[:-1] + '_' + name[-1:] for name in sparse_dnn_train_df.columns[-15:]])
sparse_dnn_train_df.columns = part_name_1

# create a dataframe for full dnn (mlogit data)
columns_ = list(full_dnn_train_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i]
n_rows = len(full_dnn_train_dic.keys())
full_dnn_train_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns=columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        full_dnn_train_df.loc[i, columns_remove_prob[j]] = full_dnn_train_dic[i][columns_remove_prob[j]]


# compute average training, validation, and testing accuracies for finer HPO
def average_var_(df, var_name):
    # df = full_dnn_finer_df
    # var_name = 'validation_accuracy'
    df['average_' + var_name] = (1 / 5) * (
                df[var_name + '_0'] + df[var_name + '_1'] + df[var_name + '_2'] + df[var_name + '_3'] + df[
            var_name + '_4'])
    return df


#
full_dnn_train_df = average_var_(full_dnn_train_df, 'validation_accuracy')
full_dnn_train_df = average_var_(full_dnn_train_df, 'train_accuracy')
full_dnn_train_df = average_var_(full_dnn_train_df, 'test_accuracy')
sparse_dnn_train_df = average_var_(sparse_dnn_train_df, 'validation_accuracy')
sparse_dnn_train_df = average_var_(sparse_dnn_train_df, 'train_accuracy')
sparse_dnn_train_df = average_var_(sparse_dnn_train_df, 'test_accuracy')
full_dnn_finer_df = average_var_(full_dnn_finer_df, 'validation_accuracy')
full_dnn_finer_df = average_var_(full_dnn_finer_df, 'train_accuracy')
full_dnn_finer_df = average_var_(full_dnn_finer_df, 'test_accuracy')
sparse_dnn_finer_df = average_var_(sparse_dnn_finer_df, 'validation_accuracy')
sparse_dnn_finer_df = average_var_(sparse_dnn_finer_df, 'train_accuracy')
sparse_dnn_finer_df = average_var_(sparse_dnn_finer_df, 'test_accuracy')

# transform sparse_dnn_df to a better magnitude
sparse_dnn_df["M"] = sparse_dnn_df["M_before"] + sparse_dnn_df["M_after"] + 1
sparse_dnn_df["n_hidden"] = (sparse_dnn_df["n_hidden_before"] * 6 + sparse_dnn_df["n_hidden_after"] * 5) / 2.0
sparse_dnn_finer_df["M"] = sparse_dnn_finer_df["M_before"] + sparse_dnn_finer_df["M_after"] + 1
sparse_dnn_finer_df["n_hidden"] = (sparse_dnn_finer_df["n_hidden_before"] * 6 + sparse_dnn_finer_df[
    "n_hidden_after"] * 5) / 2.0


# transform the units of matrices
def transform_unit_df(df1):
    df = copy.copy(df1)
    df[['l1_const', 'l2_const', 'dropout_rate', 'learning_rate']] = \
        -np.log10(df[['l1_const', 'l2_const', 'dropout_rate', 'learning_rate']])
    df[['n_iteration', 'n_mini_batch']] = \
        np.log10(df[['n_iteration', 'n_mini_batch']])
    df['batch_normalization'] = np.float_(df['batch_normalization'])
    return df


full_dnn_df_trans_unit = transform_unit_df(full_dnn_df)
sparse_dnn_df_trans_unit = transform_unit_df(sparse_dnn_df)
full_dnn_finer_df_trans_unit = transform_unit_df(full_dnn_finer_df)
sparse_dnn_finer_df_trans_unit = transform_unit_df(sparse_dnn_finer_df)

# combine all the past hyper training results
hyper_vars = ['M', 'n_hidden', 'l1_const', 'l2_const', 'dropout_rate',
              'learning_rate', 'n_iteration', 'n_mini_batch', 'batch_normalization']
hyper_vars_sparse = ['M_before', 'M_after', 'n_hidden_before', 'n_hidden_after', 'l1_const', 'l2_const', 'dropout_rate',
                     'learning_rate', 'n_iteration', 'n_mini_batch', 'batch_normalization']
combined_df = \
    pd.DataFrame(np.concatenate(
        [full_dnn_df_trans_unit[hyper_vars + ['train_accuracy', 'validation_accuracy', 'test_accuracy']].values,
         full_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_0', 'validation_accuracy_0', 'test_accuracy_0']].values,
         full_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_1', 'validation_accuracy_1', 'test_accuracy_1']].values,
         full_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_2', 'validation_accuracy_2', 'test_accuracy_2']].values,
         full_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_3', 'validation_accuracy_3', 'test_accuracy_3']].values,
         full_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_4', 'validation_accuracy_4', 'test_accuracy_4']].values,
         sparse_dnn_df_trans_unit[hyper_vars + ['train_accuracy', 'validation_accuracy', 'test_accuracy']].values,
         sparse_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_0', 'validation_accuracy_0', 'test_accuracy_0']].values,
         sparse_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_1', 'validation_accuracy_1', 'test_accuracy_1']].values,
         sparse_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_2', 'validation_accuracy_2', 'test_accuracy_2']].values,
         sparse_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_3', 'validation_accuracy_3', 'test_accuracy_3']].values,
         sparse_dnn_finer_df_trans_unit[
             hyper_vars + ['train_accuracy_4', 'validation_accuracy_4', 'test_accuracy_4']].values],
        axis=0), columns=hyper_vars + ['train_accuracy', 'validation_accuracy', 'test_accuracy'])
# note the following works because first half data is from the full dnn, the second half from the sparse dnn.
# prepare datasets for R regression...
combined_df.loc[np.int(combined_df.shape[0] / 2.0):, 'sparse_dnn'] = 1
combined_df.loc[:np.int(combined_df.shape[0] / 2.0), 'sparse_dnn'] = 0
combined_df.to_csv('output/table_combined_prediction.csv', index=False)

# identify the best sparse and full models
best_sparse_dnn_index = np.argmax(sparse_dnn_finer_df['average_validation_accuracy'])
best_full_dnn_index = np.argmax(full_dnn_finer_df['average_validation_accuracy'])
top_ten_sparse_dnn_index = sparse_dnn_finer_df.sort_values('average_validation_accuracy', ascending=False).index[:10]
top_ten_full_dnn_index = full_dnn_finer_df.sort_values('average_validation_accuracy', ascending=False).index[:10]
best_sparse_dnn_train_index = np.argmax(sparse_dnn_train_df['average_validation_accuracy'])
best_full_dnn_train_index = np.argmax(full_dnn_train_df['average_validation_accuracy'])
top_ten_sparse_dnn_train_index = sparse_dnn_train_df.sort_values('average_validation_accuracy', ascending=False).index[
                                 :10]
top_ten_full_dnn_train_index = full_dnn_train_df.sort_values('average_validation_accuracy', ascending=False).index[:10]


############################################################
############################################################
# specify hyperparameter space for fully connected DNN
# M_list = [1,2,3,4,5,6,7,8,9,10,11,12] # 5
# n_hidden_list = [60, 120, 240, 360, 480, 600] # 6
# l1_const_list = [1e-3, 1e-5, 1e-10, 1e-20]# 8
# l2_const_list = [1e-3, 1e-5, 1e-10, 1e-20]# 8
# dropout_rate_list = [1e-3, 1e-5] # 5
# batch_normalization_list = [True, False] # 2
# learning_rate_list = [0.01, 1e-3, 1e-4, 1e-5] # 5
# n_iteration_list = [500, 1000, 5000, 10000, 20000] # 5
# n_mini_batch_list = [50, 100, 200, 500, 1000] # 5
##########################
# num_top_model = 10
# full_dnn_dic_top_10 = {}
# var_list_for_elast = ['walk_walktime','bus_cost','bus_ivt','ridesharing_cost','ridesharing_ivt',
#             'drive_cost','drive_ivt','av_cost','av_ivt']
#
# elast_records_full_dnn = {}
# elast_records_full_dnn_save = {}
#
# key_choice_index = ['Walk','PT','RH','Drive','AV']
#
# for i in range(num_top_model):
#     print("------------------------")
#     print("Estimate full connected model ", str(i))
#     index = top_ten_full_dnn_index[i]
#     # M = int(full_dnn_finer_df.loc[index,'M'])
#     # n_hidden = int(full_dnn_finer_df.loc[index,'n_hidden'])
#     # l1_const = full_dnn_finer_df.loc[index,'l1_const']
#     # l2_const = full_dnn_finer_df.loc[index,'l2_const']
#     # dropout_rate = full_dnn_finer_df.loc[index,'dropout_rate']
#     # batch_normalization = full_dnn_finer_df.loc[index,'batch_normalization']
#     # learning_rate = full_dnn_finer_df.loc[index,'batch_normalization']
#     # n_iteration =int(full_dnn_finer_df.loc[index,'n_iteration'])
#     # n_mini_batch = int(full_dnn_finer_df.loc[index,'n_mini_batch'])
#
#
#     M = full_dnn_finer_dic[index]['M']
#     n_hidden = full_dnn_finer_dic[index]['n_hidden']
#     l1_const = full_dnn_finer_dic[index]['l1_const']
#     l2_const = full_dnn_finer_dic[index]['l2_const']
#     dropout_rate = full_dnn_finer_dic[index]['dropout_rate']
#     batch_normalization = full_dnn_finer_dic[index]['batch_normalization']
#     learning_rate = full_dnn_finer_dic[index]['learning_rate']
#     n_iteration = full_dnn_finer_dic[index]['n_iteration']
#     n_mini_batch = full_dnn_finer_dic[index]['n_mini_batch']
#
#
#     # store information
#     full_dnn_dic_top_10[i] = {}
#     full_dnn_dic_top_10[i]['M'] = M
#     full_dnn_dic_top_10[i]['n_hidden'] = n_hidden
#     full_dnn_dic_top_10[i]['l1_const'] = l1_const
#     full_dnn_dic_top_10[i]['l2_const'] = l2_const
#     full_dnn_dic_top_10[i]['dropout_rate'] = dropout_rate
#     full_dnn_dic_top_10[i]['batch_normalization'] = batch_normalization
#     full_dnn_dic_top_10[i]['learning_rate'] = learning_rate
#     full_dnn_dic_top_10[i]['n_iteration'] = n_iteration
#     full_dnn_dic_top_10[i]['n_mini_batch'] = n_mini_batch
#     print(full_dnn_dic_top_10[i])
#
#     var_list_index = list(df_sp_train.columns)
#     for j in range(5):
#         # five fold training with cross validation
#         df_sp_train,df_sp_validation = generate_cross_validation_set(df_sp_combined_train, j)
#         X_train = df_sp_train.iloc[:, 1:].values
#
#         index_for_var_elas = [var_list_index.index(var) - 1 for var in var_list_for_elast] # -1 because we starts from 1:
#         Y_train = df_sp_train.iloc[:, 0].values
#         X_validation = df_sp_validation.iloc[:, 1:].values
#         Y_validation = df_sp_validation.iloc[:, 0].values
#         X_test = df_sp_test.iloc[:, 1:].values
#         Y_test = df_sp_test.iloc[:, 0].values
#         # training
#         elast_records = util.full_dnn_elasticity(X_train, Y_train, X_validation, Y_validation, X_test, Y_test,
#                                        M, n_hidden, l1_const, l2_const, dropout_rate, batch_normalization, learning_rate, n_iteration, n_mini_batch,
#                                                  index_for_var_elas, df_sp_test_nonstand, j, i, var_list_index)
#
#         new_col = ['K-fold','Model_name']
#         for key in elast_records.columns: # change index to name
#             if key != 'K-fold' and key != 'Model_name':
#                 mode = key_choice_index[int(key.split('___')[0])]
#                 var = key.split('___')[1]
#                 new_key = mode + '___' + var
#                 new_col.append(new_key)
#         elast_records.columns = new_col
#
#         if len(elast_records_full_dnn) == 0:
#             elast_records_full_dnn = pd.DataFrame(elast_records)
#         else:
#             elast_records_full_dnn = pd.concat([elast_records_full_dnn, pd.DataFrame(elast_records)])
#
#
#
#         # store information
# modes_list = ['Walk','PT','RH','AV','Drive']
# # save elasticity
# elast_records_full_dnn.to_csv('output/elasticity_full_DNN_raw.csv', index=False)
#
# elast_records_full_dnn_save = {'Variables': var_list_for_elast}
# for mode in modes_list:
#     elast_records_full_dnn_save[mode] = [0] * len(var_list_for_elast)
# elast_records_full_dnn_save = pd.DataFrame(elast_records_full_dnn_save)
# for col in elast_records_full_dnn.columns:
#     if col != 'K-fold' and col != 'Model_name':
#         mode = col.split('___')[0]
#         var = col.split('___')[1]
#         elast_records_full_dnn_save.loc[
#             elast_records_full_dnn_save['Variables'] == var, mode] = elast_records_full_dnn.loc[:, col].mean()
# elast_records_full_dnn_save.to_csv('output/elasticity_full_DNN.csv', index=False)

# ############################################################
# ############################################################
# specify hyperparameter space for sparsely connected DNN
# define hyperparameter space
# note: [0,1,2,3,4] meaning [walk,bus,ridesharing,drive,av]
# df_sp_train = pd.read_csv('data/data_AV_Singapore_v1_sp_train.csv')
# df_sp_validation = pd.read_csv('data/data_AV_Singapore_v1_sp_validation.csv')
# # here we combine train and validation set to recreate training and validation sets...
# df_sp_combined_train = pd.concat([df_sp_train, df_sp_validation], axis = 0)
# df_sp_combined_train.index = np.arange(df_sp_combined_train.shape[0])
# df_sp_test = pd.read_csv('data/data_AV_Singapore_v1_sp_test.csv')
#
# y_vars = ['choice']
# z_vars = ['male', 'young_age', 'old_age', 'low_edu', 'high_edu',
#           'low_inc', 'high_inc', 'full_job', 'age', 'inc', 'edu']
# x0_vars = ['walk_walktime']
# x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
# x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
# x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
# x4_vars = ['av_cost', 'av_waittime', 'av_ivt']
#
# all_elas_var = {'x0_vars':['walk_walktime'],
#                  'x1_vars':['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt'],
#                  'x2_vars':['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt'],
#                  'x3_vars':['drive_cost', 'drive_walktime', 'drive_ivt'],
#                  'x4_vars':['av_cost', 'av_waittime', 'av_ivt']}
#
# # random draw...and HPO
# num_top_model = 10
# sparse_dnn_dic_top10 = {}
#
# elast_records_sparse_dnn = {}
# elast_records_sparse_dnn_save = {}
#
# key_choice_index = ['Walk','PT','RH','Drive','AV']
#
# for i in range(num_top_model):
#     print("------------------------")
#     print("Estimate sparse connected model ", str(i))
#     index = top_ten_sparse_dnn_index[i]
#
#
#     M_before = sparse_dnn_finer_dic[index]['M_before']
#     M_after = sparse_dnn_finer_dic[index]['M_after']
#     n_hidden_before = sparse_dnn_finer_dic[index]['n_hidden_before']
#     n_hidden_after = sparse_dnn_finer_dic[index]['n_hidden_after']
#     l1_const = sparse_dnn_finer_dic[index]['l1_const']
#     l2_const = sparse_dnn_finer_dic[index]['l2_const']
#     dropout_rate = sparse_dnn_finer_dic[index]['dropout_rate']
#     batch_normalization = sparse_dnn_finer_dic[index]['batch_normalization']
#     learning_rate = sparse_dnn_finer_dic[index]['learning_rate']
#     n_iteration = sparse_dnn_finer_dic[index]['n_iteration']
#     n_mini_batch = sparse_dnn_finer_dic[index]['n_mini_batch']
#
#
#
#     # store information
#     sparse_dnn_dic[i] = {}
#     sparse_dnn_dic[i]['M_before'] = M_before
#     sparse_dnn_dic[i]['M_after'] = M_after
#     sparse_dnn_dic[i]['n_hidden_before'] = n_hidden_before
#     sparse_dnn_dic[i]['n_hidden_after'] = n_hidden_after
#     sparse_dnn_dic[i]['l1_const'] = l1_const
#     sparse_dnn_dic[i]['l2_const'] = l2_const
#     sparse_dnn_dic[i]['dropout_rate'] = dropout_rate
#     sparse_dnn_dic[i]['batch_normalization'] = batch_normalization
#     sparse_dnn_dic[i]['learning_rate'] = learning_rate
#     sparse_dnn_dic[i]['n_iteration'] = n_iteration
#     sparse_dnn_dic[i]['n_mini_batch'] = n_mini_batch
#     print(sparse_dnn_dic[i])
#
#     for j in range(5):
#         # five fold training with cross validation
#         df_sp_train,df_sp_validation = generate_cross_validation_set(df_sp_combined_train, j)
#         X0_train = df_sp_train[x0_vars].values
#         X1_train = df_sp_train[x1_vars].values
#         X2_train = df_sp_train[x2_vars].values
#         X3_train = df_sp_train[x3_vars].values
#         X4_train = df_sp_train[x4_vars].values
#         Y_train = df_sp_train[y_vars].values.reshape(-1)
#         Z_train = df_sp_train[z_vars].values
#
#         X0_validation = df_sp_validation[x0_vars].values
#         X1_validation = df_sp_validation[x1_vars].values
#         X2_validation = df_sp_validation[x2_vars].values
#         X3_validation = df_sp_validation[x3_vars].values
#         X4_validation = df_sp_validation[x4_vars].values
#         Y_validation = df_sp_validation[y_vars].values.reshape(-1)
#         Z_validation = df_sp_validation[z_vars].values
#
#         X0_test = df_sp_test[x0_vars].values
#         X1_test = df_sp_test[x1_vars].values
#         X2_test = df_sp_test[x2_vars].values
#         X3_test = df_sp_test[x3_vars].values
#         X4_test = df_sp_test[x4_vars].values
#         Y_test = df_sp_test[y_vars].values.reshape(-1)
#         Z_test = df_sp_test[z_vars].values
#
#         # one estimation here
#
#
#
#         elast_records = util.dnn_alt_spec_elasticity(X0_train,X1_train,X2_train,X3_train,X4_train,Y_train,Z_train,
#                                             X0_validation,X1_validation,X2_validation,X3_validation,X4_validation,Y_validation,Z_validation,
#                                             X0_test,X1_test,X2_test,X3_test,X4_test,Y_test,Z_test,
#                                             M_before,M_after,n_hidden_before,n_hidden_after,l1_const,l2_const,
#                                             dropout_rate,batch_normalization,learning_rate,n_iteration,n_mini_batch,
#                                                      all_elas_var, df_sp_test_nonstand,j,i)
#
#         new_col = ['K-fold','Model_name']
#         for key in elast_records.columns: # change index to name
#             if key != 'K-fold' and key != 'Model_name':
#                 mode = key_choice_index[int(key.split('___')[0])]
#                 var = key.split('___')[1]
#                 new_key = mode + '___' + var
#                 new_col.append(new_key)
#         elast_records.columns = new_col
#
#         if len(elast_records_sparse_dnn) == 0:
#             elast_records_sparse_dnn = pd.DataFrame(elast_records)
#         else:
#             elast_records_sparse_dnn = pd.concat([elast_records_sparse_dnn, pd.DataFrame(elast_records)])
#
#         # store information
#
#



# save elasticity
elast_records_sparse_dnn.to_csv('output/elasticity_sparse_DNN_raw.csv', index=False)

var_list_for_elast = ['walk_walktime','bus_cost','bus_ivt','ridesharing_cost','ridesharing_ivt',
            'drive_cost','drive_ivt','av_cost','av_ivt']
modes_list = ['Walk','PT','RH','AV','Drive']

elast_records_sparse_dnn_save = {'Variables': var_list_for_elast}
for mode in modes_list:
    elast_records_sparse_dnn_save[mode] = [0] * len(var_list_for_elast)
elast_records_sparse_dnn_save = pd.DataFrame(elast_records_sparse_dnn_save)
for col in elast_records_sparse_dnn.columns:
    if col != 'K-fold' and col != 'Model_name':
        mode = col.split('___')[0]
        var = col.split('___')[1]
        elast_records_sparse_dnn_save.loc[
            elast_records_sparse_dnn_save['Variables'] == var, mode] = elast_records_sparse_dnn.loc[:, col].mean()
elast_records_sparse_dnn_save.to_csv('output/elasticity_sparse_DNN.csv', index=False)












   



