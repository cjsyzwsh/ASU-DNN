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

# starting time
start_time = time.time()

#%matplotlib inline
df_sp_train = pd.read_csv('data/data_AV_Singapore_v1_sp_train.csv')
df_sp_validation = pd.read_csv('data/data_AV_Singapore_v1_sp_validation.csv')
# here we combine train and validation set to recreate training and validation sets...
df_sp_combined_train = pd.concat([df_sp_train, df_sp_validation], axis = 0)
df_sp_combined_train.index = np.arange(df_sp_combined_train.shape[0])
df_sp_test = pd.read_csv('data/data_AV_Singapore_v1_sp_test.csv')

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


training_accuracy_table = pd.DataFrame(np.zeros((len(model_list), 5)), index = model_titles)
validation_accuracy_table = pd.DataFrame(np.zeros((len(model_list), 5)), index = model_titles)
testing_accuracy_table = pd.DataFrame(np.zeros((len(model_list), 5)), index = model_titles)

for j in range(5):
    # five fold training with cross validation
    df_sp_train,df_sp_validation = generate_cross_validation_set(df_sp_combined_train, j)
    X_train = df_sp_train.iloc[:, 1:].values
    Y_train = df_sp_train.iloc[:, 0].values
    X_validation = df_sp_validation.iloc[:, 1:].values
    Y_validation = df_sp_validation.iloc[:, 0].values
    X_test = df_sp_test.iloc[:, 1:].values
    Y_test = df_sp_test.iloc[:, 0].values
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
        df_sp_train,df_sp_validation = generate_cross_validation_set(df_sp_combined_train, j)
        X_train = df_sp_train.iloc[:, 1:].values
        Y_train = df_sp_train.iloc[:, 0].values
        X_validation = df_sp_validation.iloc[:, 1:].values 
        Y_validation = df_sp_validation.iloc[:, 0].values
        X_test = df_sp_test.iloc[:, 1:].values 
        Y_test = df_sp_test.iloc[:, 0].values
        # training
        train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt = \
                        util.dnn_estimation(X_train, Y_train, X_validation, Y_validation, X_test, Y_test,
                                       M, n_hidden, l1_const, l2_const, dropout_rate, batch_normalization, learning_rate, n_iteration, n_mini_batch)
        print("Training accuracy is ", train_accuracy)
        print("Validation accuracy is ", validation_accuracy)
        print("Testing accuracy is ", test_accuracy)

        # store information        
        full_dnn_dic[i]['train_accuracy_'+str(j)] = train_accuracy
        full_dnn_dic[i]['validation_accuracy_'+str(j)] = validation_accuracy            
        full_dnn_dic[i]['test_accuracy_'+str(j)] = test_accuracy            
        full_dnn_dic[i]['prob_cost_'+str(j)] = prob_cost
        full_dnn_dic[i]['prob_ivt_'+str(j)] = prob_ivt

# full dnn training time 
full_dnn_complete_time = time.time()
                    
############################################################
############################################################
# specify hyperparameter space for sparsely connected DNN
# define hyperparameter space
# note: [0,1,2,3,4] meaning [walk,bus,ridesharing,drive,av]
df_sp_train = pd.read_csv('../data/data_AV_Singapore_v1_sp_train.csv')
df_sp_validation = pd.read_csv('../data/data_AV_Singapore_v1_sp_validation.csv')
# here we combine train and validation set to recreate training and validation sets...
df_sp_combined_train = pd.concat([df_sp_train, df_sp_validation], axis = 0)
df_sp_combined_train.index = np.arange(df_sp_combined_train.shape[0])
df_sp_test = pd.read_csv('../data/data_AV_Singapore_v1_sp_test.csv')

y_vars = ['choice']
z_vars = ['male', 'young_age', 'old_age', 'low_edu', 'high_edu',
          'low_inc', 'high_inc', 'full_job', 'age', 'inc', 'edu']
x0_vars = ['walk_walktime']
x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
x4_vars = ['av_cost', 'av_waittime', 'av_ivt']

M_before_list = [0,1,2,3,4,5]
M_after_list = [0,1,2,3,4,5]
n_hidden_before_list = [10, 20, 40, 60, 80, 100]
n_hidden_after_list = [10, 20, 40, 60, 80, 100]
l1_const_list = [1e-3, 1e-5, 1e-10, 1e-20]# 8
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

    M_before = np.random.choice(M_before_list, size = 1)[0]
    M_after = np.random.choice(M_after_list, size = 1)[0]
    n_hidden_before = np.random.choice(n_hidden_before_list, size = 1)[0]
    n_hidden_after = np.random.choice(n_hidden_after_list, size = 1)[0]
    l1_const = np.random.choice(l1_const_list, size = 1)[0]
    l2_const = np.random.choice(l2_const_list, size = 1)[0]
    dropout_rate = np.random.choice(dropout_rate_list, size = 1)[0]
    batch_normalization = np.random.choice(batch_normalization_list, size = 1)[0]
    learning_rate = np.random.choice(learning_rate_list, size = 1)[0]
    n_iteration = np.random.choice(n_iteration_list, size = 1)[0]
    n_mini_batch = np.random.choice(n_mini_batch_list, size = 1)[0]

    # store information
    sparse_dnn_dic[i] = {}
    sparse_dnn_dic[i]['M_before'] = M_before
    sparse_dnn_dic[i]['M_after'] = M_after
    sparse_dnn_dic[i]['n_hidden_before'] = n_hidden_before
    sparse_dnn_dic[i]['n_hidden_after'] = n_hidden_after
    sparse_dnn_dic[i]['l1_const'] = l1_const
    sparse_dnn_dic[i]['l2_const'] = l2_const
    sparse_dnn_dic[i]['dropout_rate'] = dropout_rate
    sparse_dnn_dic[i]['batch_normalization'] = batch_normalization
    sparse_dnn_dic[i]['learning_rate'] = learning_rate
    sparse_dnn_dic[i]['n_iteration'] = n_iteration
    sparse_dnn_dic[i]['n_mini_batch'] = n_mini_batch
    print(sparse_dnn_dic[i])              

    for j in range(5):
        # five fold training with cross validation
        df_sp_train,df_sp_validation = generate_cross_validation_set(df_sp_combined_train, j)
        X0_train = df_sp_train[x0_vars].values
        X1_train = df_sp_train[x1_vars].values
        X2_train = df_sp_train[x2_vars].values
        X3_train = df_sp_train[x3_vars].values
        X4_train = df_sp_train[x4_vars].values
        Y_train = df_sp_train[y_vars].values.reshape(-1)
        Z_train = df_sp_train[z_vars].values
        
        X0_validation = df_sp_validation[x0_vars].values
        X1_validation = df_sp_validation[x1_vars].values
        X2_validation = df_sp_validation[x2_vars].values
        X3_validation = df_sp_validation[x3_vars].values
        X4_validation = df_sp_validation[x4_vars].values
        Y_validation = df_sp_validation[y_vars].values.reshape(-1)
        Z_validation = df_sp_validation[z_vars].values
        
        X0_test = df_sp_test[x0_vars].values
        X1_test = df_sp_test[x1_vars].values
        X2_test = df_sp_test[x2_vars].values
        X3_test = df_sp_test[x3_vars].values
        X4_test = df_sp_test[x4_vars].values
        Y_test = df_sp_test[y_vars].values.reshape(-1)
        Z_test = df_sp_test[z_vars].values

        # one estimation here
        train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt = \
                    util.dnn_alt_spec_estimation(X0_train,X1_train,X2_train,X3_train,X4_train,Y_train,Z_train,
                                            X0_validation,X1_validation,X2_validation,X3_validation,X4_validation,Y_validation,Z_validation,
                                            X0_test,X1_test,X2_test,X3_test,X4_test,Y_test,Z_test,
                                            M_before,M_after,n_hidden_before,n_hidden_after,l1_const,l2_const,
                                            dropout_rate,batch_normalization,learning_rate,n_iteration,n_mini_batch)
        print("Training accuracy is ", train_accuracy)
        print("Validation accuracy is ", validation_accuracy)
        print("Testing accuracy is ", test_accuracy)
    
        # store information
        sparse_dnn_dic[i]['train_accuracy'+str(j)] = train_accuracy
        sparse_dnn_dic[i]['validation_accuracy'+str(j)] = validation_accuracy
        sparse_dnn_dic[i]['test_accuracy'+str(j)] = test_accuracy
        sparse_dnn_dic[i]['prob_cost'+str(j)] = prob_cost
        sparse_dnn_dic[i]['prob_ivt'+str(j)] = prob_ivt

# sparse dnn training time 
sparse_dnn_complete_time = time.time()

# export the dictionary
import pickle
with open('output/full_dnn_results_finer.pickle', 'wb') as full_dnn_results_finer:
    pickle.dump(full_dnn_dic, full_dnn_results_finer, protocol=pickle.HIGHEST_PROTOCOL)
with open('output/sparse_dnn_results_finer.pickle', 'wb') as sparse_dnn_results_finer:
    pickle.dump(sparse_dnn_dic, sparse_dnn_results_finer, protocol=pickle.HIGHEST_PROTOCOL)
with open('output/classifiers_accuracy.pickle', 'wb') as data:
    pickle.dump(classifiers_accuracy, data, protocol=pickle.HIGHEST_PROTOCOL)

print("Full DNN training time is: ", (full_dnn_complete_time - start_time)/3600) # 30 hours
print("Sparse DNN training time is: ", (sparse_dnn_complete_time - full_dnn_complete_time)/3600) # 7 hours



#for i in range(50):
#    print("---------")
#    print(sparse_dnn_dic[i]['M_before'])
#    print(sparse_dnn_dic[i]['M_after'])
#    print(sparse_dnn_dic[i]['n_iteration'])
#    print(sparse_dnn_dic[i]['n_mini_batch'])










   



