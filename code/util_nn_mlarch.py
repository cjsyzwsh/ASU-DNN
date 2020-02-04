"""
Created on Mon Jul 23 18:38:45 2018

@author: shenhao
"""

#cd /Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/9_ml_dnn_alt_spe_util/code

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import scipy.stats as ss
import pickle

#%matplotlib inline
### need to code an DNN object that controls the local connectivity across input variables.
### input data:
#data["training"]; data["testing"]; both are dataframes.
#Use input vars to create training_dic and testing_dic
#Use K
#Use reg_const_dic: the group lasso constants. Assume symmetry. e.g. l00, l01, l02, l11, l12, l22 (K=3; 6 reg constants)

class FeedForward_DNN:
    def __init__(self, K, MODEL_NAME, data_input, var_name_dic):
        self.graph = tf.Graph()
        self.K = K
        self.MODEL_NAME = MODEL_NAME
        self.data_input=data_input
        self.var_name_dic = var_name_dic # {'x0': [], 'x1': [], 'x2': [], ...; 'y': []}. Note that the x has combined the z for each.

    def load_data(self):
        print("Loading datasets...")
#        with open(self.input_file, 'rb') as f:
#            save = pickle.load(f)
        self.data = {}
        for i in range(self.K):
            self.data['x'+str(i)+'_training'] = self.data_input['training'][self.var_name_dic['x'+str(i)]].values
            self.data['x'+str(i)+'_testing'] = self.data_input['testing'][self.var_name_dic['x'+str(i)]].values
        self.data['y'+'_training'] = self.data_input['training'][self.var_name_dic['y']].values[:, 0]
        self.data['y'+'_testing'] = self.data_input['testing'][self.var_name_dic['y']].values[:, 0]      
        print("Training observations: ", self.data['x0_training'].shape[0])
        print("Testing observations: ", self.data['x0_testing'].shape[0])

    def init_hyperparameter(self):
        # h stands for hyperparameter
        self.h = {}
        self.h['M']=3 # number of hidden layer; min = 1
        self.h['n_hidden']=20
        self.h['dropout_rate']=1e-3
        self.h['batch_normalization']=False
        self.h['learning_rate']=1e-3
        self.h['n_iteration']=1000
        self.h['n_mini_batch']=100
        # reg_const_dic
        self.h['reg_const_dic'] = {}
        for j in range(self.K):
            for i in range(self.K):
                name = "l"+str(j)+str(i)
                self.h['reg_const_dic'][name] = 0.01
            
    def change_hyperparameter(self, new_hyperparameter):
        assert bool(self.h) == True
        self.h = new_hyperparameter
    
    def init_hyperparameter_space(self):
        # hs stands for hyperparameter_space
        self.hs = {}
        # reg constant list
        self.hs['reg_const_dic_list'] = [1.0, 0.1, 0.01, 1e-3, 1e-5, 1e-10, 1e-20] # 8
        # other hyperparameters
        self.hs['M_list'] = [1,2,3,4,5,6,7,8,9,10,11,12] # number of hidden layer; min = 1
        self.hs['n_hidden_list'] = [10, 20, 30, 40, 50, 60] # 6
        self.hs['dropout_rate_list'] = [1e-3, 1e-5] # 5
        self.hs['batch_normalization_list'] = [True, False] # 2
        self.hs['learning_rate_list'] = [0.01, 1e-3, 1e-4, 1e-5] # 5
        self.hs['n_iteration_list'] = [500, 1000, 5000, 10000, 20000] # 5
        self.hs['n_mini_batch_list'] = [50, 100, 200, 500, 1000] # 5
    
    def random_sample_hyperparameter(self):
        assert bool(self.hs) == True
        assert bool(self.h) == True
        for name_ in self.h.keys():
            if name_ != 'reg_const_dic':
                self.h[name_] = np.random.choice(self.hs[name_+'_list'])
            elif name_ == 'reg_const_dic':
                for reg_const_key in self.h['reg_const_dic'].keys():
                    self.h['reg_const_dic'][reg_const_key] = np.random.choice(self.hs['reg_const_dic_list'])

    def obtain_mini_batch(self):
        self.data_batch = {}
        N, _ = self.data['x0_training'].shape
        index = np.random.choice(N, size = self.h['n_mini_batch'])

        for i in range(self.K):
            self.data_batch['x'+str(i)+'_training'] = self.data['x'+str(i)+'_training'][index, :]
        self.data_batch['y'+'_training'] = self.data['y_training'][index]
        
    def standard_hidden_layer(self, name):
        # standard layer, repeated in the following for loop.
        self.hidden = tf.layers.dense(self.hidden, self.h['n_hidden'], activation = tf.nn.relu, name = name)
        if self.h['batch_normalization'] == True:
            self.hidden = tf.layers.batch_normalization(inputs = self.hidden, axis = 1)
        self.hidden = tf.layers.dropout(inputs = self.hidden, rate = self.h['dropout_rate'])

    def build_model(self):
        with self.graph.as_default():
            # record the dimensionality
            self.D_dic = {}
            for i in range(self.K):
                self.D_dic['D'+str(i)] = self.data['x'+str(i)+'_training'].shape[1]
            # initialize inputs
            self.input = {}
            for i in range(self.K):
                self.input['x'+str(i)] = tf.placeholder(dtype = tf.float32, shape = (None, self.D_dic['D'+str(i)]), name = 'x'+str(i))
                self.input['y'] = tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')

            # initialize parameters
            # input layer
#            self.param = {}
#            self.param['input'] = {}
#            self.param['hidden'] = {}
#            self.param['output'] = {}            
#            for j in range(self.K):
#                for i in range(j, self.K):
#                    self.param['input'][str(j)+str(i)] = 

            self.hidden_tmp = {}
            self.hidden_layer = {}

            ### inputs
#            with tf.name_scope('input'):
            # map
            for i in range(self.K):
                for j in range(self.K):
                    self.hidden_tmp[str(i)+str(j)] = tf.layers.dense(self.input['x'+str(i)], self.h['n_hidden'], activation = tf.nn.relu, 
                                                                     name = 'input_from_'+str(i)+'_to_'+str(j))

            # store hidden layer 0
            self.hidden_layer[0]={}
            for i in range(self.K):
                self.hidden_layer[0][str(i)] = self.hidden_tmp['0'+str(i)]
                for j in range(1, self.K):
                    # add the first string in hidden_tmp
                    self.hidden_layer[0][str(i)] += self.hidden_tmp[str(j)+str(i)]
                    
                    
            ### hidden layers            
#            with tf.name_scope('hidden'):
            # map
            for m in range(self.h['M']-1):                
                for i in range(self.K):
                    for j in range(self.K):
                        self.hidden_tmp[str(i)+str(j)] = tf.layers.dense(self.hidden_layer[m][str(i)], self.h['n_hidden'], activation = tf.nn.relu,
                                                                         name = 'hidden_'+str(m)+'_from_'+str(i)+'_to_'+str(j))
                # store hidden layer 1
                self.hidden_layer[m+1]={}
                for i in range(self.K):
                    self.hidden_layer[m+1][str(i)] = self.hidden_tmp['0'+str(i)] 
                    for j in range(1, self.K):
                        # add the first string in hidden_tmp
                        self.hidden_layer[m+1][str(i)] += self.hidden_tmp[str(j)+str(i)]

            ### outputs            
            for i in range(self.K):
                for j in range(self.K):
                    self.hidden_tmp[str(i)+str(j)] = tf.layers.dense(self.hidden_layer[self.h['M']-1][str(i)], 1, activation = tf.nn.relu,
                                                                     name = 'output_from_'+str(i)+'_to_'+str(j)) # map to dim = 1 (each util...)
            # store hidden layer 2
            self.hidden_layer[self.h['M']]={}
            for i in range(self.K):
                self.hidden_layer[self.h['M']][str(i)] = self.hidden_tmp['0'+str(i)] 
                for j in range(1, self.K):
                    # add the first string in hidden_tmp
                    self.hidden_layer[self.h['M']][str(i)] += self.hidden_tmp[str(j)+str(i)]

            # save outputs
            self.output = self.hidden_layer[self.h['M']][str(0)]
            for i in range(1, self.K):
                self.output = tf.concat([self.output, self.hidden_layer[self.h['M']][str(i)]], axis = 1)
            self.prob=tf.nn.softmax(self.output)
            
            ### impose group LASSO.
            vars_ = tf.trainable_variables()
            vars_dic = {}            
            l1_l2_regularizer_dic={}
            regularization_penalty_dic={}
#            print(vars_)
            for i in range(self.K):
                for j in range(self.K):
                    vars_dic[str(i)+str(j)]=[var_ for var_ in vars_ if 'from_'+str(i)+'_to_'+str(j) in var_.name]
                    l1_l2_regularizer_dic[str(i)+str(j)]=tf.contrib.layers.l1_l2_regularizer(scale_l1=self.h['reg_const_dic']['l'+str(i)+str(j)], 
                                                                                          scale_l2=self.h['reg_const_dic']['l'+str(i)+str(j)])
                    regularization_penalty_dic[str(i)+str(j)]=tf.contrib.layers.apply_regularization(l1_l2_regularizer_dic[str(i)+str(j)], vars_dic[str(i)+str(j)])

            # costs
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.output, labels = self.input['y']), name = 'cost')
            # add regularization costs
            for i in range(self.K):
                for j in range(self.K):
                    self.cost += regularization_penalty_dic[str(i)+str(j)]

            #                     
            correct = tf.nn.in_top_k(self.output, self.input['y'], 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.h['learning_rate']) # opt objective
            self.training_op = self.optimizer.minimize(self.cost) # minimize the opt objective
            self.init = tf.global_variables_initializer()  
            self.saver= tf.train.Saver()                    
                        
    def train_model(self):
        with tf.Session(graph=self.graph) as sess:
            self.init.run()
            # create feed_training and testing
            feed_training = {}
            feed_testing = {}
            for j in range(self.K):
                feed_training[self.input['x'+str(j)]] = self.data['x'+str(j)+'_training']
                feed_testing[self.input['x'+str(j)]] = self.data['x'+str(j)+'_testing']
            feed_training[self.input['y']] = self.data['y_training']
            feed_testing[self.input['y']] = self.data['y_testing']
            
            for i in range(self.h['n_iteration']):
                # create feed_mini_batch
                feed_mini_batch = {}
                self.obtain_mini_batch()
                for j in range(self.K):
                    feed_mini_batch[self.input['x'+str(j)]] = self.data_batch['x'+str(j)+'_training']
                feed_mini_batch[self.input['y']] = self.data_batch['y_training']

                ### training 
                if i%50==0:
                    print("Iteration ", i, "Cost = ", self.cost.eval(feed_dict = feed_training))
                # gradient descent
                sess.run(self.training_op, feed_dict = feed_mini_batch)
            
            ## compute accuracy and loss
            self.accuracy_training = self.accuracy.eval(feed_dict = feed_training)
            self.accuracy_testing = self.accuracy.eval(feed_dict = feed_testing)
            self.loss_training = self.cost.eval(feed_dict = feed_training)
            self.loss_testing = self.cost.eval(feed_dict = feed_testing)

            ## compute util and prob
            self.util_training = self.output.eval(feed_dict=feed_training)
            self.util_testing = self.output.eval(feed_dict=feed_testing)
            self.prob_training = self.prob.eval(feed_dict=feed_training)
            self.prob_testing = self.prob.eval(feed_dict=feed_testing)
            ## save
            self.saver.save(sess, "tmp/"+self.MODEL_NAME+".ckpt")

    def init_simul_data(self):
        self.simul_data_dic = {}

    def create_one_simul_data(self, x_col_name, x_delta):
        # create a dataset in which only targetting x is ranging from min to max. All others are at mean value.
        # add it to the self.simul_data_dic
        # use min and max values in testing set to create the value range
        target_x_index = self.colnames.index(x_col_name)
        self.N_steps = np.int((np.max(self.X_test[:,target_x_index]) - np.min(self.X_test[:,target_x_index]))/x_delta) + 1
        data_x_target_varying = np.tile(np.mean(self.X_test, axis = 0), (self.N_steps, 1))
        data_x_target_varying[:, target_x_index] = np.arange(np.min(self.X_test[:,target_x_index]), np.max(self.X_test[:,target_x_index]), x_delta)
        self.simul_data_dic[x_col_name] = data_x_target_varying
        
    def compute_simul_data(self):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, "tmp/"+self.MODEL_NAME+".ckpt")
            # compute util and prob
            self.util_simul_dic={}
            self.prob_simul_dic={}
            for name_ in self.simul_data_dic.keys():
                self.util_simul_dic[name_]=self.output.eval(feed_dict={self.X:self.simul_data_dic[name_]})
                self.prob_simul_dic[name_]=self.prob.eval(feed_dict={self.X:self.simul_data_dic[name_]})


#
###def estimate_one_F_DNN(i):
##K=5
##MODEL_NAME = 'model'
##input_file="data/SGP_SP.pickle"
### 
##F_DNN = FeedForward_DNN(K,MODEL_NAME,data_input)
##F_DNN.init_hyperparameter()
##F_DNN.load_data()
##new_h = {'M': 1,
## 'batch_normalization': False,
## 'dropout_rate': 0.001,
## 'l1_const': 1e-05,
## 'l2_const': 1e-05,
## 'learning_rate': 0.001,
## 'n_hidden': 100,
## 'n_iteration': 1000,
## 'n_mini_batch': 100}
##F_DNN.change_hyperparameter(new_h)
### train
##F_DNN.build_model()
##F_DNN.train_model()
#
#
#################
## prepare data
#df_sp_train = pd.read_csv('data/data_AV_Singapore_v1_sp_train.csv')
#df_sp_validation = pd.read_csv('data/data_AV_Singapore_v1_sp_validation.csv')
## here we combine train and validation set to recreate training and validation sets...
#df_sp_combined_train = pd.concat([df_sp_train, df_sp_validation], axis = 0)
#df_sp_combined_train.index = np.arange(df_sp_combined_train.shape[0])
#df_sp_test = pd.read_csv('data/data_AV_Singapore_v1_sp_test.csv')
#
#data_input = {}
#data_input['training'] = df_sp_train
#data_input['testing'] = df_sp_test
#
## var_name_dic
#y_vars = ['choice']
#z_vars = ['male', 'young_age', 'old_age', 'low_edu', 'high_edu',
#          'low_inc', 'high_inc', 'full_job', 'age', 'inc', 'edu']
#x0_vars = ['walk_walktime']
#x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
#x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
#x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
#x4_vars = ['av_cost', 'av_waittime', 'av_ivt']
#x0_vars.extend(z_vars)
#x1_vars.extend(z_vars)
#x2_vars.extend(z_vars)
#x3_vars.extend(z_vars)
#x4_vars.extend(z_vars)
#var_name_dic = {}
#var_name_dic['x0'] = x0_vars
#var_name_dic['x1'] = x1_vars
#var_name_dic['x2'] = x2_vars
#var_name_dic['x3'] = x3_vars
#var_name_dic['x4'] = x4_vars
#var_name_dic['y'] = y_vars
#
## new hyper
#reg_const = 1e-10
#new_h = {'M': 3,
# 'batch_normalization': False,
# 'dropout_rate': 0.001,
# 'learning_rate': 0.001,
# 'n_hidden': 20,
# 'n_iteration': 500,
# 'n_mini_batch': 100,
# 'reg_const_dic': {'l00': 1e-10, 'l01': reg_const, 'l02': reg_const, 'l03': reg_const, 'l04': reg_const, 
#                   'l10': reg_const, 'l11': 1e-10, 'l12': reg_const, 'l13': reg_const, 'l14': reg_const, 
#                   'l20': reg_const, 'l21': reg_const, 'l22': 1e-10, 'l23': reg_const, 'l24': reg_const, 
#                   'l30': reg_const, 'l31': reg_const, 'l32': reg_const, 'l33': 1e-10, 'l34': reg_const, 
#                   'l40': reg_const, 'l41': reg_const, 'l42': reg_const, 'l43': reg_const, 'l44': 1e-10}} # ij: from block i to block j.
## 
#K = 5
#MODEL_NAME = 'model'
#
#####
#F_DNN = FeedForward_DNN(K,MODEL_NAME,data_input,var_name_dic)
#F_DNN.load_data()
#F_DNN.init_hyperparameter()
#F_DNN.change_hyperparameter(new_h)
#print(F_DNN.h)
##F_DNN.init_hyperparameter_space()
##F_DNN.random_sample_hyperparameter()
##print(F_DNN.h)
#
## 
#F_DNN.build_model()
#F_DNN.train_model()
#
#print(F_DNN.accuracy_training)
#print(F_DNN.accuracy_testing)
#
#
### save only training and testing accuracy
##out_file = open("save/train_"+str(i)+".txt", 'w')
##out_file.write(str(F_DNN.accuracy_test))
##
##    print(F_DNN.accuracy_test,F_DNN.accuracy_train)
##    return F_DNN.accuracy_test,F_DNN.accuracy_train
##
##if __name__ == '__main__':
##    estimate_one_F_DNN(sys.argv[1])
#
#
#
#
#




def obtain_mini_batch(X, Y, n_mini_batch):
    '''
    Return mini_batch
    '''
    N, D = X.shape                     
    index = np.random.choice(N, size = n_mini_batch)     
    X_batch = X[index, :]
    Y_batch = Y[index]
    return X_batch, Y_batch


def standard_hidden_layer(input_, n_hidden, l1_const, dropout_rate, batch_normalization, name):
    # standard layer, repeated in the following for loop.
    regularizer = tf.contrib.layers.l1_regularizer(scale=l1_const)
    hidden = tf.layers.dense(input_, n_hidden, activation = tf.nn.relu, name = name, kernel_regularizer = regularizer)
    if batch_normalization == True:
        hidden = tf.layers.batch_normalization(inputs = hidden, axis = 1)
    hidden = tf.layers.dropout(inputs = hidden, rate = dropout_rate)
    return hidden


def dnn_estimation(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, 
                   M, n_hidden, l1_const, l2_const, dropout_rate, batch_normalization, learning_rate, n_iterations, n_mini_batch, K = 5, 
                   Train = False):
    # repeat the standard_hidden_layer to construct one DNN architecture.
    
    ### build DNN models here
    tf.reset_default_graph() 
    N, D = X_train.shape
    # default
    
    X = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'X')
    Y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Y')
    
    hidden = X
    
    for i in range(M):
        name = 'hidden'+str(i)
        hidden = standard_hidden_layer(hidden, n_hidden, l1_const, dropout_rate, batch_normalization, name)
    output = tf.layers.dense(hidden, K, name = 'output')
    
    # add l2 regularization here
    l2_regularization = tf.contrib.layers.l2_regularizer(scale=l2_const, scope=None)
    vars_ = tf.trainable_variables()
    weights = [var_ for var_ in vars_ if 'kernel' in var_.name]
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularization, weights)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output, labels = Y), name = 'cost')
        cost += tf.losses.get_regularization_loss()
        cost += regularization_penalty
        
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(output, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective
    init = tf.global_variables_initializer()  
        

    ### estimate DNN models here
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # always run this to train the model
        init.run()
        for i in range(n_iterations):
            if i % 500 == 0:
                print("Epoch", i, "Cost = ", cost.eval(feed_dict = {X: X_train, Y: Y_train}))
            
            # gradient descent
            X_batch, Y_batch = obtain_mini_batch(X_train, Y_train, n_mini_batch)
            sess.run(training_op, feed_dict = {X: X_batch, Y: Y_batch})
        train_accuracy = accuracy.eval(feed_dict = {X: X_train, Y: Y_train})
        validation_accuracy = accuracy.eval(feed_dict = {X: X_validation, Y: Y_validation})
        test_accuracy = accuracy.eval(feed_dict = {X: X_test, Y: Y_test})
        
        # compute choice probability curves
        delta_cost = 0.01
        delta_ivt = 0.01
        
        drive_cost_idx = 19
        drive_ivt_idx = 21
        
        if Train:
            # actually not driving in Train dataset
            drive_cost_idx = 0
            drive_ivt_idx = 1
        
        N_cost = np.int((np.max(X_test[:,drive_cost_idx]) - np.min(X_test[:,drive_cost_idx]))/delta_cost) + 1
        N_ivt = np.int((np.max(X_test[:,drive_ivt_idx]) - np.min(X_test[:,drive_ivt_idx]))/delta_ivt) + 1
        data_cost = np.zeros((N_cost, D))
        data_ivt = np.zeros((N_ivt, D))
        data_cost[:, drive_cost_idx] = np.arange(np.min(X_test[:,drive_cost_idx]), np.max(X_test[:,drive_cost_idx]), 0.01)
        data_ivt[:, drive_ivt_idx] = np.arange(np.min(X_test[:,drive_ivt_idx]), np.max(X_test[:,drive_ivt_idx]), 0.01)

        # info for cost column
        util_matrix_cost = output.eval(feed_dict = {X: data_cost}) 
        prob_cost = np.exp(util_matrix_cost)/np.exp(util_matrix_cost).sum(1)[:,np.newaxis]
        util_matrix_ivt = output.eval(feed_dict = {X: data_ivt})
        prob_ivt = np.exp(util_matrix_ivt)/np.exp(util_matrix_ivt).sum(1)[:,np.newaxis]

    return train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt


def full_dnn_elasticity(X_train, Y_train, X_validation, Y_validation, X_test, Y_test,
                   M, n_hidden, l1_const, l2_const, dropout_rate, batch_normalization, learning_rate, n_iterations,
                   n_mini_batch, index_for_var_elas, df_sp_test_nonstand, k_fold,top_n,var_list_index, K=5,
                   Train=False):
    # repeat the standard_hidden_layer to construct one DNN architecture.

    ### build DNN models here
    tf.reset_default_graph()
    N, D = X_train.shape
    # default

    X = tf.placeholder(dtype=tf.float32, shape=(None, D), name='X')
    Y = tf.placeholder(dtype=tf.int64, shape=(None), name='Y')

    hidden = X

    for i in range(M):
        name = 'hidden' + str(i)
        hidden = standard_hidden_layer(hidden, n_hidden, l1_const, dropout_rate, batch_normalization, name)
    output = tf.layers.dense(hidden, K, name='output')

    # add l2 regularization here
    l2_regularization = tf.contrib.layers.l2_regularizer(scale=l2_const, scope=None)
    vars_ = tf.trainable_variables()
    weights = [var_ for var_ in vars_ if 'kernel' in var_.name]
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularization, weights)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=Y), name='cost')
        cost += tf.losses.get_regularization_loss()
        cost += regularization_penalty

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(output, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()

    ### estimate DNN models here

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # always run this to train the model
        init.run()
        for i in range(n_iterations):
            if i % 500 == 0:
                print("Epoch", i, "Cost = ", cost.eval(feed_dict={X: X_train, Y: Y_train}))

            # gradient descent
            X_batch, Y_batch = obtain_mini_batch(X_train, Y_train, n_mini_batch)
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})

        elast_records = {'K-fold': [k_fold]}
        model_name = 'full_dnn_top_' + str(top_n)
        elast_records['Model_name']=[model_name]
        delta_increase = 0.001
        util_matrix_old = output.eval(feed_dict={X: X_test})
        prob_old = np.exp(util_matrix_old) / np.exp(util_matrix_old).sum(1)[:, np.newaxis]
        for idx in index_for_var_elas:
            data_increase = np.copy(X_test)
            data_increase[:,idx] += delta_increase
            util_matrix_new = output.eval(feed_dict={X: data_increase})
            prob_new = np.exp(util_matrix_new) / np.exp(util_matrix_new).sum(1)[:, np.newaxis]
            for mode in range(K):
                var_name = var_list_index[idx + 1]
                elasticity_individual = (prob_new[:,mode] - prob_old[:,mode]) / prob_old[:,mode] / delta_increase * df_sp_test_nonstand.loc[:, var_name] / df_sp_test_nonstand.loc[:, var_name].std()
                elasticity = np.mean(elasticity_individual)
                elast_records[str(mode) + '___' + var_name] = [elasticity]

    return pd.DataFrame(elast_records)
#
##### training here.
#df_sp_train = pd.read_csv('../data/data_AV_Singapore_v1_sp_train.csv')
#df_sp_validation = pd.read_csv('../data/data_AV_Singapore_v1_sp_validation.csv')
#df_sp_test = pd.read_csv('../data/data_AV_Singapore_v1_sp_test.csv')
#
#X_train = df_sp_train.iloc[:, 1:].values
#Y_train = df_sp_train.iloc[:, 0].values
#X_validation = df_sp_validation.iloc[:, 1:].values 
#Y_validation = df_sp_validation.iloc[:, 0].values
#X_test = df_sp_test.iloc[:, 1:].values 
#Y_test = df_sp_test.iloc[:, 0].values
#
## specify hyperparameteers
#M = 6
#n_hidden = 120
#l1_const = 1e-10
#l2_const = 1e-5
#dropout_rate = 1e-50
#batch_normalization = True 
#learning_rate = 0.001 
#n_iterations = 2000 
#n_mini_batch = 100
#train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt = \
#                dnn_estimation(X_train, Y_train, X_validation, Y_validation, X_test, Y_test,
#                               M, n_hidden, l1_const, l2_const, dropout_rate, batch_normalization, learning_rate, n_iterations, n_mini_batch)
#print("Training accuracy is ", train_accuracy)
#print("Validation accuracy is ", validation_accuracy)
#print("Testing accuracy is ", test_accuracy)
#plt.plot(prob_cost[:, 3])
#plt.plot(prob_ivt[:, 3])
#

########################################################
########################################################
########################################################
########################################################
########################################################
### build the DNN with alt specific utilities
def standard_combine_x_z_layer(input_x, input_z, n_hidden_after, l1_const, dropout_rate, batch_normalization, name):
    # standard layer, repeated in the following for loop.
    regularizer = tf.contrib.layers.l1_regularizer(scale=l1_const)
    input_ = tf.concat([input_x, input_z], axis = 1)
    hidden = tf.layers.dense(input_, n_hidden_after, activation = tf.nn.relu, name = name, kernel_regularizer = regularizer)
    if batch_normalization == True:
        hidden = tf.layers.batch_normalization(inputs = hidden, axis = 1)
    hidden = tf.layers.dropout(inputs = hidden, rate = dropout_rate)
    return hidden

def obtain_mini_batch_dnn_alt_specific(X0,X1,X2,X3,X4,Y,Z, n_mini_batch):
    '''
    Return mini_batch
    assume that the row numbers of all input df are the same
    '''
    N, D = X0.shape                     
    index = np.random.choice(N, size = n_mini_batch)     
    X0_batch = X0[index, :]
    X1_batch = X1[index, :]
    X2_batch = X2[index, :]
    X3_batch = X3[index, :]
    X4_batch = X4[index, :]
    Z_batch = Z[index, :]
    Y_batch = Y[index]
    return X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch

def generate_numerical_x_delta(X, delta, x_col_numbers):
    '''
    Generate delta X along x1 and x2. 
    x: a list of strings as X's names, or a list of column numbers. 
    Output:
        X_delta_1
        X_delta_2    
    '''
    x1_col_num,x2_col_num = x_col_numbers    
#    assert X.columns.values[0] == 'x1' and X.columns.values[1] == 'x2', "Input does not have the right names"
    if type(X) == pd.DataFrame:                        
        x1 = X.iloc[:,x1_col_num]
        x2 = X.iloc[:,x2_col_num]
        
        x1_delta = x1 + delta
        x2_delta = x2 + delta

        X_delta_1 = copy.copy(X); X_delta_1.iloc[:, x1_col_num] = x1_delta  
        X_delta_2 = copy.copy(X); X_delta_2.iloc[:, x2_col_num] = x2_delta                             

    if type(X) == np.ndarray:
        x1 = X[:, x1_col_num]
        x2 = X[:, x2_col_num]
        
        x1_delta = x1 + delta
        x2_delta = x2 + delta

        X_delta_1 = copy.copy(X); X_delta_1[:, x1_col_num] = x1_delta
        X_delta_2 = copy.copy(X); X_delta_2[:, x2_col_num] = x2_delta

    return X_delta_1, X_delta_2     
 
## start
def dnn_alt_spec_elasticity(X0_train,X1_train,X2_train,X3_train,X4_train,Y_train,Z_train,
                            X0_validation,X1_validation,X2_validation,X3_validation,X4_validation,Y_validation,Z_validation,
                            X0_test,X1_test,X2_test,X3_test,X4_test,Y_test,Z_test,
                            M_before,M_after,n_hidden_before,n_hidden_after,l1_const,l2_const,
                            dropout_rate,batch_normalization,learning_rate,n_iterations,n_mini_batch,
                            all_elas_var, df_sp_test_nonstand,k_fold,top_n):
    '''
    This function specifies DNN with alternative specific utility
    It performs estimation and prediction
    '''  
    tf.reset_default_graph() 
    N, D0 = X0_train.shape
    N, D1 = X1_train.shape
    N, D2 = X2_train.shape
    N, D3 = X3_train.shape
    N, D4 = X4_train.shape
    N, DZ = Z_train.shape
    
    #K = 5 # default
    
    X0 = tf.placeholder(dtype = tf.float32, shape = (None, D0), name = 'X0')
    X1 = tf.placeholder(dtype = tf.float32, shape = (None, D1), name = 'X1')
    X2 = tf.placeholder(dtype = tf.float32, shape = (None, D2), name = 'X2')
    X3 = tf.placeholder(dtype = tf.float32, shape = (None, D3), name = 'X3')
    X4 = tf.placeholder(dtype = tf.float32, shape = (None, D4), name = 'X4')
    Z = tf.placeholder(dtype = tf.float32, shape = (None, DZ), name = 'Z')
    Y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Y')
    
    hidden_x0 = X0
    hidden_x1 = X1
    hidden_x2 = X2
    hidden_x3 = X3
    hidden_x4 = X4
    hidden_z = Z
    
    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z
    
    ######################## start to build models ##########################
    ### prior to combine Z and X
    # x
    for j in range(5):
        layer_name = 'x'+str(j)
        hidden_j = hidden_dic[layer_name]    
        for i in range(M_before):
            name = 'hidden_before_'+ layer_name + '_'+ str(i)
            hidden_j = standard_hidden_layer(hidden_j, n_hidden_before, l1_const, dropout_rate, batch_normalization, name)
        hidden_dic[layer_name] = hidden_j    
    # z
    for i in range(M_before):
        name = 'hidden_before_'+ 'z' + '_'+ str(i)
        hidden_z = standard_hidden_layer(hidden_z, n_hidden_before, l1_const, dropout_rate, batch_normalization, name)
    hidden_dic['z'] = hidden_z
    ### combine Z and X
    for j in range(5):
        layer_name = 'x'+str(j)
        hidden_j = hidden_dic[layer_name]
        hidden_z = hidden_dic['z']
        name = 'hidden_mix_'+ layer_name + '_'+ str(j)
        hidden_j = standard_combine_x_z_layer(hidden_j, hidden_z, n_hidden_after, l1_const, dropout_rate, batch_normalization, name)    
        hidden_dic[layer_name] = hidden_j
    ### after combining Z and X
    for j in range(5):
        layer_name = 'x'+str(j)
        hidden_j = hidden_dic[layer_name]    
        for i in range(M_after):
            name = 'hidden_after_'+ layer_name + '_'+ str(i)
            hidden_j = standard_hidden_layer(hidden_j, n_hidden_after, l1_const, dropout_rate, batch_normalization, name)
        hidden_dic[layer_name] = hidden_j
    # for the final output...note that last layer has no regularization. Should I still use regularization here???
    for j in range(5):
        layer_name = 'x'+str(j)
        hidden_j = hidden_dic[layer_name]
        regularizer = tf.contrib.layers.l1_regularizer(scale=l1_const)
        output_j = tf.layers.dense(hidden_j, 1, name = 'output'+layer_name, kernel_regularizer = regularizer)
        hidden_dic[layer_name] = output_j
    output = tf.concat([hidden_dic['x0'], hidden_dic['x1'], hidden_dic['x2'], hidden_dic['x3'], hidden_dic['x4']], axis = 1, name = 'output')
    
    # add l2 regularization here
    l2_regularization = tf.contrib.layers.l2_regularizer(scale = l2_const, scope=None)
    vars_ = tf.trainable_variables()
    weights = [var_ for var_ in vars_ if 'kernel' in var_.name]
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularization, weights)
    
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output, labels = Y), name = 'cost')
        cost += tf.losses.get_regularization_loss()
        cost += regularization_penalty
        
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(output, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective
    init = tf.global_variables_initializer()
    
    ######################## start to train models ##########################

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # always run this to train the model
        init.run()
        for i in range(n_iterations):
            if i % 500 == 0:
                print("Epoch", i, "Cost = ", cost.eval(feed_dict = {X0: X0_train, X1: X1_train, X2: X2_train, X3: X3_train, X4:X4_train,
                                                                    Y: Y_train, Z: Z_train}))
            # gradient descent
            X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = \
                                        obtain_mini_batch_dnn_alt_specific(X0_train,X1_train,X2_train,X3_train,X4_train,Y_train,Z_train,n_mini_batch)
            sess.run(training_op, feed_dict = {X0: X0_batch, X1: X1_batch, X2: X2_batch, X3: X3_batch, X4:X4_batch,
                                               Y: Y_batch, Z: Z_batch})
        ### compute prediction accuracy


        elast_records = {'K-fold': [k_fold]}
        model_name = 'sparse_dnn_top_' + str(top_n)
        elast_records['Model_name']=[model_name]
        delta_increase = 0.001
        util_matrix_old = output.eval(feed_dict = {X0: X0_test, X1: X1_test, X2: X2_test, X3: X3_test, X4:X4_test,
                                                      Y: Y_test, Z: Z_test})
        prob_old = np.exp(util_matrix_old) / np.exp(util_matrix_old).sum(1)[:, np.newaxis]

        K = 5
        for key in all_elas_var:
            for i in range(len(all_elas_var[key])):
                var_name = all_elas_var[key][i]
                if key == 'x0_vars':
                    data_increase = np.copy(X0_test)
                    data_increase[:,i] += delta_increase
                    util_matrix_new = output.eval(feed_dict={X0: data_increase, X1: X1_test, X2: X2_test, X3: X3_test, X4: X4_test,
                                           Y: Y_test, Z: Z_test})
                elif key == 'x1_vars':
                    data_increase = np.copy(X1_test)
                    data_increase[:,i] += delta_increase
                    util_matrix_new = output.eval(feed_dict={X0: X0_test, X1: data_increase, X2: X2_test, X3: X3_test, X4: X4_test,
                                           Y: Y_test, Z: Z_test})
                elif key == 'x2_vars':
                    data_increase = np.copy(X2_test)
                    data_increase[:,i] += delta_increase
                    util_matrix_new = output.eval(feed_dict={X0: X0_test, X1: X1_test, X2: data_increase, X3: X3_test, X4: X4_test,
                                           Y: Y_test, Z: Z_test})
                elif key == 'x3_vars':
                    data_increase = np.copy(X3_test)
                    data_increase[:,i] += delta_increase
                    util_matrix_new = output.eval(feed_dict={X0: X0_test, X1: X1_test, X2: X2_test, X3: data_increase, X4: X4_test,
                                           Y: Y_test, Z: Z_test})
                elif key == 'x4_vars':
                    data_increase = np.copy(X4_test)
                    data_increase[:,i] += delta_increase
                    util_matrix_new = output.eval(feed_dict={X0: X0_test, X1: X1_test, X2: X2_test, X3: X3_test, X4: data_increase,
                                           Y: Y_test, Z: Z_test})


                prob_new = np.exp(util_matrix_new) / np.exp(util_matrix_new).sum(1)[:, np.newaxis]
                for mode in range(K):
                    elasticity_individual = (prob_new[:,mode] - prob_old[:,mode]) / prob_old[:,mode] / delta_increase * df_sp_test_nonstand.loc[:, var_name] / df_sp_test_nonstand.loc[:, var_name].std()
                    elasticity = np.mean(elasticity_individual)
                    elast_records[str(mode) + '___' + var_name] = [elasticity]

    return pd.DataFrame(elast_records)


def dnn_alt_spec_estimation(X0_train, X1_train, X2_train, X3_train, X4_train, Y_train, Z_train,
                            X0_validation, X1_validation, X2_validation, X3_validation, X4_validation, Y_validation,
                            Z_validation,
                            X0_test, X1_test, X2_test, X3_test, X4_test, Y_test, Z_test,
                            M_before, M_after, n_hidden_before, n_hidden_after, l1_const, l2_const,
                            dropout_rate, batch_normalization, learning_rate, n_iterations, n_mini_batch):
    '''
    This function specifies DNN with alternative specific utility
    It performs estimation and prediction
    '''
    tf.reset_default_graph()
    N, D0 = X0_train.shape
    N, D1 = X1_train.shape
    N, D2 = X2_train.shape
    N, D3 = X3_train.shape
    N, D4 = X4_train.shape
    N, DZ = Z_train.shape

    # K = 5 # default

    X0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='X0')
    X1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='X1')
    X2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='X2')
    X3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='X3')
    X4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='X4')
    Z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='Z')
    Y = tf.placeholder(dtype=tf.int64, shape=(None), name='Y')

    hidden_x0 = X0
    hidden_x1 = X1
    hidden_x2 = X2
    hidden_x3 = X3
    hidden_x4 = X4
    hidden_z = Z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    ######################## start to build models ##########################
    ### prior to combine Z and X
    # x
    for j in range(5):
        layer_name = 'x' + str(j)
        hidden_j = hidden_dic[layer_name]
        for i in range(M_before):
            name = 'hidden_before_' + layer_name + '_' + str(i)
            hidden_j = standard_hidden_layer(hidden_j, n_hidden_before, l1_const, dropout_rate, batch_normalization,
                                             name)
        hidden_dic[layer_name] = hidden_j
        # z
    for i in range(M_before):
        name = 'hidden_before_' + 'z' + '_' + str(i)
        hidden_z = standard_hidden_layer(hidden_z, n_hidden_before, l1_const, dropout_rate, batch_normalization, name)
    hidden_dic['z'] = hidden_z
    ### combine Z and X
    for j in range(5):
        layer_name = 'x' + str(j)
        hidden_j = hidden_dic[layer_name]
        hidden_z = hidden_dic['z']
        name = 'hidden_mix_' + layer_name + '_' + str(j)
        hidden_j = standard_combine_x_z_layer(hidden_j, hidden_z, n_hidden_after, l1_const, dropout_rate,
                                              batch_normalization, name)
        hidden_dic[layer_name] = hidden_j
    ### after combining Z and X
    for j in range(5):
        layer_name = 'x' + str(j)
        hidden_j = hidden_dic[layer_name]
        for i in range(M_after):
            name = 'hidden_after_' + layer_name + '_' + str(i)
            hidden_j = standard_hidden_layer(hidden_j, n_hidden_after, l1_const, dropout_rate, batch_normalization,
                                             name)
        hidden_dic[layer_name] = hidden_j
    # for the final output...note that last layer has no regularization. Should I still use regularization here???
    for j in range(5):
        layer_name = 'x' + str(j)
        hidden_j = hidden_dic[layer_name]
        regularizer = tf.contrib.layers.l1_regularizer(scale=l1_const)
        output_j = tf.layers.dense(hidden_j, 1, name='output' + layer_name, kernel_regularizer=regularizer)
        hidden_dic[layer_name] = output_j
    output = tf.concat([hidden_dic['x0'], hidden_dic['x1'], hidden_dic['x2'], hidden_dic['x3'], hidden_dic['x4']],
                       axis=1, name='output')

    # add l2 regularization here
    l2_regularization = tf.contrib.layers.l2_regularizer(scale=l2_const, scope=None)
    vars_ = tf.trainable_variables()
    weights = [var_ for var_ in vars_ if 'kernel' in var_.name]
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularization, weights)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=Y), name='cost')
        cost += tf.losses.get_regularization_loss()
        cost += regularization_penalty

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(output, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()

    ######################## start to train models ##########################
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        for i in range(n_iterations):
            if i % 500 == 0:
                print("Epoch", i, "Cost = ",
                      cost.eval(feed_dict={X0: X0_train, X1: X1_train, X2: X2_train, X3: X3_train, X4: X4_train,
                                           Y: Y_train, Z: Z_train}))
            # gradient descent
            X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = \
                obtain_mini_batch_dnn_alt_specific(X0_train, X1_train, X2_train, X3_train, X4_train, Y_train, Z_train,
                                                   n_mini_batch)
            sess.run(training_op, feed_dict={X0: X0_batch, X1: X1_batch, X2: X2_batch, X3: X3_batch, X4: X4_batch,
                                             Y: Y_batch, Z: Z_batch})
        ### compute prediction accuracy
        train_accuracy = accuracy.eval(feed_dict={X0: X0_train, X1: X1_train, X2: X2_train, X3: X3_train, X4: X4_train,
                                                  Y: Y_train, Z: Z_train})
        validation_accuracy = accuracy.eval(
            feed_dict={X0: X0_validation, X1: X1_validation, X2: X2_validation, X3: X3_validation, X4: X4_validation,
                       Y: Y_validation, Z: Z_validation})
        test_accuracy = accuracy.eval(feed_dict={X0: X0_test, X1: X1_test, X2: X2_test, X3: X3_test, X4: X4_test,
                                                 Y: Y_test, Z: Z_test})

        ### compute probability curves by simulated data
        delta_cost = 0.01
        delta_ivt = 0.01
        drive_cost_idx = 0
        drive_ivt_idx = 2
        # only use X3_test because it is about driving cost and ivt.
        N_cost = np.int((np.max(X3_test[:, drive_cost_idx]) - np.min(X3_test[:, drive_cost_idx])) / delta_cost) + 1
        N_ivt = np.int((np.max(X3_test[:, drive_ivt_idx]) - np.min(X3_test[:, drive_ivt_idx])) / delta_ivt) + 1
        X3_data_cost = np.zeros((N_cost, D3))
        X3_data_ivt = np.zeros((N_ivt, D3))
        X3_data_cost[:, drive_cost_idx] = np.arange(np.min(X3_test[:, drive_cost_idx]),
                                                    np.max(X3_test[:, drive_cost_idx]), 0.01)
        X3_data_ivt[:, drive_ivt_idx] = np.arange(np.min(X3_test[:, drive_ivt_idx]), np.max(X3_test[:, drive_ivt_idx]),
                                                  0.01)

        X0_data_cost = np.zeros((N_cost, D0))
        X0_data_ivt = np.zeros((N_ivt, D0))
        X1_data_cost = np.zeros((N_cost, D1))
        X1_data_ivt = np.zeros((N_ivt, D1))
        X2_data_cost = np.zeros((N_cost, D2))
        X2_data_ivt = np.zeros((N_ivt, D2))
        # X3_data_cost = np.zeros((N_cost, D3)); X3_data_ivt = np.zeros((N_ivt, D3))
        X4_data_cost = np.zeros((N_cost, D4))
        X4_data_ivt = np.zeros((N_ivt, D4))
        Z_data_cost = np.zeros((N_cost, DZ))
        Z_data_ivt = np.zeros((N_ivt, DZ))
        Y_data_cost = np.zeros(N_cost)
        Y_data_ivt = np.zeros(N_ivt)
        # compute util and prob curves
        util_matrix_cost = output.eval(
            {X0: X0_data_cost, X1: X1_data_cost, X2: X2_data_cost, X3: X3_data_cost, X4: X4_data_cost,
             Y: Y_data_cost, Z: Z_data_cost})
        prob_cost = np.exp(util_matrix_cost) / np.exp(util_matrix_cost).sum(1)[:, np.newaxis]
        util_matrix_ivt = output.eval(
            {X0: X0_data_ivt, X1: X1_data_ivt, X2: X2_data_ivt, X3: X3_data_ivt, X4: X4_data_ivt,
             Y: Y_data_ivt, Z: Z_data_ivt})
        prob_ivt = np.exp(util_matrix_ivt) / np.exp(util_matrix_ivt).sum(1)[:, np.newaxis]
    return train_accuracy, validation_accuracy, test_accuracy, prob_cost, prob_ivt


### build functions for mlogit Train datasets
def obtain_mini_batch_dnn_alt_specific_train(X0,X1,Y,n_mini_batch):
    '''
    Return mini_batch
    assume that the row numbers of all input df are the same
    '''
    N, D = X0.shape                     
    index = np.random.choice(N, size = n_mini_batch)     
    X0_batch = X0[index, :]
    X1_batch = X1[index, :]
    Y_batch = Y[index]
    return X0_batch, X1_batch, Y_batch


def dnn_alt_spec_estimation_train(X0_train,X1_train,Y_train,
                                  X0_validation,X1_validation,Y_validation,
                                  X0_test,X1_test,Y_test,
                                  M,n_hidden,l1_const,l2_const,
                                  dropout_rate,batch_normalization,learning_rate,n_iterations,n_mini_batch,
                                  K=2):
    '''
    This function specifies DNN with alternative specific utility
    It performs estimation and prediction
    '''
    tf.reset_default_graph()
    N, D0 = X0_train.shape
    N, D1 = X1_train.shape
    
    #K = 5 # default
    
    X0 = tf.placeholder(dtype = tf.float32, shape = (None, D0), name = 'X0')
    X1 = tf.placeholder(dtype = tf.float32, shape = (None, D1), name = 'X1')
    Y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Y')
    
    hidden_x0 = X0
    hidden_x1 = X1
    
    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    
    ######################## start to build models ##########################
    ### prior to combine Z and X
    # x
    for j in range(K):
        layer_name = 'x'+str(j)
        hidden_j = hidden_dic[layer_name]    
        for i in range(M):
            name = 'hidden_'+ layer_name + '_'+ str(i)
            hidden_j = standard_hidden_layer(hidden_j, n_hidden, l1_const, dropout_rate, batch_normalization, name)
        hidden_dic[layer_name] = hidden_j    
    # for the final output...note that last layer has no regularization. Should I still use regularization here???
    for j in range(K):
        layer_name = 'x'+str(j)
        hidden_j = hidden_dic[layer_name]
        regularizer = tf.contrib.layers.l1_regularizer(scale=l1_const)
        output_j = tf.layers.dense(hidden_j, 1, name = 'output'+layer_name, kernel_regularizer = regularizer)
        hidden_dic[layer_name] = output_j
    output = tf.concat([hidden_dic['x0'], hidden_dic['x1']], axis = 1, name = 'output')
    
    # add l2 regularization here
    l2_regularization = tf.contrib.layers.l2_regularizer(scale = l2_const, scope=None)
    vars_ = tf.trainable_variables()
    weights = [var_ for var_ in vars_ if 'kernel' in var_.name]
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularization, weights)
    
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output, labels = Y), name = 'cost')
        cost += tf.losses.get_regularization_loss()
        cost += regularization_penalty
        
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(output, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective
    init = tf.global_variables_initializer()
    
    ######################## start to train models ##########################    
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        for i in range(n_iterations):
            if i % 500 == 0:
                print("Epoch", i, "Cost = ", cost.eval(feed_dict = {X0: X0_train, X1: X1_train,Y: Y_train}))
            # gradient descent
            X0_batch, X1_batch, Y_batch = \
                                        obtain_mini_batch_dnn_alt_specific_train(X0_train,X1_train,Y_train,n_mini_batch)
            sess.run(training_op, feed_dict = {X0: X0_batch, X1: X1_batch, Y: Y_batch})
        ### compute prediction accuracy
        train_accuracy = accuracy.eval(feed_dict = {X0: X0_train, X1: X1_train, Y: Y_train})
        validation_accuracy = accuracy.eval(feed_dict = {X0: X0_validation, X1: X1_validation, Y: Y_validation})
        test_accuracy = accuracy.eval(feed_dict = {X0: X0_test, X1: X1_test,Y: Y_test})
        
        ### compute probability curves by simulated data
        delta_cost = 0.01
        delta_ivt = 0.01
        train_0_cost_idx = 0
        train_0_ivt_idx = 1
        # only use X3_test because it is about driving cost and ivt.
        N_cost = np.int((np.max(X0_test[:,train_0_cost_idx]) - np.min(X0_test[:,train_0_cost_idx]))/delta_cost) + 1
        N_ivt = np.int((np.max(X0_test[:,train_0_ivt_idx]) - np.min(X0_test[:,train_0_ivt_idx]))/delta_ivt) + 1

        X0_data_cost = np.zeros((N_cost, D0))
        X0_data_ivt = np.zeros((N_ivt, D0))
        X0_data_cost[:, train_0_cost_idx] = np.arange(np.min(X0_test[:,train_0_cost_idx]), np.max(X0_test[:,train_0_cost_idx]), 0.01)
        X0_data_ivt[:, train_0_ivt_idx] = np.arange(np.min(X0_test[:,train_0_ivt_idx]), np.max(X0_test[:,train_0_ivt_idx]), 0.01)
              
        X1_data_cost = np.zeros((N_cost, D1))
        X1_data_ivt = np.zeros((N_ivt, D1))
        Y_data_cost = np.zeros(N_cost); Y_data_ivt = np.zeros(N_ivt)
        # compute util and prob curves
        util_matrix_cost = output.eval({X0: X0_data_cost, X1: X1_data_cost, Y: Y_data_cost})
        prob_cost = np.exp(util_matrix_cost)/np.exp(util_matrix_cost).sum(1)[:,np.newaxis]
        util_matrix_ivt = output.eval({X0: X0_data_ivt, X1: X1_data_ivt, Y: Y_data_ivt})
        prob_ivt = np.exp(util_matrix_ivt)/np.exp(util_matrix_ivt).sum(1)[:,np.newaxis]
    return train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt






#### import data
## note: [0,1,2,3,4] meaning [walk,bus,ridesharing,drive,av]
#df_sp_train = pd.read_csv('../data/data_AV_Singapore_v1_sp_train.csv')
#df_sp_validation = pd.read_csv('../data/data_AV_Singapore_v1_sp_validation.csv')
#df_sp_test = pd.read_csv('../data/data_AV_Singapore_v1_sp_test.csv')
#
#y_vars = ['choice']
#z_vars = ['male', 'young_age', 'old_age', 'low_edu', 'high_edu',
#          'low_inc', 'high_inc', 'full_job', 'age', 'inc', 'edu']
#x0_vars = ['walk_walktime']
#x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
#x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
#x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
#x4_vars = ['av_cost', 'av_waittime', 'av_ivt']
#
#X0_train = df_sp_train[x0_vars].values
#X1_train = df_sp_train[x1_vars].values
#X2_train = df_sp_train[x2_vars].values
#X3_train = df_sp_train[x3_vars].values
#X4_train = df_sp_train[x4_vars].values
#Y_train = df_sp_train[y_vars].values.reshape(-1)
#Z_train = df_sp_train[z_vars].values
#
#X0_validation = df_sp_validation[x0_vars].values
#X1_validation = df_sp_validation[x1_vars].values
#X2_validation = df_sp_validation[x2_vars].values
#X3_validation = df_sp_validation[x3_vars].values
#X4_validation = df_sp_validation[x4_vars].values
#Y_validation = df_sp_validation[y_vars].values.reshape(-1)
#Z_validation = df_sp_validation[z_vars].values
#
#X0_test = df_sp_test[x0_vars].values
#X1_test = df_sp_test[x1_vars].values
#X2_test = df_sp_test[x2_vars].values
#X3_test = df_sp_test[x3_vars].values
#X4_test = df_sp_test[x4_vars].values
#Y_test = df_sp_test[y_vars].values.reshape(-1)
#Z_test = df_sp_test[z_vars].values
#
## some hyperparameters
#M_before = 5
#M_after = 5
#n_hidden_before = 40
#n_hidden_after = 40
#l1_const = 1e-10
#l2_const = 1e-10
#dropout_rate = 1e-50
#batch_normalization = True 
#learning_rate = 0.0001 
#n_iterations = 5000 
#n_mini_batch = 100
#
## one estimation here
#train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt = \
#            dnn_alt_spec_estimation(X0_train,X1_train,X2_train,X3_train,X4_train,Y_train,Z_train,
#                                    X0_validation,X1_validation,X2_validation,X3_validation,X4_validation,Y_validation,Z_validation,
#                                    X0_test,X1_test,X2_test,X3_test,X4_test,Y_test,Z_test,
#                                    M_before,M_after,n_hidden_before,n_hidden_after,l1_const,l2_const,
#                                    dropout_rate,batch_normalization,learning_rate,n_iterations,n_mini_batch)
#
#print("Training accuracy is ", train_accuracy)
#print("Validation accuracy is ", validation_accuracy)
#print("Testing accuracy is ", test_accuracy)
#plt.plot(prob_cost[:, 3])
#plt.plot(prob_ivt[:, 3])



























