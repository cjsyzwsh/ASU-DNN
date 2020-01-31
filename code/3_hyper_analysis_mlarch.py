"""
Created on Tue Sep  4 15:40:20 2018

@author: shenhao
"""
#cd /Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/9_ml_dnn_alt_spe_util/code

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures 
import matplotlib.pyplot as plt
import copy



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
full_dnn_df = pd.DataFrame(np.zeros((n_rows, len(columns_))), columns = columns_)
for i in range(n_rows):
    for j in range(len(columns_)):
        full_dnn_df.iloc[i, j] = full_dnn_dic[i][columns_[j]]

# create a dataframe for full dnn (finer search)
columns_ = list(full_dnn_finer_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i] 
n_rows = len(full_dnn_finer_dic.keys())
full_dnn_finer_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns = columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        full_dnn_finer_df.loc[i, columns_remove_prob[j]] = full_dnn_finer_dic[i][columns_remove_prob[j]]

# create a dataframe for sparse dnn
columns_sparse = list(sparse_dnn_dic[1].keys())
columns_sparse.remove('prob_cost')
columns_sparse.remove('prob_ivt')
n_rows = len(sparse_dnn_dic.keys())
sparse_dnn_df = pd.DataFrame(np.zeros((n_rows, len(columns_sparse))), columns = columns_sparse)
for i in range(n_rows):
    for j in range(len(columns_sparse)):
        sparse_dnn_df.iloc[i, j] = sparse_dnn_dic[i][columns_sparse[j]]

# create a dataframe for sparse dnn (finer search)
columns_ = list(sparse_dnn_finer_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i] 
n_rows = len(sparse_dnn_finer_dic.keys())
sparse_dnn_finer_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns = columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        sparse_dnn_finer_df.loc[i, columns_remove_prob[j]] = sparse_dnn_finer_dic[i][columns_remove_prob[j]]
# rename sparse_dnn_finer_df due to some stupid mistake...(add "_" )
part_name = list(sparse_dnn_finer_df.columns[:-15])
part_name.extend([name[:-1]+'_'+name[-1:] for name in sparse_dnn_finer_df.columns[-15:]])
sparse_dnn_finer_df.columns = part_name

# create a dataframe for classifier accuracy
classifier_train = classifier_accuracy_dic['training'] 
classifier_validation = classifier_accuracy_dic['validation'] 
classifier_test = classifier_accuracy_dic['testing'] 
classifier_train=classifier_train.mean(1)
classifier_validation=classifier_validation.mean(1)
classifier_test=classifier_test.mean(1)

# report MNL NL accuracy
classifier_train_MNL_NL = classifiers_accuracy_MNL_NL_dic['training']
classifier_validation_MNL_NL = classifiers_accuracy_MNL_NL_dic['validation']
classifier_test_MNL_NL = classifiers_accuracy_MNL_NL_dic['testing']
classifier_train_MNL_NL=classifier_train_MNL_NL.mean(1)
classifier_validation_MNL_NL=classifier_validation_MNL_NL.mean(1)
classifier_test_MNL_NL=classifier_test_MNL_NL.mean(1)
save_mnl_nl_acc = pd.DataFrame({'Model':['MNL','NL'],'Train_acc':list(classifier_train_MNL_NL.values),
                   'Val_acc':list(classifier_validation_MNL_NL.values),'Test_acc':list(classifier_test_MNL_NL.values)})
save_mnl_nl_acc.to_csv('output/MNL_NL_accuracy.csv',index=False)

# create a dataframe for classifier accuracy (mlogit data)
classifier_train_mlogit = classifier_accuracy_train_dic['training'] 
classifier_validation_mlogit = classifier_accuracy_train_dic['validation'] 
classifier_test_mlogit = classifier_accuracy_train_dic['testing'] 
classifier_train_mlogit=classifier_train_mlogit.mean(1)
classifier_validation_mlogit=classifier_validation_mlogit.mean(1)
classifier_test_mlogit=classifier_test_mlogit.mean(1)

# create a dataframe for sparse dnn (mlogit data)
columns_ = list(sparse_dnn_train_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i]
n_rows = len(sparse_dnn_train_dic.keys())
sparse_dnn_train_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns = columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        sparse_dnn_train_df.loc[i, columns_remove_prob[j]] = sparse_dnn_train_dic[i][columns_remove_prob[j]]
# rename sparse_dnn_finer_df due to some stupid mistake...(add "_" )
part_name_1 = list(sparse_dnn_train_df.columns[:-15])
part_name_1.extend([name[:-1]+'_'+name[-1:]  for name in sparse_dnn_train_df.columns[-15:]])
sparse_dnn_train_df.columns = part_name_1

# create a dataframe for full dnn (mlogit data)
columns_ = list(full_dnn_train_dic[1].keys())
columns_remove_prob = [i for i in columns_ if 'prob_' not in i] 
n_rows = len(full_dnn_train_dic.keys())
full_dnn_train_df = pd.DataFrame(np.zeros((n_rows, len(columns_remove_prob))), columns = columns_remove_prob)
for i in range(n_rows):
    for j in range(len(columns_remove_prob)):
        full_dnn_train_df.loc[i, columns_remove_prob[j]] = full_dnn_train_dic[i][columns_remove_prob[j]]

# compute average training, validation, and testing accuracies for finer HPO
def average_var_(df, var_name):
    #df = full_dnn_finer_df
    #var_name = 'validation_accuracy'
    df['average_'+var_name] = (1/5)*(df[var_name+'_0']+df[var_name+'_1']+df[var_name+'_2']+df[var_name+'_3']+df[var_name+'_4'])
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
sparse_dnn_df["n_hidden"] = (sparse_dnn_df["n_hidden_before"]*6 + sparse_dnn_df["n_hidden_after"]*5)/2.0
sparse_dnn_finer_df["M"] = sparse_dnn_finer_df["M_before"] + sparse_dnn_finer_df["M_after"] + 1
sparse_dnn_finer_df["n_hidden"] = (sparse_dnn_finer_df["n_hidden_before"]*6 + sparse_dnn_finer_df["n_hidden_after"]*5)/2.0
# transform the units of matrices
def transform_unit_df(df1):
    df = copy.copy(df1)
    df[['l1_const', 'l2_const', 'dropout_rate', 'learning_rate']] = \
        -np.log10(df[['l1_const', 'l2_const', 'dropout_rate', 'learning_rate']])           
    df[['n_iteration','n_mini_batch']] = \
         np.log10(df[['n_iteration','n_mini_batch']])
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
            [full_dnn_df_trans_unit[hyper_vars+['train_accuracy', 'validation_accuracy', 'test_accuracy']].values,
            full_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_0', 'validation_accuracy_0', 'test_accuracy_0']].values,
            full_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_1', 'validation_accuracy_1', 'test_accuracy_1']].values,
            full_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_2', 'validation_accuracy_2', 'test_accuracy_2']].values,
            full_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_3', 'validation_accuracy_3', 'test_accuracy_3']].values,
            full_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_4', 'validation_accuracy_4', 'test_accuracy_4']].values,
            sparse_dnn_df_trans_unit[hyper_vars+['train_accuracy', 'validation_accuracy', 'test_accuracy']].values,
            sparse_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_0', 'validation_accuracy_0', 'test_accuracy_0']].values,
            sparse_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_1', 'validation_accuracy_1', 'test_accuracy_1']].values,
            sparse_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_2', 'validation_accuracy_2', 'test_accuracy_2']].values,
            sparse_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_3', 'validation_accuracy_3', 'test_accuracy_3']].values,
            sparse_dnn_finer_df_trans_unit[hyper_vars+['train_accuracy_4', 'validation_accuracy_4', 'test_accuracy_4']].values], 
             axis = 0), columns = hyper_vars + ['train_accuracy', 'validation_accuracy', 'test_accuracy'])
# note the following works because first half data is from the full dnn, the second half from the sparse dnn.
# prepare datasets for R regression...
combined_df.loc[np.int(combined_df.shape[0]/2.0):, 'sparse_dnn'] = 1
combined_df.loc[:np.int(combined_df.shape[0]/2.0), 'sparse_dnn'] = 0
combined_df.to_csv('output/table_combined_prediction.csv', index = False)
                
# identify the best sparse and full models 
best_sparse_dnn_index = np.argmax(sparse_dnn_finer_df['average_validation_accuracy'])
best_full_dnn_index = np.argmax(full_dnn_finer_df['average_validation_accuracy'])
top_ten_sparse_dnn_index = sparse_dnn_finer_df.sort_values('average_validation_accuracy', ascending = False).index[:10]
top_ten_full_dnn_index = full_dnn_finer_df.sort_values('average_validation_accuracy', ascending = False).index[:10]
best_sparse_dnn_train_index = np.argmax(sparse_dnn_train_df['average_validation_accuracy'])
best_full_dnn_train_index = np.argmax(full_dnn_train_df['average_validation_accuracy'])
top_ten_sparse_dnn_train_index = sparse_dnn_train_df.sort_values('average_validation_accuracy', ascending = False).index[:10]
top_ten_full_dnn_train_index = full_dnn_train_df.sort_values('average_validation_accuracy', ascending = False).index[:10]


### analyze 0. 
# total market share of training set
df_sp_train = pd.read_csv('data/data_AV_Singapore_v1_sp_train.csv')
df_sp_validation = pd.read_csv('data/data_AV_Singapore_v1_sp_validation.csv')
df_sp_test = pd.read_csv('data/data_AV_Singapore_v1_sp_test.csv')
choice_all = list(df_sp_train.choice) + list(df_sp_validation.choice) + list(df_sp_test.choice)
# note: [0,1,2,3,4] meaning [walk,bus,ridesharing,drive,av]
print("Sample market share is:", pd.value_counts(choice_all)/len(choice_all))

### analysis 1. 
# plot two curves from HPO search 1
sparse_validation_accuracy = sorted(sparse_dnn_df['validation_accuracy'])
full_validation_accuracy = sorted(full_dnn_df['validation_accuracy'])
# 
fig = plt.figure(figsize = (10, 10))
ax = plt.axes()
ax.plot(sparse_validation_accuracy[50:], label = 'sparse DNN validation', color = 'blue')
ax.plot(full_validation_accuracy[50:], label = 'full DNN validation', color = 'orange')
#ax.set_title("Comparison of Prediction Accuracy in Validation Sets (Search 1)")
ax.set(xlabel = 'Sorted models', ylabel = "Prediction accuracy")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15) 
ax.title.set_fontsize(20)
ax.legend(loc = 2, fontsize = 10, title = None)
plt.savefig("output/graph_validation_pred_comparison_search1.png")
plt.savefig("../paper/graph_validation_pred_comparison_search1.png")
plt.close()

## plot two curves from HPO search 2
def plot_sorted_pred(sparse_df, full_df, var_name, ylim_min, ylim_max, file_name):
    # 
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes()
    ax.plot(sorted(sparse_df[var_name+'_0'])[20: ], color = 'g', alpha = 0.2)
    ax.plot(sorted(sparse_df[var_name+'_1'])[20: ], color = 'g', alpha = 0.2)
    ax.plot(sorted(sparse_df[var_name+'_2'])[20: ], color = 'g', alpha = 0.2)
    ax.plot(sorted(sparse_df[var_name+'_3'])[20: ], color = 'g', alpha = 0.2)
    ax.plot(sorted(sparse_df[var_name+'_4'])[20: ], color = 'g', alpha = 0.2)
    ax.plot(sorted(sparse_df['average_'+var_name])[20: ], label = 'ASU-DNN', color = 'g', linewidth = 4, marker='^',markersize=10)
    ax.plot(sorted(full_df[var_name+'_0'])[20: ], color = 'r', alpha = 0.2)
    ax.plot(sorted(full_df[var_name+'_1'])[20: ], color = 'r', alpha = 0.2)
    ax.plot(sorted(full_df[var_name+'_2'])[20: ], color = 'r', alpha = 0.2)
    ax.plot(sorted(full_df[var_name+'_3'])[20: ], color = 'r', alpha = 0.2)
    ax.plot(sorted(full_df[var_name+'_4'])[20: ], color = 'r', alpha = 0.2)
    ax.plot(sorted(full_df['average_'+ var_name])[20: ], label = 'F-DNN', color = 'r', linewidth = 4,marker='s',markersize=10)
    #ax.set_title("Comparison of Prediction Accuracy in Validation Sets (Search 2)")
    ax.set_ylim([ylim_min, ylim_max])
    ax.set(xlabel = 'Sorted models', ylabel = "Prediction accuracy")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.title.set_fontsize(20)
    ax.legend(loc = 2, fontsize = 25, title = None)
    plt.savefig("output/"+file_name)
#    plt.savefig("../paper/graph_validation_pred_comparison_search2.png")
    plt.close()

# plot 1
sparse_df=sparse_dnn_finer_df
full_df=full_dnn_finer_df
var_name='validation_accuracy'
ylim_min=0.50
ylim_max=0.65
file_name="graph_validation_pred_comparison_search2.png"
plot_sorted_pred(sparse_df, full_df, var_name, ylim_min, ylim_max, file_name)
    
# plot 2
sparse_df=sparse_dnn_finer_df
full_df=full_dnn_finer_df
var_name='test_accuracy'
ylim_min=0.50
ylim_max=0.65
file_name="graph_test_pred_comparison_search2.png"
plot_sorted_pred(sparse_df, full_df, var_name, ylim_min, ylim_max, file_name)    

# plot 3
sparse_df=sparse_dnn_train_df
full_df=full_dnn_train_df
var_name='validation_accuracy'
ylim_min=0.60
ylim_max=0.75
file_name="graph_validation_pred_comparison_train.png"
plot_sorted_pred(sparse_df, full_df, var_name, ylim_min, ylim_max, file_name)    

# plot 3
sparse_df=sparse_dnn_train_df
full_df=full_dnn_train_df
var_name='test_accuracy'
ylim_min=0.60
ylim_max=0.75
file_name="graph_test_pred_comparison_train.png"
plot_sorted_pred(sparse_df, full_df, var_name, ylim_min, ylim_max, file_name)


### analyze the differences of best sparse and full DNNs in terms of testing accuracy and their chocie probabilities.
best_full_dnn_validation_accuracy = np.mean([full_dnn_finer_dic[best_full_dnn_index]['validation_accuracy_'+str(i)] for i in range(4)])
best_sparse_dnn_validation_accuracy = np.mean([sparse_dnn_finer_dic[best_sparse_dnn_index]['validation_accuracy'+str(i)] for i in range(4)])
best_full_dnn_test_accuracy = np.mean([full_dnn_finer_dic[best_full_dnn_index]['test_accuracy_'+str(i)] for i in range(4)])
best_sparse_dnn_test_accuracy = np.mean([sparse_dnn_finer_dic[best_sparse_dnn_index]['test_accuracy'+str(i)] for i in range(4)])
print("Prediction accuracy of the best sparse DNN model in testing set (SGP data) is: ", best_sparse_dnn_test_accuracy)
print("Prediction accuracy of the best full DNN model in testing set (SGP data) is: ", best_full_dnn_test_accuracy)

best_full_dnn_validation_train_accuracy = np.mean([full_dnn_train_dic[best_full_dnn_train_index]['validation_accuracy_'+str(i)] for i in range(4)])
best_sparse_dnn_validation_train_accuracy = np.mean([sparse_dnn_train_dic[best_sparse_dnn_train_index]['validation_accuracy'+str(i)] for i in range(4)])
best_full_dnn_test_train_accuracy = np.mean([full_dnn_train_dic[best_full_dnn_train_index]['test_accuracy_'+str(i)] for i in range(4)])
best_sparse_dnn_test_train_accuracy = np.mean([sparse_dnn_train_dic[best_sparse_dnn_train_index]['test_accuracy'+str(i)] for i in range(4)])
print("Prediction accuracy of the best sparse DNN model in testing set (Train) is: ", best_sparse_dnn_test_train_accuracy)
print("Prediction accuracy of the best full DNN model in testing set (Train) is: ", best_full_dnn_test_train_accuracy)

top_ten_full_dnn_validation_accuracy = 0
top_ten_sparse_dnn_validation_accuracy = 0
for index_ in top_ten_sparse_dnn_index:
    top_ten_sparse_dnn_validation_accuracy += np.mean([sparse_dnn_finer_dic[index_]['validation_accuracy'+str(i)] for i in range(4)])
for index_ in top_ten_full_dnn_index:
    top_ten_full_dnn_validation_accuracy += np.mean([full_dnn_finer_dic[index_]['validation_accuracy_'+str(i)] for i in range(4)])
top_ten_sparse_dnn_validation_accuracy = top_ten_sparse_dnn_validation_accuracy/10
top_ten_full_dnn_validation_accuracy = top_ten_full_dnn_validation_accuracy/10

top_ten_full_dnn_test_accuracy = 0
top_ten_sparse_dnn_test_accuracy = 0
for index_ in top_ten_sparse_dnn_index:
    top_ten_sparse_dnn_test_accuracy += np.mean([sparse_dnn_finer_dic[index_]['test_accuracy'+str(i)] for i in range(4)])
for index_ in top_ten_full_dnn_index:
    top_ten_full_dnn_test_accuracy += np.mean([full_dnn_finer_dic[index_]['test_accuracy_'+str(i)] for i in range(4)])
top_ten_sparse_dnn_test_accuracy = top_ten_sparse_dnn_test_accuracy/10
top_ten_full_dnn_test_accuracy = top_ten_full_dnn_test_accuracy/10
print("Prediction accuracy of the top ten sparse DNN model in testing set (SGP data) is: ", top_ten_sparse_dnn_test_accuracy)
print("Prediction accuracy of the top ten full DNN model in testing set (SGP data) is: ", top_ten_full_dnn_test_accuracy)

top_ten_full_dnn_validation_train_accuracy = 0
top_ten_sparse_dnn_validation_train_accuracy = 0
for index_ in top_ten_sparse_dnn_train_index:
    top_ten_sparse_dnn_validation_train_accuracy += np.mean([sparse_dnn_train_dic[index_]['validation_accuracy'+str(i)] for i in range(4)])
for index_ in top_ten_full_dnn_train_index:
    top_ten_full_dnn_validation_train_accuracy += np.mean([full_dnn_train_dic[index_]['validation_accuracy_'+str(i)] for i in range(4)])
top_ten_sparse_dnn_validation_train_accuracy = top_ten_sparse_dnn_validation_train_accuracy/10
top_ten_full_dnn_validation_train_accuracy = top_ten_full_dnn_validation_train_accuracy/10

top_ten_full_dnn_test_train_accuracy = 0
top_ten_sparse_dnn_test_train_accuracy = 0
for index_ in top_ten_sparse_dnn_train_index:
    top_ten_sparse_dnn_test_train_accuracy += np.mean([sparse_dnn_train_dic[index_]['test_accuracy'+str(i)] for i in range(4)])
for index_ in top_ten_full_dnn_train_index:
    top_ten_full_dnn_test_train_accuracy += np.mean([full_dnn_train_dic[index_]['test_accuracy_'+str(i)] for i in range(4)])
top_ten_sparse_dnn_test_train_accuracy = top_ten_sparse_dnn_test_train_accuracy/10
top_ten_full_dnn_test_train_accuracy = top_ten_full_dnn_test_train_accuracy/10
print("Prediction accuracy of the top ten sparse DNN model in testing set (Train) is: ", top_ten_sparse_dnn_test_train_accuracy)
print("Prediction accuracy of the top ten full DNN model in testing set (Train) is: ", top_ten_full_dnn_test_train_accuracy)

## combine these to report the pred accuracy table.
classifier_nn_validation_sgp = pd.Series([best_full_dnn_validation_accuracy, best_sparse_dnn_validation_accuracy, 
                                          top_ten_full_dnn_validation_accuracy, top_ten_sparse_dnn_validation_accuracy], 
                                  index = ["F-DNN (Top 1)", "ASU-DNN(Top 1)", "F-DNN (Top 10)", "ASU-DNN(Top 10)"])
classifier_nn_test_sgp = pd.Series([best_full_dnn_test_accuracy, best_sparse_dnn_test_accuracy, 
                           top_ten_full_dnn_test_accuracy, top_ten_sparse_dnn_test_accuracy], 
                          index = ["F-DNN (Top 1)", "ASU-DNN(Top 1)", "F-DNN (Top 10)", "ASU-DNN(Top 10)"])
classifier_nn_validation_train = pd.Series([best_full_dnn_validation_train_accuracy, best_sparse_dnn_validation_train_accuracy, 
                           top_ten_full_dnn_validation_train_accuracy, top_ten_sparse_dnn_validation_train_accuracy], 
                          index = ["F-DNN (Top 1)", "ASU-DNN(Top 1)", "F-DNN (Top 10)", "ASU-DNN(Top 10)"])
classifier_nn_test_train = pd.Series([best_full_dnn_test_train_accuracy, best_sparse_dnn_test_train_accuracy, 
                           top_ten_full_dnn_test_train_accuracy, top_ten_sparse_dnn_test_train_accuracy], 
                          index = ["F-DNN (Top 1)", "ASU-DNN(Top 1)", "F-DNN (Top 10)", "ASU-DNN(Top 10)"])

classifier_validation_sgp = pd.concat([classifier_nn_validation_sgp, classifier_validation], axis = 0)
classifier_test_sgp = pd.concat([classifier_nn_test_sgp, classifier_test], axis = 0)
classifier_validation_train = pd.concat([classifier_nn_validation_train, classifier_validation_mlogit], axis = 0)
classifier_test_train = pd.concat([classifier_nn_test_train, classifier_test_mlogit], axis = 0)

classifier_validation_sgp_h = pd.DataFrame(classifier_validation_sgp[:, np.newaxis].T, columns = classifier_validation_sgp.index)
classifier_test_sgp_h = pd.DataFrame(classifier_test_sgp[:, np.newaxis].T, columns = classifier_test_sgp.index)
classifier_validation_train_h = pd.DataFrame(classifier_validation_train[:, np.newaxis].T, columns = classifier_validation_train.index)
classifier_test_train_h = pd.DataFrame(classifier_test_train[:, np.newaxis].T, columns = classifier_test_train.index)
#
classifiers = pd.concat([classifier_validation_sgp_h, classifier_test_sgp_h, 
                         classifier_validation_train_h, classifier_test_train_h], axis = 0)
classifiers_df = pd.DataFrame(classifiers.values, 
                              index = ["Validation (SGP)", "Test (SGP)", "Validation (Train)", "Test (Train)"],
                              columns = classifiers.columns)
np.round_(classifiers_df, decimals = 3).to_csv("output/table_accuracy.csv")

### visualize the difference across hyperdimensions
def plot_scatter_with_c(data, x1, x2, file_title):
    # y should be a scalar of colors by customization
    # x1 and x2 are two variable names of data
    #x1: M
    #x2: accuracy
    # dot size;
    plt.style.use('seaborn-white')
    params = {'figure.figsize': (10, 10),
         'legend.fontsize': 25,
         'axes.labelsize': 25,
         'axes.titlesize': 25,
         'xtick.labelsize': 25,
         'ytick.labelsize': 25}
    plt.rcParams.update(params)

    # control colors
#    color_label = {0:'g', 1:'r'}
#    color_vector = data[y].replace(to_replace = color_label.keys(), value = color_label.values())
#    size_dots = data[y].replace(to_replace = size_label.keys(), value = size_label.values())
#    graph_title =  x1 + " and " + x2
#    file_title = 'hyper_' + y + '.png'
    plt.figure(figsize = (10, 10))
    plt.plot(data.loc[data['connectivity']=='sparse',x1], data.loc[data['connectivity']=='sparse',x2], 'o', color = 'g', label = 'ASU-DNN')
    plt.plot(data.loc[data['connectivity']=='full',x1], data.loc[data['connectivity']=='full',x2], 'o', color = 'r', label = 'F-DNN')

    plt.plot(data.loc[data['connectivity']=='sparse',:].groupby(x1).max()[x2], '--', color = 'g', label = 'Max; ASU-DNN')
    plt.plot(data.loc[data['connectivity']=='full',:].groupby(x1).max()[x2], '--', color = 'r', label = 'Max; F-DNN')

    # plot a linear/quadratic fit?
    x1_sparse_array = data.loc[data['connectivity']=='sparse',x1]
    x2_sparse_array = data.loc[data['connectivity']=='sparse',x2]
    x1_order_array = np.linspace(np.min(data[x1]), np.max(data[x1]), 100)
    fit_sparse = np.polyfit(x1_sparse_array, x2_sparse_array, 2)
    fit_sparse_fn = np.poly1d(fit_sparse)
    plt.plot(x1_order_array, fit_sparse_fn(x1_order_array), '-', color = 'g', label = 'Reg; ASU-DNN')

    # plot a linear/quadratic fit?
    x1_full_array = data.loc[data['connectivity']=='full',x1]
    x2_full_array = data.loc[data['connectivity']=='full',x2]
    x1_order_array = np.linspace(np.min(data[x1]), np.max(data[x1]), 100)
    fit_full = np.polyfit(x1_full_array, x2_full_array, 2)
    fit_full_fn = np.poly1d(fit_full)
    plt.plot(x1_order_array, fit_full_fn(x1_order_array), '-', color = 'r', label = 'Reg; F-DNN')

#    plt.plot(data.loc[data['connectivity']=='full',:].groupby(x1).mean()[x2], '-', color = 'r', label = 'Mean; Connectivity = Full')

    plt.legend(loc = 4)
    plt.xlabel(x1)
    plt.ylabel(x2)
    
    # legend
    # x and y lim
    xlim_delta = 0.1 * (np.max(data[x1]) - np.min(data[x1]))
    xlim_min = np.min(data[x1]) - xlim_delta
    xlim_max = np.max(data[x1]) + xlim_delta
    ylim_delta = 0.1 * (np.max(data[x2]) - np.min(data[x2]))                     
    ylim_min = np.min(data[x2]) - ylim_delta
    ylim_max = np.max(data[x2]) + ylim_delta

    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    
    plt.savefig("output/"+file_title)
    plt.savefig("../paper/"+file_title)
    plt.close()

## prepare data
#%matplotlib inline
useful_vars = ['M', 'n_hidden', 'l1_const', 'l2_const', 'dropout_rate', 'batch_normalization',
               'learning_rate', 'n_iteration', 'n_mini_batch', 'connectivity']
sparse_dnn_vars = ['M_before', 'M_after', 'n_hidden_before','n_hidden_after']
sparse_dnn_finer_df['connectivity'] = 'sparse'
full_dnn_finer_df['connectivity'] = 'full'
# round 1 results
sparse_dnn_df['M'] = sparse_dnn_df['M_before'] + sparse_dnn_df['M_after'] + 1
sparse_dnn_df['n_hidden'] = sparse_dnn_df['n_hidden_before'] + sparse_dnn_df['n_hidden_after'] 
sparse_dnn_df['connectivity'] = 'sparse'
full_dnn_df['connectivity'] = 'full'      
# combine everything
all_dnn_list = []
for i in range(5):
    all_dnn_list.append(sparse_dnn_finer_df[useful_vars + sparse_dnn_vars + ['train_accuracy_'+str(i), 'validation_accuracy_'+str(i),'test_accuracy_'+str(i)]].rename(
            columns = {'train_accuracy_'+str(i): 'train_accuracy', 'validation_accuracy_'+str(i): 'validation_accuracy', 'test_accuracy_'+str(i): 'test_accuracy'}))
    all_dnn_list.append(full_dnn_finer_df[useful_vars + ['train_accuracy_'+str(i), 'validation_accuracy_'+str(i),'test_accuracy_'+str(i)]].rename(
            columns = {'train_accuracy_'+str(i): 'train_accuracy', 'validation_accuracy_'+str(i): 'validation_accuracy', 'test_accuracy_'+str(i): 'test_accuracy'}))
all_dnn_list.append(sparse_dnn_df[useful_vars + sparse_dnn_vars +['train_accuracy', 'validation_accuracy', 'test_accuracy']])
all_dnn_list.append(full_dnn_df[useful_vars+['train_accuracy', 'validation_accuracy', 'test_accuracy']])
all_dnn_df = pd.concat(all_dnn_list, axis = 0)
# edit values in all_dnn_df
all_dnn_df['batch_normalization'] = np.int_(all_dnn_df['batch_normalization'])
all_dnn_df['n_hidden'] = pd.cut(all_dnn_df['n_hidden'], bins = [0, 25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575, 625], 
                                labels = [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
all_dnn_df['n_hidden'] = np.float32(all_dnn_df['n_hidden'])
log_list = ['l1_const', 'l2_const', 'dropout_rate', 'learning_rate']
for log_element in log_list:
    all_dnn_df[log_element] = np.log10(all_dnn_df[log_element])
    all_dnn_df.rename(columns = {log_element: 'log('+log_element+')'}, inplace = True)

## export graphs
x1_list = ['M', 'n_hidden', 'log(l1_const)', 'log(l2_const)', 'log(dropout_rate)', 'batch_normalization',
           'log(learning_rate)', 'n_iteration', 'n_mini_batch']
x1_list_sparse = ['M_before', 'M_after', 'M','n_hidden_before','n_hidden_after', 'n_hidden', 'log(l1_const)', 'log(l2_const)', 'log(dropout_rate)', 'batch_normalization',
           'log(learning_rate)', 'n_iteration', 'n_mini_batch']
x2_list = ['validation_accuracy', 'test_accuracy']
for x1 in x1_list:
    for x2 in x2_list:
        file_title = 'graph_visual_'+x2+'_'+x1+'.png'
        plot_scatter_with_c(all_dnn_df, x1, x2, file_title)


#### analyze the 700 models & export top five for sparse and fully connected DNN 
sparse_dnn_top_five = all_dnn_df.loc[all_dnn_df.connectivity == 'sparse', :].sort_values('validation_accuracy', ascending = False)[['validation_accuracy']+x1_list_sparse].head(5)
full_dnn_top_five = all_dnn_df.loc[all_dnn_df.connectivity == 'full', :].sort_values('validation_accuracy', ascending = False)[['validation_accuracy']+x1_list].head(5)
sparse_dnn_top_five.index = np.arange(sparse_dnn_top_five.shape[0])
full_dnn_top_five.index = np.arange(full_dnn_top_five.shape[0])
np.round_(sparse_dnn_top_five,decimals = 3).T.to_csv('output/table_top_five_sparse_dnn.csv')
np.round_(full_dnn_top_five,decimals = 3).T.to_csv('output/table_top_five_full_dnn.csv')


##### compare their interpretation by visualization
# note: [0,1,2,3,4] meaning [walk,bus,ridesharing,drive,av]
# visualize top ten curves and best curve for 5 validation data sets. (10 * 5)
def plot_driving_prob_curves(data, indices, driving_cost_min, driving_cost_max, sparse = True):
    '''
    data is the initial dictionary
    plot the relationship between driving probabilities and driving costs
    driving_cost_min, driving_cost_max are used to transform units
    '''
    if sparse == True:
        prob_cost_len = data[0]['prob_cost0'].shape[0]
        name = 'prob_cost'
    elif sparse == False:
        prob_cost_len = data[0]['prob_cost_0'].shape[0]
        name = 'prob_cost_'
    
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes()
        ###                
    average_sparse_prob_cost_walk = np.zeros(prob_cost_len)
    average_sparse_prob_cost_bus = np.zeros(prob_cost_len)
    average_sparse_prob_cost_ridesharing = np.zeros(prob_cost_len)
    average_sparse_prob_cost_drive = np.zeros(prob_cost_len)
    average_sparse_prob_cost_av = np.zeros(prob_cost_len)

    for index_ in indices:
        for i in range(5):
            # walk index: 0
            cost_array = data[index_][name + str(i)][:, 0]
            ax.plot(cost_array, color = 'r', alpha = 0.2)
            average_sparse_prob_cost_walk += cost_array
            # bus index: 1
            cost_array = data[index_][name + str(i)][:, 1]
            ax.plot(cost_array, color = 'g', alpha = 0.2)
            average_sparse_prob_cost_bus += cost_array
            # ridesharing index: 2
            cost_array = data[index_][name + str(i)][:, 2]
            ax.plot(cost_array, color = 'c', alpha = 0.2)
            average_sparse_prob_cost_ridesharing += cost_array
            # driving index: 3
            cost_array = data[index_][name + str(i)][:, 3]
            ax.plot(cost_array, color = 'b', alpha = 0.2)
            average_sparse_prob_cost_drive += cost_array                
            # av index: 4
            cost_array = data[index_][name + str(i)][:, 4]
            ax.plot(cost_array, color = 'y', alpha = 0.2)
            average_sparse_prob_cost_av += cost_array
            
    # plot average                
    ax.plot(average_sparse_prob_cost_walk/(len(indices)*5), color = 'r', alpha = 1, linewidth = 3, label = 'Walking',linestyle='dotted')
    ax.plot(average_sparse_prob_cost_bus/(len(indices)*5), color = 'g', alpha = 1, linewidth = 3, label = 'Bus',linestyle=(0,(7,3,1,3)))
    ax.plot(average_sparse_prob_cost_ridesharing/(len(indices)*5), color = 'c', alpha = 1, linewidth = 3, label = 'Ridesharing',linestyle=(0,(5,2,1,2,1,2)))
    ax.plot(average_sparse_prob_cost_drive/(len(indices)*5), color = 'b', alpha = 1, linewidth = 3, label = 'Driving',linestyle='solid')
    ax.plot(average_sparse_prob_cost_av/(len(indices)*5), color = 'y', alpha = 1, linewidth = 3, label = 'AV',linestyle='dashed')
    ax.set_ylim([0, 1.05])

    ax.legend(loc = 1, fontsize = 'xx-large',frameon=True)
    ax.set_xticks(np.linspace(0, prob_cost_len, 7))
    ax.set_xticklabels(np.round_(np.linspace(driving_cost_min, driving_cost_max, 7), decimals = 2))
#        ax.set_title("Choice probabilities of driving (sparse DNN"+ " top "+ str(len(indices))+" )")

#    if sparse == False:
#        average_full_prob_cost = np.zeros(data[0]['prob_cost_0'].shape[0])
#        for index_ in indices:
#            for i in range(5):
#                cost_array = data[index_]['prob_cost_' + str(i)][:, 3]
#                average_full_prob_cost += cost_array
#                ax.plot(cost_array, color = 'r', alpha = 0.2)
#        ax.plot(average_full_prob_cost/(len(indices)*5), color = 'r', alpha = 1, linewidth = 3)
#        ax.set_xticks(np.linspace(0, len(average_full_prob_cost), 7))
#        ax.set_xticklabels(np.round_(np.linspace(driving_cost_min, driving_cost_max, 7), decimals = 2))
##        ax.set_title("Choice probabilities of driving (full DNN"+ " top "+ str(len(indices))+" )")
        
    x_label = 'Driving costs ($)'
    y_label = 'Mode choice probabilities'
    ax.set(xlabel=x_label, ylabel=y_label)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    ax.title.set_fontsize(20)

    font_size = 22
    ax.set_xlabel(x_label, fontsize = font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='y', labelsize = font_size)
    ax.tick_params(axis='x', labelsize = font_size)

#    ax.legend(loc = 2, fontsize = 10, title = None)
    if sparse == True:
        plt.savefig("output/graph_driving_prob_sparse_"+"top"+str(len(indices))+".png")
        plt.savefig("../paper/graph_driving_prob_sparse_"+"top"+str(len(indices))+".png")
    if sparse == False:
        plt.savefig("output/graph_driving_prob_full_"+"top"+str(len(indices))+".png")
        plt.savefig("../paper/graph_driving_prob_full_"+"top"+str(len(indices))+".png")
    plt.close()


def plot_driving_prob_curves_MNL_NL(data, driving_cost_min, driving_cost_max, MNL=True):
    '''
    data is the initial dictionary
    plot the relationship between driving probabilities and driving costs
    driving_cost_min, driving_cost_max are used to transform units
    '''
    if MNL:
        prob_cost_len = data['MNL']['prob_cost']['prob_cost0'].shape[0]
        model_name = 'MNL'
    else:
        prob_cost_len = data['NL']['prob_cost']['prob_cost0'].shape[0]
        model_name = 'NL'

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ###
    average_sparse_prob_cost_walk = np.zeros(prob_cost_len)
    average_sparse_prob_cost_bus = np.zeros(prob_cost_len)
    average_sparse_prob_cost_ridesharing = np.zeros(prob_cost_len)
    average_sparse_prob_cost_drive = np.zeros(prob_cost_len)
    average_sparse_prob_cost_av = np.zeros(prob_cost_len)

    name = 'prob_cost'
    for i in range(5):
        # walk index: 0
        cost_array = data[model_name]['prob_cost'][name + str(i)][:, 0]
        ax.plot(cost_array, color='r', alpha=0.2)
        average_sparse_prob_cost_walk += cost_array
        # bus index: 1
        cost_array = data[model_name]['prob_cost'][name + str(i)][:, 1]
        ax.plot(cost_array, color='g', alpha=0.2)
        average_sparse_prob_cost_bus += cost_array
        # ridesharing index: 2
        cost_array = data[model_name]['prob_cost'][name + str(i)][:, 2]
        ax.plot(cost_array, color='c', alpha=0.2)
        average_sparse_prob_cost_ridesharing += cost_array
        # driving index: 3
        cost_array = data[model_name]['prob_cost'][name + str(i)][:, 3]
        ax.plot(cost_array, color='b', alpha=0.2)
        average_sparse_prob_cost_drive += cost_array
        # av index: 4
        cost_array = data[model_name]['prob_cost'][name + str(i)][:, 4]
        ax.plot(cost_array, color='y', alpha=0.2)
        average_sparse_prob_cost_av += cost_array

    # plot average
    ax.plot(average_sparse_prob_cost_walk / 5, color='r', alpha=1, linewidth=3, label='Walking',linestyle='dotted')
    ax.plot(average_sparse_prob_cost_bus / 5, color='g', alpha=1, linewidth=3, label='Bus',linestyle=(0,(7,3,1,3)))
    ax.plot(average_sparse_prob_cost_ridesharing / 5, color='c', alpha=1, linewidth=3,
            label='Ridesharing',linestyle=(0,(5,2,1,2,1,2)))
    ax.plot(average_sparse_prob_cost_drive / 5, color='b', alpha=1, linewidth=3, label='Driving',linestyle='solid')
    ax.plot(average_sparse_prob_cost_av / 5, color='y', alpha=1, linewidth=3, label='AV',linestyle='dashed')
    ax.set_ylim([0, 1.05])
    ax.legend(loc=1, fontsize='xx-large',frameon=True)

    ax.set_xticks(np.linspace(0, prob_cost_len, 7))
    ax.set_xticklabels(np.round_(np.linspace(driving_cost_min, driving_cost_max, 7), decimals=2))
    #        ax.set_title("Choice probabilities of driving (sparse DNN"+ " top "+ str(len(indices))+" )")

    #    if sparse == False:
    #        average_full_prob_cost = np.zeros(data[0]['prob_cost_0'].shape[0])
    #        for index_ in indices:
    #            for i in range(5):
    #                cost_array = data[index_]['prob_cost_' + str(i)][:, 3]
    #                average_full_prob_cost += cost_array
    #                ax.plot(cost_array, color = 'r', alpha = 0.2)
    #        ax.plot(average_full_prob_cost/(len(indices)*5), color = 'r', alpha = 1, linewidth = 3)
    #        ax.set_xticks(np.linspace(0, len(average_full_prob_cost), 7))
    #        ax.set_xticklabels(np.round_(np.linspace(driving_cost_min, driving_cost_max, 7), decimals = 2))
    ##        ax.set_title("Choice probabilities of driving (full DNN"+ " top "+ str(len(indices))+" )")
    x_label = 'Driving costs ($)'
    y_label = 'Mode choice probabilities'
    ax.set(xlabel=x_label, ylabel=y_label)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    ax.title.set_fontsize(20)

    font_size = 22
    ax.set_xlabel(x_label, fontsize = font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='y', labelsize = font_size)
    ax.tick_params(axis='x', labelsize = font_size)
    #    ax.legend(loc = 2, fontsize = 10, title = None)
    if MNL:
        plt.savefig("output/graph_driving_prob_MNL.png")
        plt.savefig("../paper/graph_driving_prob_MNL.png")
    else:
        plt.savefig("output/graph_driving_prob_NL.png")
        plt.savefig("../paper/graph_driving_prob_NL.png")
    plt.close()


# get min and max
driving_cost_min = np.min(sp_full_nonstand_df.drive_cost)
driving_cost_max = np.max(sp_full_nonstand_df.drive_cost)
# sparse dnn
plot_driving_prob_curves(sparse_dnn_finer_dic, list(top_ten_sparse_dnn_index), driving_cost_min, driving_cost_max)
plot_driving_prob_curves(sparse_dnn_finer_dic, [best_sparse_dnn_index], driving_cost_min, driving_cost_max)
# full dnn
plot_driving_prob_curves(full_dnn_finer_dic, list(top_ten_full_dnn_index), driving_cost_min, driving_cost_max, sparse = False)
plot_driving_prob_curves(full_dnn_finer_dic, [best_full_dnn_index], driving_cost_min, driving_cost_max, sparse = False)
# MNL_NL
plot_driving_prob_curves_MNL_NL(classifiers_accuracy_MNL_NL_dic, driving_cost_min, driving_cost_max, MNL = True)
plot_driving_prob_curves_MNL_NL(classifiers_accuracy_MNL_NL_dic, driving_cost_min, driving_cost_max, MNL = False)

#######
# 600 * 10
n_fully_connected_params = 24*600 + 600*600*9 + 600*5
# 100 * 10
n_sparse_connected_params = 4*100*6 + 100*100*6*5 + 100*100*5*5
print("Number of parameters in fully connected DNN is: ", n_fully_connected_params)
print("Number of parameters in sparsely connected DNN is: ", n_sparse_connected_params)















## not sure the following is useful...
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################


# need to do a quadratic transformation with L1 norm regularization
def quadratic_transform_df(X, interaction = True):
    '''
    sklearn could only transform matrix. I need dataframe transformation with new col names
    X should be a dataframe
    '''
    col_names = X.columns
    n_rows = X.shape[0]
    if interaction == True:
        quadratic_column_n = np.int(1 + 2*len(col_names) + len(col_names)*(len(col_names) - 1)/2)
    else:
        quadratic_column_n = np.int(1 + 2*len(col_names))
    
    # initialize df        
    quadratic_x = pd.DataFrame(np.zeros((n_rows, quadratic_column_n)))
    # constant terms
    quadratic_x.iloc[:, 0] = 1.0
    # linear terms                 
    quadratic_x.iloc[:, 1:len(col_names)+1] = X.values
    # quadratic terms
    quadratic_x.iloc[:, len(col_names)+1:2*len(col_names)+1] = np.square(X.values)

    if interaction == True:
        # interaction terms
        interaction_names = ['a'] * np.int(len(col_names)*(len(col_names) - 1)/2)
        count = 0
        for i in range(len(col_names)):
            for j in range(i+1, len(col_names)):
                quadratic_x.iloc[:, 2*len(col_names) + 1 + count] = X.loc[:,col_names[i]]*X.loc[:,col_names[j]]
                name = col_names[i] + '*' + col_names[j]
                interaction_names[count] = name
                count += 1
    # names
    quadratic_column_names = ['const']
    quadratic_column_names.extend(col_names)
    quadratic_column_names.extend([s+'_square' for s in col_names])

    if interaction == True:
        quadratic_column_names.extend(interaction_names)
    # 
    quadratic_x.columns = quadratic_column_names
    return quadratic_x

## test
#X = full_dnn_df[['l1_const', 'l2_const', 'dropout_rate', 'learning_rate']]
#quadratic_x = quadratic_transform_df(X)

#### 1. linear regression without interaction terms
# import statsmodels.api as sm
# x_full_dnn_df = full_dnn_df.iloc[:, :-3]
# y = full_dnn_df.iloc[:, -2] # validation accuracy
# quadratic_x_full_dnn_df = quadratic_transform_df(x_full_dnn_df, interaction=False)
# model = sm.OLS(y, quadratic_x_full_dnn_df)
# results = model.fit()
# results.summary()
#
# # what to do?
# # what to do?
#
# #### 2. LASSO regression with interaction terms
# # You could obtain the quadratic optimization with LASSO solution here...or include the Lasso as one step in hyper gradient.
# from sklearn.linear_model import Lasso
# x_full_dnn_df_train = full_dnn_df.iloc[:80, :-3]
# x_full_dnn_df_test = full_dnn_df.iloc[80:, :-3]
# quadratic_x_full_dnn_df_train = quadratic_transform_df(x_full_dnn_df_train)
# x_full_dnn_df_test.index = np.arange(x_full_dnn_df_test.shape[0])
# quadratic_x_full_dnn_df_test= quadratic_transform_df(x_full_dnn_df_test)
# y_train = full_dnn_df.iloc[:80, -2] # validation accuracy
# y_test = full_dnn_df.iloc[80:, -2] # validation accuracy
# # reindex test datasets
# y_test.index = np.arange(y_test.shape[0])
# # start lasso
# alpha_list = [1e-10, 1e-5, 1e-2, 0.1, 0.2, 0.5, 1.0]
# coefficient_table = pd.DataFrame(np.zeros((len(quadratic_x_full_dnn_df_train.columns), len(alpha_list))),
#                                  index = quadratic_x_full_dnn_df_train.columns, columns = alpha_list)
# score_list_train = np.zeros(len(alpha_list))
# score_list_test = np.zeros(len(alpha_list))
# for alpha in alpha_list:
#     model_l1 = Lasso(alpha = alpha)
#     model_l1.fit(quadratic_x_full_dnn_df_train, y_train)
#     coefficient_table.loc[:, alpha] = model_l1.coef_
#     score_list_train[alpha_list.index(alpha)] = model_l1.score(quadratic_x_full_dnn_df_train, y_train)
#     score_list_test[alpha_list.index(alpha)] = model_l1.score(quadratic_x_full_dnn_df_test, y_test)














