

# read datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# starting time
start_time = time.time()

# %matplotlib inline
df_sp_train = pd.read_csv('data/data_AV_Singapore_v1_sp_train.csv')
df_sp_validation = pd.read_csv('data/data_AV_Singapore_v1_sp_validation.csv')
# here we combine train and validation set to recreate training and validation sets...
df_sp_combined_train = pd.concat([df_sp_train, df_sp_validation], axis=0)
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
    data_shuffled = data  # may not need to shuffle the data...
    #    data_shuffled = data.loc[n_index_shuffle, :]
    # use validation index to split; validation index: 0,1,2,3,4
    validation_set = data_shuffled.iloc[
                     np.int(n_index / 5) * validation_index:np.int(n_index / 5) * (validation_index + 1), :]
    train_set = pd.concat([data_shuffled.iloc[: np.int(n_index / 5) * validation_index, :],
                           data_shuffled.iloc[np.int(n_index / 5) * (validation_index + 1):, :]])
    return train_set, validation_set


## test
# train_set, validation_set = generate_cross_validation_set(df_sp_combined_train, 3)

############################################################
############################################################
### use other models as benchmarks
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.expressions as bioexp
import os
import biogeme.models as biomodels
model_titles = ['NL'] #'NL''MNL',,'NL'

classifiers_accuracy = {}
classifiers_accuracy['training'] = pd.DataFrame()
classifiers_accuracy['validation'] = pd.DataFrame()
classifiers_accuracy['testing'] = pd.DataFrame()

for name in model_titles:
    classifiers_accuracy[name] = {}
    classifiers_accuracy[name]['prob_cost'] = {}
    classifiers_accuracy[name]['prob_ivt'] = {}

#
training_accuracy_table = pd.DataFrame(np.zeros((len(model_titles), 5)), index=model_titles)
validation_accuracy_table = pd.DataFrame(np.zeros((len(model_titles), 5)), index=model_titles)
testing_accuracy_table = pd.DataFrame(np.zeros((len(model_titles), 5)), index=model_titles)


y_vars = ['choice']
z_vars = ['male', 'young_age', 'old_age', 'low_edu', 'high_edu',
          'low_inc', 'high_inc', 'full_job', 'age', 'inc', 'edu']
x0_vars = ['walk_walktime']
x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
x4_vars = ['av_cost', 'av_waittime', 'av_ivt']

modes_list = ['Walk','PT','RH','AV','Drive']
att = {'Walk':x0_vars, 'PT':x1_vars,'RH':x2_vars,'AV':x4_vars,'Drive':x3_vars}
key_choice_index = {'Walk': 0, 'PT': 1, 'RH': 2, 'AV': 4, 'Drive': 3}

def train_MNL(data):
    for mode in modes_list:
        # availability
        data[mode+'_avail'] = 1
    database = db.Database("MNL_SGP", data)
    beta_dic = {}
    variables = {}

    ASC_WALK = bioexp.Beta('B___ASC___Walk',0,None,None,1) #fixed
    ASC_PT = bioexp.Beta('B___ASC___PT',0,None,None,0)
    ASC_RIDEHAIL = bioexp.Beta('B___ASC___RH',0,None,None,0)
    ASC_AV = bioexp.Beta('B___ASC___AV',0,None,None,0)
    ASC_DRIVE = bioexp.Beta('B___ASC___Drive',0,None,None,0)
    for key in att:
        beta_dic[key] = {}
        if key != 'Walk':
            for var in  z_vars:
                if var not in variables:
                    variables[var] = bioexp.Variable(var)
                beta_name = 'B___' + var + '___' + key
                beta_dic[key][beta_name] = bioexp.Beta(beta_name, 0, None, None, 0)
        for var in att[key]:
            if var not in variables:
                variables[var] = bioexp.Variable(var)
            beta_name = 'B___' + var + '___' + key
            beta_dic[key][beta_name] = bioexp.Beta(beta_name, 0, None, None, 0)


    V = {key_choice_index['Walk']:ASC_WALK, key_choice_index['PT']:ASC_PT,
         key_choice_index['RH']:ASC_RIDEHAIL,key_choice_index['AV']:ASC_AV,
         key_choice_index['Drive']:ASC_DRIVE}
    AV = {}

    for key in att:
        AV[key_choice_index[key]] = bioexp.Variable(key+'_avail')
        if key != 'Walk':
            for var in z_vars:
                beta_name = 'B___' + var + '___' + key
                V[key_choice_index[key]] += variables[var] * beta_dic[key][beta_name]
        for var in att[key]:
            beta_name = 'B___' + var + '___' + key
            V[key_choice_index[key]] += variables[var] * beta_dic[key][beta_name]
    CHOICE = bioexp.Variable('choice')
    logprob = bioexp.bioLogLogit(V, AV, CHOICE)
    formulas = {'loglike': logprob}
    biogeme = bio.BIOGEME(database, formulas,numberOfThreads = 4)
    biogeme.modelName = "MNL_SGP"
    results = biogeme.estimate()
    os.remove("MNL_SGP.html")
    os.remove("MNL_SGP.pickle")
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



def train_NL(data):
    for mode in modes_list:
        # availability
        data[mode+'_avail'] = 1
    database = db.Database("NL_SGP", data)
    beta_dic = {}
    variables = {}

    ASC_WALK = bioexp.Beta('B___ASC___Walk',0,None,None,1) #fixed
    ASC_PT = bioexp.Beta('B___ASC___PT',0,None,None,0)
    ASC_RIDEHAIL = bioexp.Beta('B___ASC___RH',0,None,None,0)
    ASC_AV = bioexp.Beta('B___ASC___AV',0,None,None,0)
    ASC_DRIVE = bioexp.Beta('B___ASC___Drive',0,None,None,0)
    for key in att:
        beta_dic[key] = {}
        if key != 'Walk':
            for var in  z_vars:
                if var not in variables:
                    variables[var] = bioexp.Variable(var)
                beta_name = 'B___' + var + '___' + key
                beta_dic[key][beta_name] = bioexp.Beta(beta_name, 0, None, None, 0)
        for var in att[key]:
            if var not in variables:
                variables[var] = bioexp.Variable(var)
            beta_name = 'B___' + var + '___' + key
            beta_dic[key][beta_name] = bioexp.Beta(beta_name, 0, None, None, 0)


    V = {key_choice_index['Walk']:ASC_WALK, key_choice_index['PT']:ASC_PT,
         key_choice_index['RH']:ASC_RIDEHAIL,key_choice_index['AV']:ASC_AV,
         key_choice_index['Drive']:ASC_DRIVE}
    AV = {} # availability

    for key in att:
        AV[key_choice_index[key]] = bioexp.Variable(key+'_avail')
        if key != 'Walk':
            for var in z_vars:
                beta_name = 'B___' + var + '___' + key
                V[key_choice_index[key]] += variables[var] * beta_dic[key][beta_name]
        for var in att[key]:
            beta_name = 'B___' + var + '___' + key
            V[key_choice_index[key]] += variables[var] * beta_dic[key][beta_name]

    CHOICE = bioexp.Variable('choice')

    MU_car = bioexp.Beta('MU_car', 1, None, None, 0)
    MU_pt_walk = bioexp.Beta('MU_pt_walk', 1, None, None, 1)#fixed
    N_car = MU_car, [key_choice_index['RH'],key_choice_index['AV'],key_choice_index['Drive']]
    N_pt_walk = MU_pt_walk, [key_choice_index['Walk'], key_choice_index['PT']]
    nests = N_car, N_pt_walk
    logprob = biomodels.lognested(V, AV, nests, CHOICE)
    formulas = {'loglike': logprob}
    biogeme = bio.BIOGEME(database, formulas,numberOfThreads = 4)
    biogeme.modelName = "NL_SGP"
    results = biogeme.estimate()
    os.remove("NL_SGP.html")
    os.remove("NL_SGP.pickle")
    # Print the estimated values
    betas = results.getBetaValues()
    # beta={}
    # for k, v in betas.items():
    #     beta[k] = v
    biogeme_file={'V':V, 'av':AV, 'nests':nests,'database':database}
    return betas, biogeme_file

def predict_NL_2(betas, biogeme_file, data):
    for mode in modes_list:
        # availability
        data[mode+'_avail'] = 1
    database = db.Database("NL_SGP", data)
    # The choice model is a nested logit

    prob_Walk = biomodels.nested(biogeme_file['V'], biogeme_file['av'], biogeme_file['nests'], key_choice_index['Walk'])
    prob_PT = biomodels.nested(biogeme_file['V'], biogeme_file['av'], biogeme_file['nests'], key_choice_index['PT'])
    prob_RH = biomodels.nested(biogeme_file['V'], biogeme_file['av'], biogeme_file['nests'], key_choice_index['RH'])
    prob_AV = biomodels.nested(biogeme_file['V'], biogeme_file['av'], biogeme_file['nests'], key_choice_index['AV'])
    prob_Drive = biomodels.nested(biogeme_file['V'], biogeme_file['av'], biogeme_file['nests'], key_choice_index['Drive'])

    simulate = {'prob_Walk': prob_Walk,
                'prob_PT': prob_PT,
                'prob_RH': prob_RH,
                'prob_AV':prob_AV,
                'prob_Drive':prob_Drive}

    biogeme = bio.BIOGEME(database, simulate)


    # Extract the values that are necessary
    betaValues = betas

    # simulatedValues is a Panda dataframe with the same number of rows as
    # the database, and as many columns as formulas to simulate.
    simulatedValues = biogeme.simulate(betaValues)

    prob_list = list(simulatedValues.columns)
    data_test = data
    for key in prob_list:
        data_test[key] = 0
    data_test.loc[:,prob_list] = simulatedValues.loc[:, prob_list]
    data_test['max_prob'] = data_test[prob_list].max(axis=1)
    data_test['CHOOSE'] = 0
    for mode in key_choice_index:
        col_nameprob = 'prob_' + mode
        data_test.loc[data_test[col_nameprob]==data_test['max_prob'],'CHOOSE'] = key_choice_index[mode]

    acc = len(data_test.loc[data_test['CHOOSE']==data_test['choice']])/len(data_test)

    return acc, data_test

def predict_NL(data_test,betas):
    for mode in modes_list:
        col_name = 'exp_U_' + mode
        data_test[col_name] = 0
    MU_est = {'MU_car':betas['MU_car'],'MU_pt_walk': 1}
    for k in betas:
        if 'MU' in k:
            continue
        v = betas[k]
        mode = k.split('___')[2]
        col_name = 'exp_U_' + mode
        if 'ASC' in k:
            data_test[col_name] += 1 * v
        else:
            var_name = k.split('___')[1]
            data_test[col_name] += data_test[var_name] * v

    data_test['exp_U_car'] = 0
    data_test['exp_U_pt_walk'] = 0

    mode_car = ['RH','AV','Drive']
    mode_pt_walk = ['Walk', 'PT']
    for mode in mode_car:
        col_name = 'exp_U_' + mode
        data_test[col_name] = np.exp(MU_est['MU_car'] * data_test[col_name])
        data_test['exp_U_car'] += data_test[col_name]
    for mode in mode_pt_walk:
        col_name = 'exp_U_' + mode
        data_test[col_name] = np.exp(MU_est['MU_pt_walk'] * data_test[col_name])
        data_test['exp_U_pt_walk'] += data_test[col_name]

    prob_motor_list = []
    data_test['log_sum_exp_car'] = 0
    for mode in mode_car:
        col_nameprob = 'cond_prob_' + mode
        prob_motor_list.append(col_nameprob)
        col_name = 'exp_U_' + mode
        data_test[col_nameprob] = data_test[col_name] / data_test['exp_U_car']
        data_test['log_sum_exp_car'] += data_test[col_name]
    data_test['log_sum_exp_car'] += np.exp((1/MU_est['MU_car'])*np.log(data_test['log_sum_exp_car']))



    prob_unmotor_list = []
    data_test['log_sum_exp_pt_walk'] = 0
    for mode in mode_pt_walk:
        col_nameprob = 'cond_prob_' + mode
        prob_unmotor_list.append(col_nameprob)
        col_name = 'exp_U_' + mode
        data_test[col_nameprob] = data_test[col_name] / data_test['exp_U_pt_walk']
        data_test['log_sum_exp_pt_walk'] += data_test[col_name]
    data_test['log_sum_exp_pt_walk'] += np.exp((1 / MU_est['MU_pt_walk']) * np.log(data_test['log_sum_exp_pt_walk']))

    data_test['prob_car'] = data_test['log_sum_exp_car'] / (data_test['log_sum_exp_car'] + data_test['log_sum_exp_pt_walk'])
    data_test['prob_pt_walk'] = data_test['log_sum_exp_pt_walk'] / (
                data_test['log_sum_exp_car'] + data_test['log_sum_exp_pt_walk'])

    prob_list = []
    for mode in mode_car:
        col_cond_nameprob = 'cond_prob_' + mode
        col_nameprob = 'prob_' + mode
        prob_list.append(col_nameprob)
        data_test[col_nameprob] = data_test[col_cond_nameprob] * data_test['prob_car']
    for mode in mode_pt_walk:
        col_cond_nameprob = 'cond_prob_' + mode
        col_nameprob = 'prob_' + mode
        prob_list.append(col_nameprob)
        data_test[col_nameprob] = data_test[col_cond_nameprob] * data_test['prob_pt_walk']



    data_test['max_prob'] = data_test[prob_list].max(axis=1)
    data_test['CHOOSE'] = 0
    for mode in key_choice_index:
        col_nameprob = 'prob_' + mode
        data_test.loc[data_test[col_nameprob]==data_test['max_prob'],'CHOOSE'] = key_choice_index[mode]

    acc = len(data_test.loc[data_test['CHOOSE']==data_test['choice']])/len(data_test)

    return acc, data_test

def calculate_prob_cost_ivt(data, model_name, beta):
    ### compute probability curves by simulated data
    delta_cost = 0.01
    delta_ivt = 0.01
    N_cost = np.int((np.max(data.loc[:, 'drive_cost']) - np.min(data.loc[:, 'drive_cost'])) / delta_cost) + 1
    N_ivt = np.int((np.max(data.loc[:, 'drive_ivt']) - np.min(data.loc[:, 'drive_ivt'])) / delta_ivt) + 1
    data_cost = pd.DataFrame(np.zeros((N_cost, len(data.columns))),columns = data.columns)
    data_ivt = pd.DataFrame(np.zeros((N_ivt, len(data.columns))),columns = data.columns)
    data_cost.loc[:, 'drive_cost'] = np.arange(np.min(data.loc[:, 'drive_cost']), np.max(data.loc[:, 'drive_cost']),
                                                delta_cost)
    data_ivt.loc[:, 'drive_ivt'] = np.arange(np.min(data.loc[:, 'drive_ivt']), np.max(data.loc[:, 'drive_ivt']),
                                              delta_ivt)
    # set other to zero
    for key in att:
        if (key != 'drive_cost') & (key != 'drive_ivt'):
            data_cost.loc[:, key] = 0
            data_ivt.loc[:, key] = 0
        else:
            if key == 'drive_cost':
                data_ivt.loc[:, key] = 0
            else:
                data_cost.loc[:, key] = 0
    if model_name == 'MNL':
        _, data_cost = predict_MNL(data_cost, beta)
        _, data_ivt = predict_MNL(data_ivt, beta)
        col_list = ['prob_Walk','prob_PT','prob_RH','prob_Drive','prob_AV'] # sequence is important 0 - 4
        prob_cost = np.array(data_cost.loc[:,col_list])
        prob_ivt = np.array(data_ivt.loc[:, col_list])
    else:
        _, data_cost = predict_NL(data_cost, beta)
        _, data_ivt = predict_NL(data_ivt, beta)
        col_list = ['prob_Walk','prob_PT','prob_RH','prob_Drive','prob_AV'] # sequence is important 0 - 4
        prob_cost = np.array(data_cost.loc[:,col_list])
        prob_ivt = np.array(data_ivt.loc[:, col_list])
    return prob_cost, prob_ivt

var_list_for_elast = ['walk_walktime','bus_cost','bus_ivt','ridesharing_cost','ridesharing_ivt',
            'drive_cost','drive_ivt','av_cost','av_ivt']
def calculate_elasticity(data, model_name, biofiles, beta, j):
    # percent_increase = 0.01 #1%
    delta_increase = 0.001

    elast_dic = {'Walk':['walk_walktime'], 'PT':['bus_cost','bus_ivt'],
                 'RH':['ridesharing_cost','ridesharing_ivt'],
                 'Drive':['drive_cost','drive_ivt'],'AV':['av_cost','av_ivt']}
    elast_records = {'K-fold':[j]}

    for var in var_list_for_elast:
        data_increase = data.copy()
        x_old = df_sp_test_nonstand.loc[:, var].mean()
        # data_increase = df_sp_test_nonstand.copy()
        # data_increase.loc[:,var] = data_increase.loc[:,var]*(1+percent_increase)
        # data_increase.loc[:,:] = StandardScaler().fit_transform(data_increase.loc[:,:])
        #
        data_increase.loc[:, var] = data_increase.loc[:, var] + delta_increase
        if model_name == 'MNL':
            _, data_prob_new = predict_MNL(data_increase, beta)
        if model_name == 'NL':
            _, data_prob_new = predict_NL_2(beta, biofiles, data_increase)
        for mode in elast_dic:
            data['prob_diff'] = data_prob_new.loc[:,'prob_' + mode] - data.loc[:,'prob_' + mode]
            data['elas'] = data['prob_diff']  / delta_increase * df_sp_test_nonstand[var] / data.loc[:,'prob_' + mode] / df_sp_test_nonstand[var].std() #
            elasticity = data['elas'].mean()
            #print(elasticity)
            elast_records[mode + '___' + var + '___' + model_name] = [elasticity]
    return elast_records


elast_records_MNL = {}
elast_records_NL = {}
for j in range(5):
    # five fold training with cross validation
    df_sp_train, df_sp_validation = generate_cross_validation_set(df_sp_combined_train, j)

    ###
    for name in model_titles:
        tic = time.time()
        print("Training model ", name, " ...")
        if name == 'MNL':
            beta = train_MNL(df_sp_train)
            Training_time = round((time.time() - tic), 2)
            print('Training time', Training_time, 'seconds')
            training_accuracy,_ = predict_MNL(df_sp_train, beta)
            validation_accuracy,_ = predict_MNL(df_sp_validation, beta)
            testing_accuracy,df_sp_test_prob = predict_MNL(df_sp_test, beta)
        else:
            beta, biofiles = train_NL(df_sp_train)
            Training_time = round((time.time() - tic), 2)
            print('Training time', Training_time, 'seconds')
            training_accuracy,_ = predict_NL_2(beta, biofiles, df_sp_train)
            validation_accuracy,_ = predict_NL_2(beta, biofiles, df_sp_validation)
            testing_accuracy,df_sp_test_prob = predict_NL_2(beta, biofiles, df_sp_test)


        # compute accuracy

        print("Its training accuracy is:", training_accuracy)
        print("Its validation accuracy is:", validation_accuracy)
        print("Its testing accuracy is:", testing_accuracy)

        training_accuracy_table.loc[name, j] = training_accuracy
        validation_accuracy_table.loc[name, j] = validation_accuracy
        testing_accuracy_table.loc[name, j] = testing_accuracy

        # compute prob cost and prob ivt
        prob_cost, prob_ivt = calculate_prob_cost_ivt(df_sp_test, name, beta)
        classifiers_accuracy[name]['prob_cost']['prob_cost'+str(j)] = prob_cost
        classifiers_accuracy[name]['prob_ivt']['prob_ivt'+str(j)] = prob_ivt

        # compute elasticity

        if name == 'MNL':
            biofiles = {}
            elast_records_temp_MNL = calculate_elasticity(df_sp_test_prob,name,biofiles,beta, j)
            if len(elast_records_MNL) == 0:
                elast_records_MNL = pd.DataFrame(elast_records_temp_MNL)
            else:
                elast_records_MNL = pd.concat([elast_records_MNL, pd.DataFrame(elast_records_temp_MNL)])
        if name == 'NL':
            elast_records_temp_NL = calculate_elasticity(df_sp_test_prob,name,biofiles,beta, j)
            if len(elast_records_NL) == 0:
                elast_records_NL = pd.DataFrame(elast_records_temp_NL)
            else:
                elast_records_NL = pd.concat([elast_records_NL, pd.DataFrame(elast_records_temp_NL)])
        print (' ================== ')

classifiers_accuracy['training']=training_accuracy_table
classifiers_accuracy['validation']=validation_accuracy_table
classifiers_accuracy['testing']=testing_accuracy_table

###
if len(elast_records_MNL) >0:
    elast_records_MNL_save = {'Variables':var_list_for_elast}
    for mode in modes_list:
        elast_records_MNL_save[mode] = [0] * len(var_list_for_elast)
    elast_records_MNL_save = pd.DataFrame(elast_records_MNL_save)
    for col in elast_records_MNL.columns:
        if col != 'K-fold':
            mode = col.split('___')[0]
            var = col.split('___')[1]
            elast_records_MNL_save.loc[elast_records_MNL_save['Variables'] == var ,mode] = elast_records_MNL.loc[:,col].mean()
    elast_records_MNL_save.to_csv('output/elasticity_MNL.csv', index=False,columns = ['Variables','Walk','PT','RH','Drive','AV'])

if len(elast_records_NL) >0:
    elast_records_NL_save = {'Variables':var_list_for_elast}
    for mode in modes_list:
        elast_records_NL_save[mode] = [0] * len(var_list_for_elast)
    elast_records_NL_save = pd.DataFrame(elast_records_NL_save)
    for col in elast_records_NL.columns:
        if col != 'K-fold':
            mode = col.split('___')[0]
            var = col.split('___')[1]
            elast_records_NL_save.loc[elast_records_NL_save['Variables'] == var ,mode] = elast_records_NL.loc[:,col].mean()
    elast_records_NL_save.to_csv('output/elasticity_NL.csv', index=False,columns = ['Variables','Walk','PT','RH','Drive','AV'])

import pickle
with open('output/classifiers_accuracy_MNL_NL.pickle', 'wb') as data:
    pickle.dump(classifiers_accuracy, data, protocol=pickle.HIGHEST_PROTOCOL)















