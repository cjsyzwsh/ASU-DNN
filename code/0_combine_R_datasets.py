"""
Created on Thu Jun 13 08:40:52 2019
# combine all R datasets for choice analysis
@author: shenhao
"""

#cd /Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/9_ml_dnn_alt_spe_util/code
 
import numpy as np
import pandas as pd
import pickle

data_list = ['Beef_long',
             'Car_wide',
             'Catsup_wide',
             'Cracker_wide',
             'Electricity_wide',
             'Fishing_wide',
             'HC_wide',
             'Heating_wide',
             'Ketchup_wide',
             'MobilePhones_long',
             'Mode_wide',
             'ModeCanada_long',
             'Telephone_long',
             'TollRoad_wide',
             'Train_wide',
             'Tuna_wide']

data_dic = {}
for name in data_list:
    data_dic[name]=pd.read_csv("data/mlogit_"+name+".csv", index_col = 0)
#print(data_list)

# save
with open('data/mlogit_choice_data.pickle', 'wb') as data:
    pickle.dump(data_dic, data, protocol=pickle.HIGHEST_PROTOCOL)








