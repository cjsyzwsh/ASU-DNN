# export datasets
library(mlogit,AER)

setwd('/Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/9_ml_dnn_alt_spe_util/code')

# import datasets
# For DNN work, all datasets need to be transformed to individual-based datasets. (From long to wide)
data("Beef")
data("Car")
data("Catsup")
data("Cracker")
data("Electricity")
data("Fishing")
data("Game")
data("Game2")
data("HC")
data("Heating")
data("Ketchup")
data("MobilePhones")
data("Mode")
data("ModeCanada")
data("Telephone")
data("TollRoad")
data("Train")
data("Tuna")

# export 16 files
Beef_long = Beef
Car_wide = Car
Catsup_wide = Catsup
Cracker_wide = Cracker
Electricity_wide = Electricity
Fishing_wide = Fishing
HC_wide = HC
Heating_wide = Heating
Ketchup_wide = Ketchup
MobilePhones_long = MobilePhones
Mode_wide = Mode
ModeCanada_long = ModeCanada
Telephone_long = Telephone
TollRoad_wide = TollRoad
Train_wide = Train
Tuna_wide = Tuna

# save 16 files
write.csv(Beef_long, '../data/mlogit_Beef_long.csv')
write.csv(Car_wide, '../data/mlogit_Car_wide.csv')
write.csv(Catsup_wide, '../data/mlogit_Catsup_wide.csv')
write.csv(Cracker_wide, '../data/mlogit_Cracker_wide.csv')
write.csv(Electricity_wide, '../data/mlogit_Electricity_wide.csv')
write.csv(Fishing_wide, '../data/mlogit_Fishing_wide.csv')
write.csv(HC_wide, '../data/mlogit_HC_wide.csv')
write.csv(Heating_wide, '../data/mlogit_Heating_wide.csv')
write.csv(Ketchup_wide, '../data/mlogit_Ketchup_wide.csv')
write.csv(MobilePhones_long, '../data/mlogit_MobilePhones_long.csv')
write.csv(Mode_wide, '../data/mlogit_Mode_wide.csv')
write.csv(ModeCanada_long, '../data/mlogit_ModeCanada_long.csv')
write.csv(Telephone_long, '../data/mlogit_Telephone_long.csv')
write.csv(TollRoad_wide, '../data/mlogit_TollRoad_wide.csv')
write.csv(Train_wide, '../data/mlogit_Train_wide.csv')
write.csv(Tuna_wide, '../data/mlogit_Tuna_wide.csv')


















