import pandas as pd
# change this to switch algorithm & types of validation (jho, jkfold, jloo)
from MLR.lasso import jloo 


# load data
url       = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = pd.read_csv(url, header=None)
data      = dataframe.values
feat      = data[:, :-1]  
label     = data[:, -1]  

# parameters
alpha = 1
opts  = {'alpha':alpha}
# LR with k-fold
mdl   = jloo(feat, label, opts) 

# overall mse
mse = mdl['mse']

# overall r2 score
r2  = mdl['r2']


