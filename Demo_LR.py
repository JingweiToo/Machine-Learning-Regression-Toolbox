import numpy as np
# change this to switch algorithm & types of validation (jho, jkfold, jloo)
from MLR.lr import jkfold 
import matplotlib.pyplot as plt
from sklearn import datasets


# load data
X, Y  = datasets.load_diabetes(return_X_y=True)
feat  = X[:, np.newaxis, 2]
label = Y

# parameters
kfold = 10
opts  = {'kfold':kfold}
# LR with k-fold
mdl   = jkfold(feat, label, opts) 

# overall mse
mse = mdl['mse']

# overall r2 score
r2  = mdl['r2']


# Plot outputs
xtest = mdl['xtest']
ytest = mdl['ytest']
ypred = mdl['ypred']

plt.scatter(xtest, ytest,  color='black')
plt.plot(xtest, ypred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
