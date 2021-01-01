# Jx-MLT : A Machine Learning Toolbox for Regression

---
> "Toward Talent Scientist: Sharing and Learning Together"
>  --- [Jingwei Too](https://jingweitoo.wordpress.com/)
---

![Wheel](https://github.com/JingweiToo/Machine-Learning-Regression-Toolbox/blob/main/Capture.JPG)


## Introduction
* This toolbox contains 7 widely used machine learning algorithms   
* The < Demo_LR > provide the examples of how to use these methods on benchmark dataset 


## Usage
You may switch the algorithm by changing the 'lr' in 'from MLR.lr import jkfold' to [other abbreviations](/README.md#list-of-available-machine-learning-methods)   
* If you wish to use linear regression ( LR ) then you may write
```code 
from MLR.lr import jkfold 
```

* If you want to use decision tree ( DT ) then you may write
```code 
from MLR.dt import jkfold  
```


## Input
* *feat*    : feature vector matrix ( Instance *x* Features )
* *label*   : label matrix ( Instance *x* 1 )
* *opts*    : parameter settings
  + *ho*    : ratio of testing data in hold-out validation
  + *kfold* : number of folds in *k*-fold cross-validation


## Output
* *mdl* : Machine learning model ( It contains several results )  
  + *mse* : mean square error 
  + *r2*  : R square score


## How to choose the validation scheme?
There are three types of performance validations. These validation strategies are listed as following ( *LR* is adopted as an example ). 
  + Hold-out cross-validation
```code 
from MLR.lr import jho
```
  + *K*-fold cross-validation
```code 
from MLR.lr import jkfold
```
  + Leave-one-out cross-validation
```code 
from MLR.lr import jloo
```


### Example 1 : Linear Regression ( LR ) with *k*-fold cross-validation
```code 
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
```


### Example 2 : LASSO Regression with hold-out validation
```code 
import numpy as np
# change this to switch algorithm & types of validation (jho, jkfold, jloo)
from MLR.lasso import jho 
import matplotlib.pyplot as plt
from sklearn import datasets


# load data
X, Y  = datasets.load_diabetes(return_X_y=True)
feat  = X[:, np.newaxis, 2]
label = Y

# parameters
ho    = 0.3    # ratio of testing data
alpha = 1
opts  = {'alpha':alpha, 'ho':ho}
# LR 
mdl   = jho(feat, label, opts) 

# overall mse
mse = mdl['mse']

# overall r2 score
r2  = mdl['r2']
```



### Example 3 : Decision Tree ( DT ) with leave-one-out validation
```code 
import numpy as np
# change this to switch algorithm & types of validation (jho, jkfold, jloo)
from MLR.dt import jloo 
import matplotlib.pyplot as plt
from sklearn import datasets


# load data
X, Y  = datasets.load_diabetes(return_X_y=True)
feat  = X[:, np.newaxis, 2]
label = Y

# parameters
maxDepth = 5      # maximum depth of tree
opts  = {'maxDepth':maxDepth}
# DT 
mdl   = jloo(feat, label, opts) 

# overall mse
mse = mdl['mse']

# overall r2 score
r2  = mdl['r2']
```


## Requirement

* Python 3 
* Numpy
* Pandas
* Scikit-learn
* Matplotlib


## List of available machine learning methods
* Click on the name of algorithm to check the parameters 
* Use the *opts* to set the specific parameters  
* If you do not set extra parameters then the algorithm will use default setting in [here](/Description.md)


| No. | Abbreviation | Name                                                                              | 
|-----|--------------|-----------------------------------------------------------------------------------|
| 07  | en           | [Elastic Net](Description.md#elastic-net-en)                                      |
| 06  | nn           | [Neural Network](Description.md#neural-network-nn)                                |
| 05  | svr          | [Support Vector Regression](Description.md#support-vector-regression-svr)         |
| 04  | ridge        | [Ridge Regression](Description.md#ridge-regression)                               |
| 03  | lasso        | [Lasso Regression](Description.md#lasso-regression)                               |
| 02  | dt           | [Decision Tree](Description.md#decision-tree-dt)                                  | 
| 01  | lr           | Linear Regression                                                                 | 


  
