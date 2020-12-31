import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score


def jho(feat, label, opts):
    ho = 0.3   # ratio of testing set
    
    if 'ho' in opts:
        ho = opts['ho']
    
    # number of instances
    num_data = np.size(feat, 0)
    label    = label.reshape(num_data)  # Solve bug
    
    # prepare data
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=ho) 
    # train model
    mdl     = LinearRegression()
    mdl.fit(xtrain, ytrain)
    
    # prediction
    ypred   = mdl.predict(xtest)
    
    # mean square error
    mse     = np.mean((ytest - ypred) ** 2)
    # r2 score
    r2      = r2_score(ytest, ypred)
    print("Mean Square Error (LR_HO):", mse)
    print("R2 Score (LR_HO):", r2)
    
    lr = {'mse': mse, 'r2': r2, 'xtest': xtest, 'ytest': ytest, 'ypred': ypred}
    
    return lr
    

def jkfold(feat, label, opts):
    kfold = 10   # number of k in kfold
    
    if 'kfold' in opts:
        kfold = opts['kfold']
    
    # number of instances
    num_data = np.size(feat, 0)
    # define selected features
    x_data   = feat
    y_data   = label.reshape(num_data)  # Solve bug
    
    fold     = KFold(n_splits=kfold)
    fold.get_n_splits(x_data, y_data)
    
    ytest2 = []
    ypred2 = []
    t      = 0
    for train_idx, test_idx in fold.split(x_data, y_data):
        xtrain  = x_data[train_idx,:] 
        ytrain  = y_data[train_idx]
        xtest   = x_data[test_idx,:]
        ytest   = y_data[test_idx]
        # train model
        mdl     = LinearRegression()
        mdl.fit(xtrain, ytrain)
        # prediction
        ypred   = mdl.predict(xtest)
        
        ytest2  = np.concatenate((ytest2, ytest), axis=0)
        ypred2  = np.concatenate((ypred2, ypred), axis=0)
        
        if t == 0:
            xtest2  = xtest
        else:
            xtest2  = np.concatenate((xtest2, xtest), axis=0)
        
        t += 1

    # mean square error
    mse  = np.mean((ytest2 - ypred2) ** 2)
    # r2 score
    r2   = r2_score(ytest2, ypred2)
        
    print("Mean Square Error (LR_K-fold):", mse)
    print("R2 Score (LR_K-fold):", r2)
    
    lr = {'mse': mse, 'r2': r2, 'xtest': xtest2, 'ytest': ytest2, 'ypred': ypred2}
    
    return lr


def jloo(feat, label, opts):
    
    # number of instances
    num_data = np.size(feat, 0)
    # define selected features
    x_data   = feat
    y_data   = label.reshape(num_data)  # Solve bug
 
    loo      = LeaveOneOut()
    loo.get_n_splits(x_data)
    
    ytest2 = []
    ypred2 = []
    t      = 0
    for train_idx, test_idx in loo.split(x_data):
        xtrain = x_data[train_idx,:] 
        ytrain = y_data[train_idx]
        xtest  = x_data[test_idx,:]
        ytest  = y_data[test_idx]
        # train model
        mdl     = LinearRegression()
        mdl.fit(xtrain, ytrain)
        # prediction
        ypred   = mdl.predict(xtest)
        
        ytest2  = np.concatenate((ytest2, ytest), axis=0)
        ypred2  = np.concatenate((ypred2, ypred), axis=0)
        
        if t == 0:
            xtest2  = xtest
        else:
            xtest2  = np.concatenate((xtest2, xtest), axis=0)
        
        t += 1
        
    # mean square error
    mse  = np.mean((ytest2 - ypred2) ** 2)
    # r2 score
    r2   = r2_score(ytest2, ypred2)
        
    print("Mean Square Error (LR_LOO):", mse)
    print("R2 Score (LR_LOO):", r2)
    
    lr = {'mse': mse, 'r2': r2, 'xtest': xtest2, 'ytest': ytest2, 'ypred': ypred2}
    
    return lr

