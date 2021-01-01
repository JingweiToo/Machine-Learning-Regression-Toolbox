# Detail Parameter Settings / Default Setting

## Decision Tree (DT) 
* Maximum depth of tree
```code 
maxDepth = 5
opts = {'maxDepth':maxDepth, 'kfold':kfold}
```

## Elastic Net (EN)
* EN contains several parameters 
```code
alpha  = 1.0      # constant that multiplies the penalty term
ratio  = 0.5      # Elastic Net mixing parameter
opts = {'alpha':alpha, 'ratio':ratio, 'kfold':kfold}
```    

## LASSO Regression 
* Constant that multiplies the L1 term
```code
alpha = 1
opts = {'alpha':alpha, 'kfold':kfold}
```

## Neural Network (NN)
* NN contains several parameters 
```code
hls      = 100      # hidden layer size 
fun      = 'relu'   # activation function ( 'relu' or 'identity' or 'logistic' or 'tanh' )
max_iter = 100      # maximum iterations
opts = {'hls':hls, 'fun':fun, 'max_iter':max_iter, 'kfold':kfold}
```

## RIDGE Regression
* Regularization strength
```code
alpha = 1
opts = {'alpha':alpha, 'kfold':kfold}
```

## Support Vector Regression (SVR)
* Kernel function
```code
kernel = 'rbf'     # kernel ( 'linear' or 'poly' or 'rbf' or 'sigmoid' or 'precomputed' )
opts = {'kernel':kernel, 'kfold':kfold}
```




