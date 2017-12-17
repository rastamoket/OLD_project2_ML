# -*- coding: utf-8 -*-
""" Different regressors """
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from helpers import *

def try_linRegressors(regressions_method, x_tr, x_te, y_tr, y_te):
    ''' To try all the regressors and select the one that gives the smallest RMSE

    :param regressions_method: methods to apply regressions, these methods are from sklearn (linear_model.method)
    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :return: best_reg (the best regressor), best_rmse (the rmse associated with the best regressor)
    '''

    best_rmse = 1000 # Put very large like that it will be update to the RMSE find with one of the method
    #******* select the best regressor ******
    for regressor in regressions_method: # Loop over all the regressors
        reg = regressor() 
        reg.fit(x_tr, y_tr) # Training of the regressor
        pred = x_te.dot(reg.coef_) # Compute the prediction by multiply each column (feature) by the coef find in the training of the reg

        rmse = np.sqrt(mean_squared_error(y_te, pred)) # Compute rmse
        if rmse < best_rmse: # Condition that allow us to choose the best regressors by taking the smallest rmse
            best_rmse = rmse
            best_reg = reg

    return best_reg, best_rmse

def lin_regressors(data, regressions_method, regressor_hyperparam = None, ratio = 0.85):
    ''' To apply every linear regressors on the predictions from all the algorithms

    :param data: predictions from every algorithms
    :param regressions_method: methods to apply regressions, these methods are from sklearn (linear_model.method)
    :param regressor_hyperparam: if there are some, otherwise = None
    :param ratio: for the splitting of training and test set (default = 0.85)
    :return: best_lin_regressors = the best regressors based on RMSE, best_rmse = rmse associated with the best regressor
    '''

    ########## Splitting the data in training and test set #############
    #******** split the data in predictions and label ******
    data_predictions_np, data_label_np = get_label_predictions(data)

    #******** Split in training and test set ********
    x_tr, x_te, y_tr, y_te = split_predictions_data(data_predictions_np, data_label_np, ratio) # call the method to split the predictions

    ########### Try all the linear regressors and select the best one ######
    best_lin_regressor, best_rmse = try_linRegressors(regressions_method, x_tr, x_te, y_tr, y_te)


    return best_lin_regressor, best_rmse






