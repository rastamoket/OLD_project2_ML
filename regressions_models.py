# -*- coding: utf-8 -*-
""" Different regressors """
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from helpers import *

def try_linRegressors(regressions_method, data, label):
    ''' To try all the regressors and select the one that gives the smallest RMSE

    :param regressions_method: methods to apply regressions, these methods are from sklearn (linear_model.method)
    :param data: data set, onwhich we are going to apply a cross-validation
    :param label: the real ratings for the "data" given
    :return: best_reg (the best regressor), best_rmse (the rmse associated with the best regressor)
    '''

    # Define the folds for the cross validation
    n_fold = 3
    kf = KFold(n_fold)

    best_rmse = 1000 # Put very large like that it will be update to the RMSE find with one of the method
    #******* select the best regressor ******
    for i,regressor in enumerate(regressions_method): # Loop over all the regressors
        rmse = 0 # Reset for each regressor
        count_fold = 0 # for the "print" in the cross-validation
        print('regression loop:\t{}/{}'.format(i+1, len(regressions_method)))
        reg = regressor
        for train, test in kf.split(data): # Loop for the cross-validation
            count_fold += 1 # Only for the print below
            print('\tFold {}'.format(count_fold))

            reg.fit(data[train,:], label[train]) # Training of the regressor
            #pred = data[test,:].dot(reg.coef_) # Compute the prediction by multiply each column (feature) by the coef find in the training of the reg
            pred = reg.predict(data[test,:])
            rmse += np.sqrt(mean_squared_error(label[test], pred)) # Compute rmse

        rmse = rmse / n_fold # Compute the mean of the RMSE over the folds
        if rmse < best_rmse: # Condition that allow us to choose the best regressors by taking the smallest rmse
            best_rmse = rmse
            best_reg = reg

    return best_reg, best_rmse

def lin_regressors(data, regressions_method, regressor_hyperparam = None):
    ''' To apply every linear regressors on the predictions from all the algorithms

    :param data: predictions from every algorithms
    :param regressions_method: methods to apply regressions, these methods are from sklearn (linear_model.method)
    :param regressor_hyperparam: if there are some, otherwise = None
    :return: best_lin_regressors = the best regressors based on RMSE, best_rmse = rmse associated with the best regressor
    '''

    ########## Splitting the data in training and test set #############
    #******** split the data in predictions and label ******
    data_predictions_np, data_label_np = get_label_predictions(data)

    #******** Add the offset **********
    col_one = pd.DataFrame(np.ones(data_predictions_np.shape[0])) # Creation of the column of 1 in DataFrame
    data_predictions_np= pd.DataFrame(data_predictions_np) # Cast data_predictions_np in DataFrame
    data_predictions_np = pd.concat([col_one, data_predictions_np], axis=1) # Add the column of 1
    data_predictions_np = data_predictions_np.as_matrix()


    ########### Try all the linear regressors and select the best one ######
    best_lin_regressor, best_rmse = try_linRegressors(regressions_method, data_predictions_np, data_label_np)


    return best_lin_regressor, best_rmse

def optimization_regressor(data, regressor, hyperparameters):
    ''' To optimize the hyperparameters of the regressor

    :param data: predictions from every algorithms
    :param regressor: method to apply regression
    :param hyperparameters: the different possible values of certain parameter of the regressor
    :return: regressor_optimized: contain the regressor with all the optimization information
    '''

    #******** split the data in predictions and label ******
    data_predictions_np, data_label_np = get_label_predictions(data)

    #******** GridSearchCV ********************
    regressor_optimized = GridSearchCV(regressor, param_grid=hyperparameters, verbose = 1)
    regressor_optimized.fit(data_predictions_np, data_label_np)
    
    
    return regressor_optimized






