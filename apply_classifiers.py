# -*- coding: utf-8 -*-
''' Functions to apply classifiers on predictions '''
from classifiers_models import *
from helpers import *
import pandas as pd
from sklearn.model_selection import KFold

def try_allClassifiers(classifiers_method, data, label):
    ''' To try all the classifiers on the same data (Train and Test) in order to find the lowest test error

    :param classifiers_method: contain the methods for each classifiers
    :param data: data set, onwhich we are going to apply a cross-validation
    :param label: the real ratings for the "data" given
    :return: best_clf: the best classifier (based on the lowest rmse), best_rmse: the rmse associated with the best classifier
    '''

    best_rmse = 1000 # This is only in order to define the variables and then it will be updated with the lowest rmse

    # Define the folds for the cross validation
    n_fold = 3
    kf = KFold(n_fold)

    for i,method_clf in enumerate(classifiers_method): # loop over the methods
        rmse = 0 # Reset after each classifier
        count_fold = 0 # This is for "print" in the cross-validation
        print('classifier loop:\t{}/{}'.format(i+1, len(classifiers_method))) # Just to give info on the run
        for train, test in kf.split(data): # Loop for the cross validation
            count_fold += 1
            print('\tFold {}'.format(count_fold))

            clf, train_error, test_error, rmse_cv = method_clf(data[train,:], data[test,:], label[train], label[test]) # Apply the classifier, sum the rmse
            rmse += rmse_cv
        rmse = rmse/n_fold # Compute the mean of the RMSE over the folds

        if rmse < best_rmse: # Find the smallest RMSE --> best classifier
            best_rmse = rmse # Keep the best_rmse
            best_clf = clf # keep the best classifier

    return best_clf, best_rmse
            

def apply_classifier(data, classifiers_method, classifiers_hyperparam = None):
    ''' To apply the classifier on the predictions made by all the algos we used and find the best classifier

    :param data: dataframe, columns: real ratings + each model's predictions, rows: userID and movieID
    :param classifiers_method: method of the classifier we want to use
    :param classifiers_hyperparam: if there are some, otherwise None
    :return: best_clf: the classifier, best_rmse: the lowest rmse (associates with the best_clf)
    '''

    ########### Splitting the data in training and test sets ###############
    data_predictions_np, data_label_np = get_label_predictions(data)

    best_clf, best_rmse = try_allClassifiers(classifiers_method, data_predictions_np, data_label_np) # call the method to choose the best classifier

    return best_clf, best_rmse


