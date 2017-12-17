# -*- coding: utf-8 -*-
''' Functions to apply classifiers on predictions '''
from classifiers_models import *
from helpers import *
import pandas as pd

def try_allClassifiers(classifiers_method, x_tr, x_te, y_tr, y_te):
    ''' To try all the classifiers on the same data (Train and Test) in order to find the lowest test error

    :param classifiers_method: contain the methods for each classifiers
    :param x_tr: data for training
    :param x_te: data for test
    :param y_tr: label for training
    :param y_te: label for test
    :return:
    '''

    best_test_error = 1 # This is only in order to define the variables and then it will be updated with the lowest test_error

    for method_clf in classifiers_method: # loop over the methods
        clf, train_error, test_error = method_clf(x_tr, x_te, y_tr, y_te)
        if test_error < best_test_error:
            best_test_error = test_error
            best_clf = clf
    return best_clf, best_test_error
            

def apply_classifier(data, classifiers_method, classifiers_hyperparam = None, ratio = 0.85):
    ''' To apply the classifier on the predictions made by all the algos we used and find the best classifier

    :param data: dataframe, columns: real ratings + each model's predictions, rows: userID and movieID
    :param classifiers_method: method of the classifier we want to use
    :param classifiers_hyperparam: if there are some, otherwise None
    :param ratio: for the splitting in training and test set (default = 0.85)
    :return: best_clf: the classifier, best_test_error: the lowest test error (associates with the best_clf)
    '''

    ########### Splitting the data in training and test sets ###############
    data_predictions_np, data_label_np = get_label_predictions(data)
    
    #******* Apply the splitting *******
    x_tr, x_te, y_tr, y_te = split_predictions_data(data_predictions_np, data_label_np, ratio) # call the method to split the predictions

    ########### Apply all the classifiers and select the best one ##############
    best_clf, best_test_error = try_allClassifiers(classifiers_method, x_tr, x_te, y_tr, y_te)

    return best_clf, best_test_error


