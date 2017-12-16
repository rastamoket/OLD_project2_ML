# -*- coding: utf-8 -*-
from classifiers_models import *
import pandas as pd



def split_predictions_data(x, y, ratio, seed=1): # Come from project 1
    """split the dataset based on the split ratio."""
    np.random.seed(seed) #set seed
    #******* generate random indices ********
    num_row = len(y) # get the number of rows
    indices = np.random.permutation(num_row) # to randomize
    index_split = int(np.floor(ratio * num_row)) # where we split
    index_tr = indices[: index_split] # indexes for training
    index_te = indices[index_split:] # indexes for test
    
    #******** create split *********
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]

    return x_tr, x_te, y_tr, y_te

def get_label_predictions(data):
    ''' To get the label (real ratings) and the predictions from the data
        Arguments:
            data: dataframe, columns: real ratings + each model's predictions, rows: userID and movieID
        Returns:
            data_predictions_np: all the predicitions from each algos (as numpy array)
            data_label_np: label (real ratings) (as numpy array)
    '''
    
    ########### Splitting the data in training and test sets ###############
    #******* Split in label and predictions from data *********
    data_predictions = data.copy() # In order to not modify the original data
    data_predictions = data_predictions.drop('Label', axis = 1) # Contain only the predictions made by each model
    data_label = data['Label'] # Contain only the labels --> the real ratings

    #******* Transform in numpy ********
    data_predictions_np = data_predictions.as_matrix()
    data_label_np = data_label.as_matrix()
    
    return data_predictions_np, data_label_np

def try_allClassifiers(classifiers_method, x_tr, x_te, y_tr, y_te):
    ''' To try all the classifiers on the same data (Train and Test) in order to find the lowest test error
        Arguments:
            classifiers_methods: contain the methods for each classifiers
            x_tr, y_tr: data and label for training
            x_te, y_te: data and label for test
        Returns:
            best_clf: the classifier that gives the lowest test error
            best_test_error: the lowest test error (associates with the best_clf)
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
        Arguments:
            data: dataframe, columns: real ratings + each model's predictions, rows: userID and movieID
            classifiers_method: method of the classifier we want to use
            ??? hyperparameter ????
        Returns:
            best_clf: the classifier
            best_test_error: the lowest test error (associates with the best_clf)
    '''
    ########### Splitting the data in training and test sets ###############
    data_predictions_np, data_label_np = get_label_predictions(data)
    
    #******* Apply the splitting *******
    x_tr, x_te, y_tr, y_te = split_predictions_data(data_predictions_np, data_label_np, ratio) # call the method to split the predictions

    ########### Apply all the classifiers and select the best one ##############
    best_clf, best_test_error = try_allClassifiers(classifiers_method, x_tr, x_te, y_tr, y_te)

    return best_clf, best_test_error


