# -*- coding: utf-8 -*-
from classifiers_models import *
import pandas as pd



def split_predictions_data(x, y, ratio, seed=1): # Come from project 1
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row) # to randomize
    index_split = int(np.floor(ratio * num_row)) # where we split
    index_tr = indices[: index_split] # indexes for training
    index_te = indices[index_split:] # indexes for test
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]

    return x_tr, x_te, y_tr, y_te

def apply_classifier(data):
    ########### Splitting the data in training and test sets ###############
    #******* Split in label and predictions from data *********
    data_predictions = data.copy() # In order to not modify the original data
    data_predictions = data_predictions.drop('Label', axis = 1) # Contain only the predictions made by each model
    data_label = data['Label'] # Contain only the labels --> the real ratings

    #******* Transform in numpy ********
    data_predictions_np = data_predictions.as_matrix()
    data_label_np = data_label.as_matrix()

    #******* Apply the splitting *******
    x_tr, x_te, y_tr, y_te = split_predictions_data(data_predictions_np, data_label_np, 0.85)

    ########### Apply the classifier ##############
    clf, train_error, test_error = support_vectorMachine(x_tr, x_te, y_tr, y_te)

    return clf


