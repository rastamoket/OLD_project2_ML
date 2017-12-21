# -*- coding: utf-8 -*-
""" Run everything to obtain our best prediction """
from helpers import *
from trainings_submissions import *
import pandas as pd
import numpy as np
from surprise import SVDpp
from surprise import KNNBaseline
from surprise import SVD
from surprise import SlopeOne
from surprise import BaselineOnly
from surprise import KNNWithMeans
from surprise import NMF
from surprise import CoClustering
from surprise import KNNBasic
from surprise import KNNWithZScore # not scored --> to be tested quickly
from surprise import dataset
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import GridSearch
from surprise import accuracy
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


def run():
    ''' To run everything to obtain the best prediction

    :return: Nothing, it creates a .CSV file that is ready for the online submission
    '''

    ############# Load the data #############
    print('Loading the data')
    ratings = load_data('./data_train.csv') # Load the training set
    test = load_data('./sample_submission.csv') # Load the test set, in order to create the submission file later

    ############# Apply algorithms from Surprise #############
    print('Initializing the algorithms')
    #************ Initialize the lists of the algorithms with parameters already optimized ************
    algorithms = [SVDpp(n_factors=10,lr_all=0.00177827941004,reg_all=0.001),
             KNNBaseline(k=96, min_k=8,sim_options={'name': 'pearson_baseline','user_based': False,'shrinkage': 500},
                         bsl_options={'method': 'als','reg_u': 14.4,'reg_i': 0.3}),
             NMF(n_factors=35,reg_pu=10**(-1.5),reg_qi=10**(-0.5)),
             SVD(n_factors=100,lr_all=0.001,reg_all=10**(-1.5)),
             SlopeOne(),
             BaselineOnly(bsl_options={'method': 'als', 'reg_u': 14.4, 'reg_i': 0.3}),
             KNNWithZScore(k=100, min_k=7, sim_options={'name':'pearson_baseline','user_based':False,'shrinkage':500})
             ]

    columns_name = ['SVDpp', # This list is usefull to define the name of the columns of the predictions
                    'KNNBaseline',
                    'NMF',
                    'SVD',
                    'SlopeOne',
                    'BaselineOnly',
                    'KNNWithZScore'
                    ]

    #*********** TRAINING **************

    #----------- Splitting ------------
    train_algo_ratings, train_reg_ratings = split_data(ratings, prob_test=0.3) # splitting in train and test set
    #----------- Training the algorithms on "train_algo_ratings" and apply them on "train_reg_ratings" ---------------
    print('Training of the algorithms')
    algos_trained = first_train(ratings, algorithms) # Trained the algorithms on the "train_algo_ratings" set

    #******** prepare the validation set *********
    print('---- Start the predictions -----')
    train_reg_df, train_reg_surprise = formating_data_surprise(train_reg_ratings, True) # Formating the data in order to use Surprise
    train_reg_set = train_reg_surprise.build_full_trainset() # Build trainset
    reg_set_pred = train_reg_set.build_testset() # Build iterable object in order to test

    prediction_reg_df = train_reg_df.copy() # Initialization of the DataFrame we will return
    ######### Predictions by the trained algorithms #############
    for i, algo_t in enumerate(algos_trained): # Loop over all the trained algorithms
        pred = algo_t.test(reg_set_pred) # Make the prediction

        ########## Creation of the list: estim ########
        estim = [] # initialization of the list estim

        for p in pred: # To loop over the prediction done by the algo on the test set
            estim.append(p.est) # fill this list with the ratings

        d = {'prediction' : pd.Series(estim)}
        temp = pd.DataFrame(d)
        prediction_reg_df = pd.concat([prediction_reg_df,temp], axis=1)
    first_col = ['movies ID', 'Label', 'users ID'] # In order to put the right name on the columns
    all_col = first_col + columns_name # In order to put the right name on the columns
    prediction_reg_df.columns = all_col # In order to put the right name on the columns
    print('---- End of the predictions -----')
    #----------- Training the regressor on the predictions on the "train_reg_ratings" ---------------

    prediction_reg_cleaned = prediction_reg_df.copy() # Copy the original data, keep it intact
    prediction_reg_cleaned = prediction_reg_cleaned.drop(['movies ID', 'users ID'], axis = 1) # Remove the columns we don't need

    print('Apply the regressor')
    regressor =  linear_model.SGDRegressor(alpha = 0.0001, epsilon= 0.01, l1_ratio= 0.3) # This is the best regressor we've found and optimized
    training_predictions_set, training_predictions_label = get_label_predictions(prediction_reg_cleaned) # Take the predictions and the labels

    #___________ Adding the offset parameter (column of 1) __________________
    col_one = pd.DataFrame(np.ones(training_predictions_set.shape[0])) # Create a column of ones (offset parameter)
    training_predictions_set= pd.DataFrame(training_predictions_set) # Put the training_prediction_set in Dataframe type
    training_predictions_set = pd.concat([col_one, training_predictions_set], axis=1) # Add the column of 1, the offset at the prediction set

    #___________ Training of the regressor _______________
    regressor.fit(training_predictions_set, training_predictions_label) # Here we do the training, find the weights of the regression


    ############ Predict the unknown ratings #########
    print('------ Start the predictions on the unknown --------')
    #*********** Prepare the test set ***************
    test_df, test_surprise = formating_data_surprise(test, True) #Put the data in the correct format
    test_set = test_surprise.build_full_trainset() # Build trainset
    test_set_pred = test_set.build_testset() # Build iterable object in order to test
    print('\tApply the algorithms')
    prediction_test_df = test_df.copy() # Initialization of the DataFrame we will return
    #*********** Prediction **************
    for i, algo_t in enumerate(algos_trained): # Loop over all the trained algorithms
        pred = algo_t.test(test_set_pred) # Make the prediction

        #_________ Creation of the list: estim __________
        estim = [] # initialization of the list estim

        for p in pred: # To loop over the prediction done by the algo on the test set
            estim.append(p.est) # fill this list with the ratings

        d = {'prediction' : pd.Series(estim)}
        temp = pd.DataFrame(d)
        prediction_test_df = pd.concat([prediction_test_df,temp], axis=1)
    first_col = ['movies ID', 'Label', 'users ID'] # In order to put the right name on the columns
    all_col = first_col + columns_name # In order to put the right name on the columns
    prediction_test_df.columns = all_col # In order to put the right name on the columns

    #___________ Remove the not wanted columns _______________
    predictions_only = prediction_test_df.copy() # Copy in order to not act on the original one
    predictions_only = predictions_only.drop(predictions_only.columns['movies ID','users ID', 'Label'], axis = 1) # remove the "label", "movies ID" and "users ID" column, keep only the predictions

    #___________ Adding the offset parameter (column of 1) __________________
    col_one_unknown = pd.DataFrame(np.ones(predictions_only.shape[0])) # Create a column of ones (offset parameter)
    predictions_only = pd.concat([col_one_unknown, predictions_only], axis=1) # Add the column of 1, the offset at the prediction set

    #----------- Apply regression on the predictions of the unknown ratings ------------------
    print('\tApply the regressor')
    moviesID_usersID_prediction = prediction_test_df['movies ID','users ID']
    predicted = predictions_only.dot(regressor.coef_) # Compute the predictions of the unknown ratings
    moviesID_usersID_prediction['Prediction'] = predicted # Now the variable "movies_usersID_df" contains all the values we need to create the submission file

    ############ Creation of the submission file #############
    print('Create the submission file')
    name = 'best_submission.csv'
    create_csv_submission(moviesID_usersID_prediction['users ID'], moviesID_usersID_prediction['movies ID'], moviesID_usersID_prediction['Prediction'], name)
