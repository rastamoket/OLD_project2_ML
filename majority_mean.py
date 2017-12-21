# -*- coding: utf-8 -*-
''' Some functions to apply Majority or Mean vote '''

import numpy as np
import pandas as pd
from surprise import dataset
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import GridSearch
from surprise import accuracy
from sklearn.metrics import mean_squared_error

def mean_vote(prediction_df):
    ''' To apply the algorithm "Majority vote" on a data set with given algorithm

    :param prediction_df: contains (from validation set) "movies ID", "users ID", "ratings" and the "prediction"
    :return: final_prediction: prediction done by Mean vote, rmse: root mean squared error of the prediction against the real ratings
    '''

    prediction_df_copy = prediction_df.copy() # Just to keep the original intact
    ######## Prepare the predictions ##########
    only_prediction_df = prediction_df_copy.drop(['movies ID', 'users ID', 'Label'], axis = 1) # Take only the predictions

    ######## Apply Mean vote ############
    final_prediction = only_prediction_df.mean(axis = 1)

    ####### Compute RMSE ##############
    rmse = np.sqrt(mean_squared_error(prediction_df['Label'], final_prediction))

    return final_prediction, rmse



def majority_vote(prediction_df):
    ''' To apply the algorithm "Majority vote" on a data set with given algorithm

    :param prediction_df: contains (from validation set) "movies ID", "users ID", "ratings" and the "prediction"
    :return: final_prediction: prediction done by Majority vote, rmse: root mean squared error of the prediction against the real ratings
    '''

    prediction_df_copy = prediction_df.copy() # Just to keep the original intact

    ######## Prepare the predictions ##########
    only_prediction_df = prediction_df_copy.drop(['movies ID', 'users ID', 'Label'], axis = 1) # Take only the predictions
    only_prediction_df = only_prediction_df.round() # We round the predictions
    only_prediction_df['Majority'] = 0 # Add a new column and set it to 0

    ######## Apply Majority vote ##########
    for i ,row in only_prediction_df.iterrows(): # We iterate over all the rows to analyse each prediction and choose the Majority of the prediction
        row_ = row.as_matrix() # Tranform row in np array
        unique, counts = np.unique(row_, return_counts=True) # We get set of ratings and their respective count repetition
        index_of_max = np.where(counts == np.max(counts)) # We select the index of the max count
        max_ = unique[index_of_max]
        if max_.shape[0]>1: # If there are same amount of max count, we chose the one with the highest score as we saw
            # that the rating distribution is shifter upwards there is
            max_ = max_[-1]
        only_prediction_df.loc[i,'Majority'] = max_

    final_prediction = only_prediction_df['Majority']

    ####### Compute RMSE ##############
    rmse = np.sqrt(mean_squared_error(prediction_df['Label'], final_prediction))

    return final_prediction, rmse

