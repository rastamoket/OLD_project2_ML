''' Best model: SVDpp optimized '''

import numpy as np
import pandas as pd
import re # Used for the conversion of "r##_c##" in only the numbers --> TODO: check where it comes from
from IPython.display import display
from helpers import *
from play_with_data import *
from pre_processing import *
from matrix_factorization import *
from cross_validation import *
from apply_classifiers import *
from trainings_submissions import *
from regressions_models import *
from majority_mean import *
import scipy.sparse as sp # In order to use sparse
# Predictors imported in performance order (best to worst, according to http://surpriselib.com/)
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
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run():
    ''' Run our best model and create the prediction '''
    ############ Loading the data ################
    print('Loading the data')
    #******** Creation of a sparse matrix of the data **********
    ratings = load_data('./data_train.csv')
    test = load_data('./sample_submission.csv')

    ########### Define: algo, dataset (trainset ##############
    ratings_ = formating_data_surprise(ratings)
    trainset = ratings_.build_full_trainset()

    ########### Define: testset ##############
    dataF_test_ratings_, test_ratings_ = formating_data_surprise(test, True)
    test_trainset = test_ratings_.build_full_trainset()
    testset = test_trainset.build_testset()

    ########## Algorithm ###############
    algorithm_sim = SVDpp(n_factors=10,lr_all=0.00177827941004,reg_all=0.001)
    print('Training of the algorithm')
    algorithm_sim.train(trainset)
    pred = algorithm_sim.test(testset)

    ########## Apply the algorithm ########
    print('Predict')
    row_users = [] # initialization of the list row_users
    col_movies = [] # initialization of the list col_movies
    estim = [] # initialization of the list estim
    for p in pred: # To loop over the prediction done by the algo on the test set
        row_users.append(p.uid) # fill this list with the indices of the users
        col_movies.append(p.iid) # fill this list with the indices of the movies
        estim.append(p.est) # fill this list with the ratings

    print('create csv file')
    name = 'best_model_submission.csv'
    create_csv_submission(row_users, col_movies, estim, name) # To create the CSV file
    print('DONE')
