# -*- coding: utf-8 -*-
from classifiers_models import *
from pre_processing import *

def first_train(ratings,algorithm, test = [],submit = False):
    
    if not submit:
        ########### Define: algo, dataset (trainset ##############
        dataF_train, ratings_train = formating_data_surprise(ratings, True) # Create the Dataset for surprise (training set)

        train_set = ratings_train.build_full_trainset() # Build trainset
        trainset_pred = train_set.build_testset() # Build iterable object in order to test 

        # C'est notre test set donc utiliser plus tard
        #ratings_test = formating_data_surprise(validation_ratings)
        #validation_set = ratings_test.build_full_trainset()

        prediction_df = dataF_train.copy()
        
        algos_trained = []
        
        for i, algo in enumerate (algorithm):

            algo.train(train_set) # Training of the algo
            algos_trained.append(algo)
            pred = algo.test(trainset_pred) # Make the prediction

            ########## Creation of the lists: row_users, col_movies, estim ########
            estim = [] # initialization of the list estim

            for p in pred: # To loop over the prediction done by the algo on the test set
                estim.append(p.est) # fill this list with the ratings

            d = {'prediction' : pd.Series(estim)}
            temp = pd.DataFrame(d)
            prediction_df = pd.concat([prediction_df,temp], axis=1)  

        return prediction_df, algos_trained


    else:
        
        ########### Define: algo, dataset (trainset ##############
        data_train_df ,ratings_ = formating_data_surprise(ratings, True) # Create the Dataset for surprise (training set)
        trainset = ratings_.build_full_trainset()
        
        data_test , ratings_test = formating_data_surprise(test, True) # Create the Dataset for surprise (training set)
        test_trainset = ratings_test.build_full_trainset()
        testset = test_trainset.build_testset()
        
        prediction_test_df = data_test.copy()
        for i, algo in enumerate (algorithm):
            algo.train(trainset) # Training of the algo
            pred = algo.test(testset) # Make the prediction

            ########## Creation of the lists: row_users, col_movies, estim ########
            estim = [] # initialization of the list estim

            for p in pred: # To loop over the prediction done by the algo on the test set
                estim.append(p.est) # fill this list with the ratings
    
            d = {'prediction' : pd.Series(estim)}
            temp = pd.DataFrame(d)
            prediction_test_df = pd.concat([prediction_test_df,temp], axis=1)
        
        return prediction_test_df

def second_train_df(df, list_algo_name):
    
    columns = ['Label']
    columns = sum([columns, list_algo_name],[])
    moviesID_userID_df = df[['movies ID','users ID']]
    train_df = df.copy() # Copy 
    train_df = train_df.drop(train_df.columns[[0, 2]], axis=1) # In order to keep only the real ratings and then the predictions for all algos
    train_df.columns = columns
    
    return train_df, moviesID_userID_df