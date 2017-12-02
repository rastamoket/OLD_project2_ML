# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import scipy.sparse as sp # In order to use sparse 
from helpers import *


def init_MF(train, num_features): # Based on ex10
    """init the parameter for matrix factorization."""
    num_users, num_movies = train.shape # To get the dimensions
    
    #********* Initialization of the features Matrix ***********
    users_feat = np.random.rand(num_users, num_features) # rows = users, columns = features
    movies_feat = np.random.rand(num_movies, num_features) # rows = movies, columns = features
    
    return users_feat, movies_feat

def rmse_movie_mean(train, test): # Idea on ex10
    ''' Compute the mean for each movie among the movies 
        For the "train set":
            take each movie (one-by-one), find the non-zero ratings, compute the mean
        For the "test set":
            take the same movie (one by one), find the non-zero ratings, compute the mse (with the train mean)
    '''
    mse = 0 # Initialization of the variable to store the mse
    num_movies = train.shape[1] # Get the number of movies
    
    for ind_m in range(num_movies):
        #********* On the Train set ********
        train_movie = train[:,ind_m] # Take only one column = one movie
        train_movie_nnz = train_movie[train_movie.nonzero()] # Take only the non-zero ratings in this column (movie)
        
        if train_movie_nnz.shape[0] != 0: # The movie has been at least one time rated
            mean_train = train_movie_nnz.mean() # Compute the mean for this movie
        
        #********* On the Test set *********
        test_movie = test[:,ind_m] # Take only one column = one movie (same as the one used to compute mean of the train)
        test_movie_nnz = test_movie[test_movie.nonzero()].todense() # take only the non-zero ratings, the "todense()" is here to be able to handle test_movie as a numpy 
        
        mse += calculate_mse(test_movie_nnz, mean_train) # to compute the mse for the whole test
    return np.sqrt(mse/test.nnz)

    
    

