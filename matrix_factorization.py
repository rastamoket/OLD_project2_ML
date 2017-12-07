# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import scipy.sparse as sp # In order to use sparse 
from helpers import *

def rmse_movie_mean(train, test): # Idea on ex10
    
    ''' 
        Compute the mean for each movie among the movies 
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

def init_MF(train, num_features): # Based on ex10
    """init the parameter for matrix factorization."""
    num_users, num_movies = train.shape # To get the dimensions
    
    #********* Initialization of the features Matrix ***********
    users_feat = np.random.rand(num_users, num_features) # rows = users, columns = features
    movies_feat = np.random.rand(num_movies, num_features) # rows = movies, columns = features
    # OK
    
    return users_feat, movies_feat

def compute_error(data, user_features, movie_features, nz): # From ex10
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # User_features: 

    mse = 0
    for row, col in nz:

        movie_info = movie_features[col, :]
        user_info = user_features[row, :]
        mse += (data[row, col] - user_info.T.dot(movie_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))

def matrix_factorization_SGD(train, test, default,gamma, num_features, lambda_user, lambda_movie): # From ex10
    """matrix factorization by SGD."""
    # define parameters
    if default:
        gamma = 0.01
        num_features = 20   # K in the lecture notes
        lambda_user = 0.1
        lambda_movie = 0.7

    num_epochs = 20     # number of full passes through the train set
    errors = [0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, movie_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_user_ind, nz_movie_ind = train.nonzero() # To get the indices of the non-zero ratings (train)
    nz_train = list(zip(nz_user_ind, nz_movie_ind)) # To create a list with the indices of non-zero (e.g (user,movie) )
    nz_user_indTest, nz_movie_indTest = test.nonzero() # To get the indices of the non-zero ratings (test)
    nz_test = list(zip(nz_user_indTest, nz_movie_indTest)) # To create a list with the indices of non-zero (e.g (user,movie) )

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # update W_d (movie_features[d,:]) and Z_n (user_features[n,:])
            movie_info = movie_features[n,:]
            user_info = user_features[d, :]
            err = train[d, n] - user_info.T.dot(movie_info)

            # calculate the gradient and update
            movie_features[n, :] += gamma * (err * user_info - lambda_movie * movie_info)
            user_features[d,:] += gamma * (err * movie_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, movie_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        errors.append(rmse)

    # evaluate the test error
    rmse = compute_error(test, user_features, movie_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

# def update_user_features(train, movies_features, lambda_user, nnz_movies_per_user, nz_user_movieIndices): # Based on ex10
#     ''' Do the update of the user_features based on the gradient of the "sum of squared error" with a regularization term
#         Inputs:
#             train: data sparse matrix with a pourcentage of valid ratings
#             movies_features: numpy matrix that contains the features for the movies (num of feat is defined = K)
#             lambda_user: it is the regularization term for the user
#             nnz_movies_per_user: indices of the movies that contains non-zero entries (dimension: n_movies x 1)
#             nz_user_movieIndices: the indices of the movies that contains non-zero ratings for the user
#     '''
#     num_user = nnz_movies_per_user.shape[0]
#     num_feature = movies_features.shape[0]
#     lambda_I = lambda_user * sp.eye(num_feature)
#     updated_user_features = np.zeros((num_feature, num_user))
#
#     for user, items in nz_user_itemindices:
#         # extract the columns corresponding to the prediction for given item
#         M = item_features[:, items]
#
#         # update column row of user features
#         V = M @ train[items, user]
#         A = M @ M.T + nnz_items_per_user[user] * lambda_I
#         X = np.linalg.solve(A, V)
#         updated_user_features[:, user] = np.copy(X.T)
#     return updated_user_features
#
#
# def als
    
    

