import numpy as np
from matrix_factorization import *


def build_k_indices(train_set, k_fold,seed):
    ''' Here we want to get the indices of the train_kfold_set and the indices of the test_kfold_set

    :param train_set: the training set
    :param k_fold: number of folds
    :param seed: to randomize
    :return: array of indices
    '''

    num_row = train_set.shape[0]
    interval = int(num_row/k_fold) #Here we calculate the lenght of a k-fold
    np.random.seed(seed)
    indices = np.random.permutation(num_row) # we create the index set randomly ordered 
    k_indices = [indices[k * interval: (k + 1) * interval] # we create the k-fold indices
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(train_set, k_indices, k):
    ''' Here we want to perform a cross validation of the training set
       What we will do is to split the train set in two set. The test set
       containing the k-fold indices of the train_set and the train set the
       rest.

    :param train_set: the training set
    :param k_indices: the indices for the fold
    :param k: number of fold
    :return: cv_tr: training, cv_te: test
    '''

    
    index_set = np.arange(train_set.shape[0]) # We create the whole index
    tr_index = set(k_indices[k,:]).symmetric_difference(index_set) # Creation of the train set which is the non intersection of the k-fold indix.
    tr_index = np.array(list(tr_index)) # Convert set to array
    
    cv_te = train_set[k_indices[k,:],:] # K-fold test set
    cv_tr = train_set[tr_index,:] # K-fold train set
    
    return cv_tr, cv_te
    


def cross_validation_application(train_set,default,k_fold,gamma, num_features, lambda_user, lambda_movie,seed):
    ''' This is the main function that will help us to find the optimal set of parametters i.e. The number of features
        , the penality term for the users, and the penality term for the movies.

    :param train_set: training data
    :param default: boolean
    :param k_fold: number of k_fold
    :param gamma: hyperparameter
    :param num_features: number of features
    :param lambda_user: regularization term for the user
    :param lambda_movie: regularization term for the movie
    :param seed: to randomize
    :return: validation_error: the error on the validation set
    '''


    # split data in k fold
    k_indices = build_k_indices(train_set, k_fold, seed) # creating the k-fold indices
    validation_rmse = []
    validation_error = []
    for i in range(k_fold):
        
        cv_tr, cv_te = cross_validation(train_set, k_indices, i) # Creating the k-fold data sets
        rmse = matrix_factorization_SGD(cv_tr, cv_te, default ,gamma, num_features, lambda_user, lambda_movie) # calculating the rmse of the k-fold.
        validation_rmse.append(rmse)
    
    validation_error = np.mean(validation_rmse)
    return validation_error