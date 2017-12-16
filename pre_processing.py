# -*- coding: utf-8 -*-
''' Some functions to do the pre-processing '''
import scipy.sparse as sp # In order to use sparse Matrix
import pandas as pd
import numpy as np 
from surprise import dataset
from surprise import Dataset

def valid_ratings(ratings, num_items_per_user, num_users_per_item, min_num_ratings): # This is based on the ex10
    ''' To select only the users and items that give "enough" ratings
        Arguments:
            rating: the original values 
            num_items_per_user: the number of movies rated per user
            num_users_per_item: the number of users that had rate the movie
            min_num_ratings: number of minimal ratings we want for users and movies
        Returns:
            valid_ratings: only the ratings for the users and movies that has at least min_num_ratings
    '''
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0] # Take only the users that have at least the min_num_ratings (indices)
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0] # Take only the movies that have at least the min_num_ratings (indices)

    valid_ratings = ratings[valid_items, :][: , valid_users] # Create the matrix with only the valid users and items       
    return valid_ratings

def split_data(ratings, prob_test=0.1): # This is based on the ex10
    ''' To split the data set in training and test 
        Arguments:
            ratings: the matrix of ratings that we want to split
            prob_test: the probability to put in the test set (default = 0.1)
        Returns:
            train: training matrix (90% of the ratings for each column)
            test: test matrix (10% of the ratings for each column)
    '''
    #******** Creation of the two matrices ***********
    train = sp.lil_matrix((ratings.shape[0], ratings.shape[1])) # Training set 
    test = sp.lil_matrix((ratings.shape[0], ratings.shape[1])) # Test set
    
    #******** Distribution of the samples in the two matrices ********
    nonz_users, nonz_items = ratings.nonzero() # to have the indices of the elements that have non-zero rating
    
    for nz_u in set(nonz_users): # Loop over all the indices of the users that contain at least one non-zero value
        r,c = ratings[nz_u, :].nonzero() # To have only the non-zero ratings for this user "nz_u"
        te = np.random.choice(c, size=round(len(c)*prob_test)) # We choose randomly which values will go to the test (for each column)
        tr = list(set(c) - set(te)) # All the others elements in this column are in the train
        
        test[nz_u,te] = ratings[nz_u,te] # Fill in the matrix test
        train[nz_u, tr] = ratings[nz_u, tr] # Fill in the matrix train
            
    return train, test
    
def formating_data_surprise(ratings, dataF_return = False):
    ''' To put the data in the format needed to use "surprise"
        Arguments:
            ratings: the ratings (sparse matrix)
            dataF_return: to choose if we want to return or not the dataframe (default = False)
        Returns:
            dataF: dataframe with movies ID, ratings, users ID (only returned if dataF_return = True)
            Dataset.load_from_df: this is a dataset for surprise    
    '''
    movies, users, ratings_nnz = sp.find(ratings.T) # get the movies, users and non-zero ratings
    # Create the dictionary with all the values (movies ID, ratings, users ID)
    IDs_dict = {'movies ID': movies+1,
                'ratings': ratings_nnz,
                'users ID': users+1
               }

    ratings_representation = pd.DataFrame.from_dict(IDs_dict) # Creation of the dataframe from the dictionary

    dataF = ratings_representation[ratings_representation.ratings != 0] # Take only the ratings that are non zero (this is maybe already good)
    reader = dataset.Reader(rating_scale=(1, 5)) # Creation of the reader to create a dataset (ratings scale: from 1 to 5), this is something from surprise
    if dataF_return: # Condition to return also the dataF
        return dataF, Dataset.load_from_df(dataF[['users ID', 'movies ID', 'ratings']], reader)
    else:
        return Dataset.load_from_df(dataF[['users ID', 'movies ID', 'ratings']], reader)
    