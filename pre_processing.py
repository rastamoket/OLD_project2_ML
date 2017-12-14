# -*- coding: utf-8 -*-
''' Some functions to do some pre-processing '''
import scipy.sparse as sp # In order to use sparse Matrix
import pandas as pd
import numpy as np 

def valid_ratings(ratings, num_items_per_user, num_users_per_item, min_num_ratings): # This is based on the ex10
    ''' To select only the users and items that give "enough" ratings'''
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]

    valid_ratings = ratings[valid_items, :][: , valid_users]      
    return valid_ratings

def split_data(ratings, prob_test=0.1): # This is based on the ex10
    ''' To split the data set in training and test '''
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
    
def formating_data_surprise(ratings):
    movies, users, ratings_nnz = sp.find(ratings.T)
    IDs_dict = {'movies ID': movies+1,
                'ratings': ratings_nnz,
                'users ID': users+1
               }

    ratings_representation = pd.DataFrame.from_dict(IDs_dict) # Creation of the dataframe from the dictionary

    return ratings_representation[ratings_representation.ratings != 0]
    