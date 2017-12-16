# -*- coding: utf-8 -*-
"""some functions to better understand the data"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def info_general(nUser, nItem, x, data):
    ''' To give some general information
        Arguments:
            nUser: the number of users
            nItem: the number of movies
            x: the ratings
            data: matrix of the data (numpy)
    '''
    print("The number of zero in the data given:\t{}".format(np.where(x == 0)[0].shape[0]))
    print("The number of ratings we have:\t\t\t{}".format(x.shape[0]))
    print("The number of missing values we expected:\t{}".format(nUser*nItem - x.shape[0]))
    print("The number of missing values:\t\t\t{}".format(np.where(data == 0)[0].shape[0]))
    print("Pourcentage of missing values:\t{} %".format(np.where(data == 0)[0].shape[0] * 100/(nUser * nItem)))
    print("Pourcentage of good values:\t{} %".format(x.shape[0] * 100 / (nUser * nItem)))
    if(nUser*nItem - x.shape[0] == np.where(data == 0)[0].shape[0]): # if expected missing = really missing
        print("\nThe loading of the data is well done")
        
def info_ratings(data):
    ''' To give some information about the ratings
        Arguments:
            data: matrix of the data (numpy)
    '''
    ratings = np.arange(1,6)
    nb_ratings = np.zeros_like(ratings)
    
    for r in ratings:
        nb_ratings[r-1] = np.where(data == r)[0].shape[0]
        print("The ratings {} is present {} times".format(r, nb_ratings[r-1]))
    
    plt.close('all')
    plt.figure()
    plt.bar(ratings, nb_ratings)
    plt.xlabel('Ratings')
    plt.ylabel('Number of users')
    plt.show()
    
def plot_raw_data(ratings): # this come from: Machine Learning course, ex10 from the file "plots.py"
    """ plot the statistics result on raw rating data.
        Arguments:
            ratings: matrix of ratings
        Returns:
            num_items_per_user: the number of movies rated per users
            num_users_per_item: the number of users that have rated this movie (for each movie)
            
    """
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    ax2.set_xticks(np.arange(0, ratings.shape[0], 2000))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item
        
        
              