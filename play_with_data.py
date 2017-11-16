# -*- coding: utf-8 -*-
"""some functions to better understand the data"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def info_general(nUser, nItem, x, data):
    print("The number of zero in the data given:\t{}".format(np.where(x == 0)[0].shape[0]))
    print("The number of ratings we have:\t\t\t{}".format(x.shape[0]))
    print("The number of missing values we expected:\t{}".format(nUser*nItem - x.shape[0]))
    print("The number of missing values:\t\t\t{}".format(np.where(data == 0)[0].shape[0]))
    print("Pourcentage of missing values:\t{} %".format(np.where(data == 0)[0].shape[0] * 100/(nUser * nItem)))
    print("Pourcentage of good values:\t{} %".format(x.shape[0] * 100 / (nUser * nItem)))
    if(nUser*nItem - x.shape[0] == np.where(data == 0)[0].shape[0]): # if expected missing = really missing
        print("\nThe loading of the data is well done")
        
def info_ratings(data):
    ratings = np.arange(1,6)
    nb_ratings = np.zeros_like(ratings)
    
    for r in ratings:
        nb_ratings[r-1] = np.where(data == r)[0].shape[0]
        print("The ratings {} is present {} times".format(r, nb_ratings[r-1]))
    
    plt.close('all')
    plt.figure()
    plt.bar(ratings, nb_ratings)
    plt.show()
        
        
              