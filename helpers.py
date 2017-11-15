# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(dataSetCSVfile):
    """Load data and convert it to the metrics system."""
    path_dataset = dataSetCSVfile # path of the data we want to load
    # To get the id of the samples
    rowCol_samples = np.genfromtxt(
        path_dataset, dtype=np.unicode_, delimiter=",", skip_header=1, usecols=0, deletechars ='r')


    # "data" will contain all the features for each samples so it will be a NxM matrix (with N = sumber of sample, M = number of features)
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=1)
    return rowCol_samples, data