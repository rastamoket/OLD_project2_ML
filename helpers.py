# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from itertools import groupby
import scipy.sparse as sp # In order to use sparse 
import csv


def load_data_old(dataSetCSVfile): # This come from from the first project (modified)
    """Load data and convert it to the metrics system."""
    path_dataset = dataSetCSVfile # path of the data we want to load
    # To get the id of the samples
    rowCol_samples = np.genfromtxt(
        path_dataset, dtype=np.unicode_, delimiter=",", skip_header=1, usecols=0, deletechars ='r')


    # "data" will contain all the features for each samples so it will be a NxM matrix (with N = sumber of sample, M = number of features)
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=1)
    return rowCol_samples, data

def create_csv_submission(row_users, col_movies, estim, name): # This come from the first project
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: row_users (the indices of the users)
               col_movies (the indices of the movies)
               estim (the estimated ratings)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w',  newline='') as csvfile:
        fieldnames = ['Id', 'Prediction'] # Define the fieldnames
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames) # creation of the writer with a dictionary
        writer.writeheader() # To write the header row
        for r,c,e in zip(row_users, col_movies, estim): # Loop over the indices and ratings
            writer.writerow({'Id':'r{}_c{}'.format(r, c),'Prediction':int(e)}) # In order to have everything in each row in the right format




def read_txt(path): # This come from the ex10, 'helpers.py'
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset): # This come from the ex10, 'helpers.py'
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data): # This come from the ex10, 'helpers.py'
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of users: {}, number of items: {}".format(max_row, max_col)) # Little modification here, in order to print the correct numbers

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index): # This come from the ex10, 'helpers.py'
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train): # This come from the ex10, 'helpers.py'
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction): # This come from the ex10, 'helpers.py'
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)
