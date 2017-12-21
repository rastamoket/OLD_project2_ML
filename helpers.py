# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import pandas as pd
from itertools import groupby
import scipy.sparse as sp # In order to use sparse 
import csv


def load_data_old(dataSetCSVfile): # This come from from the first project (modified)
    '''Load data and convert it to the metrics system.

    :param dataSetCSVfile: path of the CSV file
    :return: rowCol_samples: position (r##_c##), data: ratings
    '''

    path_dataset = dataSetCSVfile # path of the data we want to load
    # To get the id of the samples
    rowCol_samples = np.genfromtxt(
        path_dataset, dtype=np.unicode_, delimiter=",", skip_header=1, usecols=0, deletechars ='r')


    # "data" will contain all the features for each samples so it will be a NxM matrix (with N = sumber of sample, M = number of features)
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=1)
    return rowCol_samples, data

def create_csv_submission(row_users, col_movies, estim, name):
    ''' Creates an output file in csv format for submission to kaggle

    :param row_users: the indices of the users
    :param col_movies: the indices of the movies
    :param estim: the estimated ratings
    :param name: string name of .csv output file to be created
    :return: Nothing, it creates a csv file
    '''

    with open(name, 'w',  newline='') as csvfile:
        fieldnames = ['Id', 'Prediction'] # Define the fieldnames
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames) # creation of the writer with a dictionary
        writer.writeheader() # To write the header row
        for r,c,e in zip(row_users, col_movies, estim): # Loop over the indices and ratings
            writer.writerow({'Id':'r{}_c{}'.format(r, c),'Prediction':e}) # In order to have everything in each row in the right format




def read_txt(path): # This come from the ex10, 'helpers.py'
    ''' to read txt

    :param path: path of the file
    :return: what is read
    '''

    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset): # This come from the ex10, 'helpers.py'
    ''' Load data in text format, one rating per line, as in the kaggle competition

    :param path_dataset: path of the file that contains the data (.csv)
    :return: the pre-processed data
    '''

    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data): # This come from the ex10, 'helpers.py'
    ''' preprocessing the text data, conversion to numerical array format

    :param data: original data set
    :return: ratings: pre-processed data
    '''
    """."""
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
    ''' group list of list by a specific index

    :param data: what we want to sort
    :param index: index
    :return: groupby_data: data sorted
    '''

    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train): # This come from the ex10, 'helpers.py'
    ''' build groups for nnz rows and cols

    :param train: train set that need to be split several fold
    :return: nz_train, nz_row_colindices, nz_col_rowindices
    '''

    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

def get_label_predictions(data):
    ''' To get the label (real ratings) and the predictions from the data

    :param data: dataframe, columns: real ratings + each model's predictions, rows: userID and movieID
    :return: data_predictions_np: all the predicitions from each algos (as numpy array), data_label_np: label (real ratings) (as numpy array)
    '''

    ########### Splitting the data in training and test sets ###############
    #******* Split in label and predictions from data *********
    data_predictions = data.copy() # In order to not modify the original data
    data_predictions = data_predictions.drop('Label', axis = 1) # Contain only the predictions made by each model
    data_label = data['Label'] # Contain only the labels --> the real ratings

    #******* Transform in numpy ********
    data_predictions_np = data_predictions.as_matrix()
    data_label_np = data_label.as_matrix()

    return data_predictions_np, data_label_np

def split_predictions_data(x, y, ratio, seed=1): # Come from project 1
    ''' split the dataset based on the split ratio

    :param x: data set
    :param y: label
    :param ratio: ratio of data that will be in the training set
    :param seed: to randomize
    :return: x_tr: training data, x_te: test data, y_tr: training label, y_te: test label
    '''

    np.random.seed(seed) #set seed
    #******* generate random indices ********
    num_row = len(y) # get the number of rows
    indices = np.random.permutation(num_row) # to randomize
    index_split = int(np.floor(ratio * num_row)) # where we split
    index_tr = indices[: index_split] # indexes for training
    index_te = indices[index_split:] # indexes for test

    #******** create split *********
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]

    return x_tr, x_te, y_tr, y_te


def calculate_mse(real_label, prediction): # This come from the ex10, 'helpers.py'
    ''' calculate MSE

    :param real_label: the correct label
    :param prediction: the prediction of the label
    :return: the Mean Squared Error
    '''
    t = real_label - prediction
    return 1.0 * t.dot(t.T)
