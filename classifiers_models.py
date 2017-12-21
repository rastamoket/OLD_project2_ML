# -*- coding: utf-8 -*-
""" Different classifiers """
import numpy as np
import scipy.sparse as sp # In order to use sparse
import sklearn as sk
# Import for sklearn, in order to implement the different classifiers
from sklearn.model_selection import KFold # for cross validation
from sklearn.neighbors import KNeighborsClassifier # For k nearest neighbor
from sklearn import tree # For the decision tree
from sklearn.neural_network import MLPClassifier # Neural network
from sklearn import svm # Support Vector Machine
from sklearn.naive_bayes import GaussianNB # Naive Bayes, Gaussian
from sklearn.naive_bayes import MultinomialNB # Naive Bayes, multinomial
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Linear Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # Quadratic Discriminant Analysis
from sklearn.metrics import mean_squared_error

def prediction_error(lab_real, lab_pred):
    ''' To compute the error of prediction. This is done by comparison of the prediction and the real labels

    :param lab_real: real label
    :param lab_pred: prediction
    :return: prediction error (number of wrong classification / number total of label)
    '''
    return np.where(lab_real != lab_pred)[0].shape[0]/lab_real.shape[0]

############## Methods for Naive Bayes #################
def training_gaussNB(x_tr, y_tr, hyperparam = None):
    ''' Train the classifier Naive Bayes (gaussian)

    :param x_tr: training data
    :param y_tr: training label
    :param hyperparam: if there are some (their values) (default = None)
    :return: clf: the classifier trained
    '''

    if hyperparam is not None:
        clf = GaussianNB() # If any hyperparameters --> add here
    else:
        clf = GaussianNB()
    return clf.fit(x_tr, y_tr)

def naive_bayes(x_tr, x_te, y_tr, y_te, hyperparam = None):
    ''' Method to train and test naive bayes

    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :param hyperparam: if there are some (their values) (default = None)
    :return: clf: classifier that is trained,
            train_error: error of prediction on the training set,
            test_error: error of prediction on the test set,
            rmse: root mean squared error with prediction and real ratings
    '''

    if hyperparam is not None:
        clf = training_gaussNB(x_tr, y_tr, hyperparam)
    else:
        clf = training_gaussNB(x_tr, y_tr)

    y_pred_test = clf.predict(x_te)
    y_pred_train = clf.predict(x_tr)

    train_error = prediction_error(y_tr, y_pred_train)
    test_error = prediction_error(y_te, y_pred_test)

    #******* Compute RMSE *********
    rmse = mean_squared_error(y_te, y_pred_test)

    return clf, train_error, test_error, rmse


############## Methods for K-nearest Neighbors #################
def training_kNeigh(x_tr, y_tr, hyperparam = None):
    ''' Train the classifier for K nearest neighbor

    :param x_tr: training data
    :param y_tr: training label
    :param hyperparam: if there are some (their values) (default = None)
    :return: neigh: the classifier trained
    '''

    if hyperparam is not None:
        neigh = KNeighborsClassifier(n_neighbors=hyperparam)
    else:
        neigh = KNeighborsClassifier()
    return neigh.fit(x_tr, y_tr)

def kNearestNeigh(x_tr, x_te, y_tr, y_te, hyperparam = None):
    ''' Method to train and test K nearest neighbor

    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :param hyperparam: if there are some (their values) (default = None)
    :return: neigh: classifier that is trained,
            train_error: error of prediction on the training set,
            test_error: error of prediction on the test set,
            rmse: root mean squared error with prediction and real ratings
    '''

    if hyperparam is not None:
        neigh = training_kNeigh(x_tr, y_tr, hyperparam)
    else:
        neigh = training_kNeigh(x_tr, y_tr)

    y_pred_test = neigh.predict(x_te)
    y_pred_train = neigh.predict(x_tr)

    train_error = prediction_error(y_tr, y_pred_train)
    test_error = prediction_error(y_te, y_pred_test)

    #******* Compute RMSE *********
    rmse = mean_squared_error(y_te, y_pred_test)

    return neigh, train_error, test_error, rmse


############## Methods for Decision Tree #################
def decision_tree(x_tr, x_te, y_tr, y_te, hyperparam=None):
    ''' Method to train and test decision tree

    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :param hyperparam: if there are some (their values) (default = None)
    :return: clf: classifier that is trained,
            train_error: error of prediction on the training set,
            test_error: error of prediction on the test set,
            rmse: root mean squared error with prediction and real ratings
    '''

    if hyperparam == None: # For the case without hyperparam to optimize
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_tr, y_tr)
    else: # If we try to optimize an hyperparam
        clf = tree.DecisionTreeClassifier(max_depth=hyperparam[0], min_samples_leaf=hyperparam[1]) # Max_depth was optimize with a cross validation
        clf = clf.fit(x_tr, y_tr)

    y_pred_train = clf.predict(x_tr)
    y_pred_test = clf.predict(x_te)

    train_error = prediction_error(y_tr, y_pred_train)
    test_error = prediction_error(y_te, y_pred_test)

    #******* Compute RMSE *********
    rmse = mean_squared_error(y_te, y_pred_test)

    return clf, train_error, test_error, rmse


############## Methods for Neural Networks #################
def neural_net(x_tr, x_te, y_tr, y_te, hyperparam = None):
    ''' Method to train and test neural net

    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :param hyperparam: if there are some (their values) (default = None)
    :return: clf: classifier that is trained,
            train_error: error of prediction on the training set,
            test_error: error of prediction on the test set,
            rmse: root mean squared error with prediction and real ratings
    '''

    if hyperparam is not None:
        clf = MLPClassifier(solver='lbfgs', alpha=hyperparam)
        clf.fit(x_tr, y_tr)
    else:
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5)
        clf.fit(x_tr, y_tr)

    y_pred_train = clf.predict(x_tr)
    y_pred_test = clf.predict(x_te)

    train_error = prediction_error(y_tr, y_pred_train)
    test_error = prediction_error(y_te, y_pred_test)

    #******* Compute RMSE *********
    rmse = mean_squared_error(y_te, y_pred_test)

    return clf, train_error, test_error, rmse

################# Method for SVM ####################
def support_vectorMachine(x_tr, x_te, y_tr, y_te, hyperparam = None):
    ''' Method to train and test support vector machine

    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :param hyperparam: if there are some (their values) (default = None)
    :return: lin_clf: classifier that is trained,
            train_error: error of prediction on the training set,
            test_error: error of prediction on the test set,
            rmse: root mean squared error with prediction and real ratings
    '''

    if hyperparam is not None:
        lin_clf = svm.LinearSVC() # Add the hyperparam
        lin_clf.fit(x_tr, y_tr)
    else:
        lin_clf = svm.LinearSVC()
        lin_clf.fit(x_tr, y_tr)

    y_pred_train = lin_clf.predict(x_tr)
    y_pred_test = lin_clf.predict(x_te)

    train_error = prediction_error(y_tr, y_pred_train)
    test_error = prediction_error(y_te, y_pred_test)

    #******* Compute RMSE *********
    rmse = mean_squared_error(y_te, y_pred_test)

    return lin_clf, train_error, test_error, rmse

############## Methods for quadratic discr. analysis #################
def training_QDA(x_tr, y_tr, hyperparam = None):
    ''' Train the classifier Naive Bayes (gaussian)

    :param x_tr: training data
    :param y_tr: training label
    :param hyperparam: if there are some (their values) (default = None)
    :return: clf: the classifier trained
    '''

    if hyperparam is not None:
        clf = QuadraticDiscriminantAnalysis(reg_param=hyperparam)
    else:
        clf = QuadraticDiscriminantAnalysis()
    return clf.fit(x_tr, y_tr)

def discr_analysis(x_tr, x_te, y_tr, y_te, hyperparam = None):
    ''' Method to train and test QDA

    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :param hyperparam: if there are some (their values) (default = None)
    :return: clf: classifier that is trained,
            train_error: error of prediction on the training set,
            test_error: error of prediction on the test set,
            rmse: root mean squared error with prediction and real ratings
    '''

    if hyperparam is not None:
        clf = training_QDA(x_tr, y_tr, hyperparam)
    else:
        clf = training_QDA(x_tr, y_tr)

    clf.fit(x_tr, y_tr)

    y_pred_train = clf.predict(x_tr)
    y_pred_test = clf.predict(x_te)

    train_error = prediction_error(y_tr, y_pred_train)
    test_error = prediction_error(y_te, y_pred_test)

    #******* Compute RMSE *********
    rmse = mean_squared_error(y_te, y_pred_test)

    return clf, train_error, test_error, rmse

############## Methods for Linear discriminant analysis #################
def lin_discr_analysis(x_tr, x_te, y_tr, y_te, hyperparam = None):
    ''' Method to train and test LDA

    :param x_tr: training data
    :param x_te: test data
    :param y_tr: training label
    :param y_te: test label
    :param hyperparam: if there are some (their values) (default = None)
    :return: clf: classifier that is trained,
            train_error: error of prediction on the training set,
            test_error: error of prediction on the test set,
            rmse: root mean squared error with prediction and real ratings
    '''

    if hyperparam is not None:
        clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=hyperparam)
        clf.fit(x_tr, y_tr)
    else:
        clf = LinearDiscriminantAnalysis()
        clf.fit(x_tr, y_tr)

    y_pred_train = clf.predict(x_tr)
    y_pred_test = clf.predict(x_te)

    train_error = prediction_error(y_tr, y_pred_train)
    test_error = prediction_error(y_te, y_pred_test)

    #******* Compute RMSE *********
    rmse = mean_squared_error(y_te, y_pred_test)

    return clf, train_error, test_error, rmse
