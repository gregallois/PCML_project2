# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:51:59 2016
helpers_marc.py
@author: Marc
"""

from itertools import groupby
from helpers import *
import numpy as np
import scipy.sparse as sp
import math
import csv

#%% For loading the data 


def select_indices(d, n,  p):
    """
        select at random a fraction p of the indices in [0,d-1]x[0,n-1]
        returns the selected in test_indices and the non selected in train_indices
        inputs : 
        
        "d" : int, number of rows
        "n" : int, number of columns
        "p" : float, fraction that should be selected
        
        outputs : 
        "test_indices" : list of tuples (item,user) belonging to the test set
        "training_indices": list of tuples (item,user) belonging to the train set
    """
    
    # Initiate the list of indices 
    indices = []
    for i in range(d):
        for j in range(n):
            indices.append((i,j))
    
    # shuffle the list
    np.random.shuffle(indices)
    
    # select the first fraction
    num_select = int(n*d*p)

    test_indices = indices[:num_select]
    train_indices = indices[num_select:]
    
    # return the sets of indices 
    return test_indices, train_indices
    
    
def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """
    Take the ratings matrix (item x users)
    Split it into training data and test data.
    
    Inputs:
        ratings : sparse matrix item x user containing the ratings
        num_items_per_user : 1D array containing for each user the number of items it has rated
        num_users_per_item : 1D array containing for each item the number of users that rated it.
        min_num_ratings : threshold for selecting the users and items we keep
        p_test : fraction of the selected ratings that should be affected to the test set
        
    Outputs :
        train : sparse matrix items x user, copy of ratings from which the values of the test set have been removed
        test : sparse matrix items x user, copy of ratings from which the values of the training set have been removed
        
    """
    # set seed
    np.random.seed(988)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users]  
    
    #print("valid ratings selected")    
    
    # numbers of items and users
    d, n = valid_ratings.shape
    
    # training and test matrices
    train = sp.lil_matrix((d, n))
    test = sp.lil_matrix((d, n))
    
    # split the indices 
    test_indices, train_indices = select_indices(d, n,  p_test)
    
    
    
    # fill the two matrices with 1 indicating the appartenance
    for train_i in train_indices:
        train[train_i] = valid_ratings[train_i]
    for test_i in test_indices:
        test[test_i] = valid_ratings[test_i] 
    
    
    # multiply cell by cell train with valid_ratings and test with valid ratings
    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

    
    
def calculate_mse_from_matrix(M, prediction):
    """
        calculate the MSE (restricted to non zeros entries of the original matrix) of the prediction matrix 
        
        inputs : 
        "M" : 2D sparse matrix (of ints), original ratings matrix
        "prediction" : 2D dense matrix of same dimensions, predicted ratings matrix 
        
        outputs : 
        "mse" : float, mean square error of the prediction with respect to the original matrix 
    """
    
    mse = 0
    #list of indices associated to non zero entries
    nz_row, nz_col = M.nonzero()
    indices = list(zip(nz_row, nz_col))
    
    for indic in indices:
        mse = mse + 0.5*(M[indic]-prediction[indic])**2
    return mse
    

#%% Initialization of the Matrix Factorization 

def init_MF(train, num_features):
    """
        This function initializes the matrix factorization 
        
        inputs : 
        " train " : item x user sparse matrix, the matrix we want to factorize
        " num_features " : int, number of features that we want for the matrix factorization
        
        outputs :
        "user_features" : (num_features x user) matrix of floats, the matrix of features associated to the users
        "item_features" : (num_features x item) matrix of floats, the matrix of features associated to the items
        
    """
    
    num_items, num_users = train.shape 
    # naive initialization 
    user_features = np.ones((num_features,num_users))/(num_features*num_users)
    item_features = np.ones((num_features, num_items))/(num_features*num_users)
    return user_features,item_features


#%% Compute the error of the matrix factorization    
def compute_error(data, user_features, item_features, nz):
    """
    function that computes the loss (RMSE) associated to the predicted features (with respect to the original matrix : data)
    
    inputs :
    " data ": (item x user) sparse matrix, original ratings matrix
    " user_features" : (num_features x user) dense matrix, features predicted for the users
    " item_features" : (num_features x item) matrix, features predicted for the items
    " nz" : number of non zero cells in the matrix data
    
    outputs :
    "rmse" : float, rmse (root mean square error) of the prediction with respect to the original ratings, the rmse is computed over the set of indices associated to non zero original entries
    
    """
    predict = np.dot(np.transpose(item_features),user_features)
    rmse = math.sqrt(2*calculate_mse_from_matrix(data, predict)/data.nnz)
    
    return rmse


#%% Matrix factorization using SGD

def matrix_factorization_SGD(train, test, K, num_epochs, lambda_user, lambda_item, gamma):
    """
        function that computes the matrix factorization of the training matrix, 
        The Matrix factorization is computed through Stochastic Gradient Descent of the regularized MSE function 
        (MSE with respect to the non-zero entries of train only) 
        It also computes and prints the rmse (over the test set) of this matrix factorization 
        
        inputs : 
        "train" : (item x user) sparse matrix, contains the ratings of the training set, matrix to factorize.
        "test" : (item x user) sparse matrix, contains the ratings of the test set, used to measure the accuracy of the matrix factorization.
        "K" : int, number of features for the matrix factorization
        "num_epochs" : number of full passes of the gradient descent through the whole set of non zero entries
        "lambda_user " : float, >0, regularization parameter associated to the frobenius norm of the user features
        "lambda_item " : float, >0, regularization parameter associated to the frobenius norm of the item features
        "gamma" : float, >0 initial step of descent of the stochastic gradient
        
        outputs :
        "user_features" : (num_features x user) matrix of floats, contains the predicted features associated to the users
        "item_features" : (num_features x item) matrix of floats, contains the predicted features associated to the items
        "rmse_test" : rmse of the factorization matrix found, computed with respect to the entries of the test set
    """
    
    num_features = K   
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        print("full pass number : ", it)
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
        # update user features and item_features
        # stochastic gradient with the function associated to d, n 
        # don't touch the other d',n'     
            new_user_features = user_features[:,n]
            new_item_features = item_features[:,d]
            for k in range(num_features):
                # updating w[k,d]
                gradi = -2.0*user_features[k,n]*(train[d,n]-np.dot(item_features[:,d],np.transpose(user_features[:,n])))+lambda_item*item_features[k,d]
                new_item_features[k] = new_item_features[k] - gamma*gradi
                # updating z[k,n]
                gradu = -2.0*item_features[k,d]*(train[d,n]-np.dot(item_features[:,d],np.transpose(user_features[:,n])))+lambda_user*user_features[k,n]
                new_user_features[k] = new_user_features[k] - gamma*gradu

            # storing the update 
            user_features[:,n] = new_user_features
            item_features[:,d] = new_item_features

        
        #errors.append(rmse)
    rmse_train = compute_error(train, user_features, item_features, nz_train)
    rmse_test = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on train data: {}.".format(rmse_train))
    print("RMSE on test data: {}.".format(rmse_test))
    
    return user_features, item_features, rmse_test, rmse_train


    
def matrix2file(filename, prediction, evaluated_on):
    """
    Creates an output file in csv format for submission to kaggle
    """
    print("Creating csv result file")
    with open(filename, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for user in range(prediction.shape[1]):
            required_rates = np.where(evaluated_on[:, user]>0)[0]
            for i in range(required_rates.shape[0]): 
                writer.writerow({'Id': 'r'+str(required_rates[i]+1)+'_c'+str(user+1),'Prediction': prediction[required_rates[i], user]})
