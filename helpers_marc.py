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
        args : 
        "d" : number of rows
        "n" : number of columns
        "p" : fraction that should be selected
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
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
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

#%% Auxiliary functions for the matrix factorization

def nz_vector(v):
    """
        return the list of indices i of v such that v[i] != 0
    """
    l = len(v)
    nz =[]
    for i in range(l):
        if v[i]!=0:
            nz.append(i)
    return nz
    

def non_zero_indices(M): 
    """ return the list of indices (i,j) of M that are non_zero
    """
    m, n =np.shape(M)
    nz=[]
    for i in range(m):
        for j in range(n): 
            if (M[i,j]!=0):
                nz.append((i,j))
                
    return nz
    
    
def calculate_mse_from_matrix(M, prediction):
    """
        calculate the MSE using the original matrix and the prediction matrix
    """
    mse = 0
    indices = non_zero_indices(M)
    for indic in indices:
        mse = mse + 0.5*(M[indic]-prediction[indic])**2
    #print(rmse)
    #rmse = math.sqrt(2*mse)
    return mse
    

#%% Initialization of the Matrix Factorization 

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    
    num_items, num_users = train.shape 
    user_features = np.ones((num_features,num_users))/(num_features*num_users)
    item_features = np.ones((num_features, num_items))/(num_features*num_users)
    return user_features,item_features


#%% Compute the error of the matrix factorization    
def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""

    predict = np.dot(np.transpose(item_features),user_features)
    rmse = math.sqrt(2*calculate_mse_from_matrix(data, predict)/data.nnz)
    
    return rmse


#%% Matrix factorization using SGD

def matrix_factorization_SGD(train, test, K, num_epochs, lambda_user, lambda_item, gamma):
    """matrix factorization by SGD."""
    
    num_features = K   # K in the lecture notes
       # number of full passes through the train set
    errors = [0]
    
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
        #computing the error 
        #rmse = compute_error(train, user_features, item_features, nz_train)
        #print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        #errors.append(rmse)
    rmse_train = compute_error(train, user_features, item_features, nz_train)
    rmse_test = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on train data: {}.".format(rmse_train))
    print("RMSE on test data: {}.".format(rmse_test))
    
    return user_features, item_features, rmse_test, rmse_train

#print("matrix factorization read again")

#%% Matrix factorization using Alternative Least Squares method

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    
    num_user   = train.shape[1]
    num_feature = item_features.shape[0]
    user_features = np.zeros((num_feature,num_user))
    nnz = train.nnz
    train = train.todense()
    
    for n in range(num_user):
        #lambd = lambda_user * nnz_items_per_user[n]
        lambd = lambda_user * nnz
        X = train[nz_user_itemindices[n],n]
        W = item_features[:,nz_user_itemindices[n]]
        gram = np.dot(W,np.transpose(W))
        A = gram + lambd * np.identity(num_feature)
        b = np.dot(W,X)
        
        c= np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            c[i] = b[i,0]
        
        user_features[:,n] = np.linalg.solve(A,c)
    
    return user_features
    

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    
    num_item   = train.shape[0]
    num_feature = user_features.shape[0]
    item_features = np.zeros((num_feature,num_item))
    nnz = train.nnz
    train = train.todense()
    
    for d in range(num_item):
        
        #lambd = lambda_item * nnz_users_per_item[d]
        lambd = lambda_item * nnz
        X = train[d,nz_item_userindices[d]].transpose()
        Z = user_features[:,nz_item_userindices[d]]
        gram = np.dot(Z,np.transpose(Z))
        A = gram + lambd * np.identity(num_feature)
        b = np.dot(Z,X)
        
        c= np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            c[i] = b[i,0]
        
        item_features[:,d] = np.linalg.solve(A,c)
        
        
    return item_features


def ALS(train, test,K, lambda_user, lambda_item, stop_criterion, it_max):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = K   # K in the lecture notes
    #lambda_user = 1e-8
    #lambda_item = 1e-8
    #stop_criterion = 1e-4
    change = 1
    error_list = [float("inf")]
    #it_max = 20
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # ALS algorithm
    # counters and variables 
    it = 0 
    
    num_item, num_user = train.shape
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    print("Here: " + str(len(nz_item_userindices)))
    nz_test, niut, nuit = build_index_groups(test)
    
    for i in range(len(nz_item_userindices)):
        nz_item_userindices[i] = nz_item_userindices[i][1]
    for j in range(len(nz_user_itemindices)):
        nz_user_itemindices[j] = nz_user_itemindices[j][1]
        
    nnz_items_per_user = np.zeros(num_user)
    
    for i in range(num_user):
        nnz_items_per_user[i] = len(nz_user_itemindices[i])
        
    nnz_users_per_item = np.zeros(num_item)
    
    #print(len(nnz_users_per_item))
    #print(len(nz_item_userindices))
    #print("J max:" + str(9991))*
    
    for j in range(num_item):
        nnz_users_per_item[j] = len(nz_item_userindices[j])
    
    #nnz_items_per_user = np.zeros(len(nz_user_itemindices))
    #for i in range(len(nz_user_itemindices)):
    #    nnz_items_per_user[i] = len(nz_user_itemindices[i])
        
    #nnz_users_per_item = np.zeros(len(nz_item_userindices))

    
    #for j in range(len(nz_item_userindices)):
    #    nnz_users_per_item[j] = len(nz_item_userindices[j])

        
        
    # modification of the initialisation : 
    # assigning the average rating for the movies as the first row.
    im = train.sum(axis =1)
    for i in range(num_item):
    #for i in range(len(nnz_users_per_item)):    
        item_features[0,i]=im[i,0]/len(nz_item_userindices[i])
        for k in range(1,num_features):
            item_features[k,i] = item_features[k,i]*np.random.random()/train.nnz  
    
    #print(item_features)
    
    print("preprocessing done")
    while ((change > stop_criterion) &(it<it_max) ): 
        it = it+1 
        print("iteration of the alternative least square : ", it)
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)
        error = compute_error(train, user_features, item_features, nz_train)
        print( "error commited during this iteration : ", error)
        error_list.append(error)
        change = error_list[-2]-error_list[-1]
        
    rmse_test = compute_error(test, user_features, item_features, nz_test)
    
    return user_features, item_features, rmse_test


    
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
