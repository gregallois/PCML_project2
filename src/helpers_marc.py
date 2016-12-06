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

################################################################################################################################################################# FUNCTIONS FOR THE DATA Pre-Processing      #####################################################

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
    Select the items that have at least min_num_ratings, select the users that have rated at least min_num_ratings movies.
    Then, 
    Split the selected ratings into training data and test data.
    
    Inputs:
        ratings : sparse matrix item x user containing the ratings
        min_num_ratings: int, all users and items we keep must have at least min_num_ratings. 
        num_items_per_user : 1D array, one cell per user, containing the number of movies rated by this user
        num_users_per_item : 1D array, one cell per item, containing the number of rates for this movie
        p_test : fraction of the selected ratings that should be affected to the test set
        
    Outputs :
        valid_ratings : sparse matrix items x users reduced from the ratings matrix
        train : matrix items x user, copy of valid_ratings from which the values of the test set have been removed
        test : matrix items x user, copy of valid_ratings from which the values of the training set have been removed
        
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

################################################################################################################################################################# FUNCTIONS FOR THE SGD MATRIX FACTORIZATION #####################################################

def nz_vector(v):
    """
        return the list of indices i of vector v such that v[i] != 0
        inputs :
        "v" : 1D array
        outputs : 
        "nz" : list of ints (indices)
    """
    l = len(v)
    nz =[]
    for i in range(l):
        if v[i]!=0:
            nz.append(i)
    return nz
    

def non_zero_indices(M): 
    """ 
        return the list of indices (i,j) of M that are associated to a non_zero value
        
        inputs :
        "M" : 2D array
        
        outputs : 
        "nz" : list of int x int tuples (indices tuples)
    
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
        calculate the MSE (restricted to non zeros entries of the original matrix) of the prediction matrix 
        
        inputs : 
        "M" : 2D sparse matrix (of ints), original ratings matrix
        "prediction" : 2D matrix of same dimensions, predicted ratings matrix 
        
        outputs : 
        "mse" : float, mean square error of the prediction with respect to the original matrix 
    """
    # initialize the mse
    mse = 0
    # compute the list of indices tuples associated to the non-zero values of M
    indices = non_zero_indices(M)
    
    for indic in indices:
        # update the MSE
        mse = mse + 0.5*(M[indic]-prediction[indic])**2
    
    return mse
    

#%% Initialization of the Matrix Factorization 

def init_MF(train, num_features):
    """
        This function initializes the matrix factorization 
        ( first version : a very basic way)
        
        inputs : 
        " train " : item x user matrix, the matrix we want to factorize
        " num_features " : int, number of features that we want for the matrix factorization
        
        outputs :
        "user_features" : (num_features x user) matrix of floats, the matrix of features associated to the users
        "item_features" : (num_features x item) matrix of floats, the matrix of features associated to the items
        
    """
    # get the shape of the matrix to factorize
    num_items, num_users = train.shape 
    # initialize the features by a small constant  (ones matrix normalized)
    user_features = np.ones((num_features,num_users))/(1.0*num_features*num_users)
    item_features = np.ones((num_features, num_items))/(1.0*num_features*num_users)
    
    # return the output matrices
    return user_features,item_features

    
def compute_error(data, user_features, item_features, nz):
    """
    
    function that computes the loss (RMSE) associated to the predicted features (with respect to the original matrix : data)
    
    inputs :
    " data ": (item x user) matrix, original ratings matrix
    " user_features" : (num_features x user) matrix, features predicted for the users
    " item_features" : (num_features x item) matrix, features predicted for the items
    " nz " : finally unused entry - to be removed
    
    outputs :
    "rmse" : float, rmse (root mean square error) of the prediction with respect to the original ratings, the rmse is computed over the set of indices associated to non zero original entries
    
    """
    
    # compute the prediction from the features
    predict = np.dot(np.transpose(item_features),user_features)
    # compute the rmse 
    rmse = math.sqrt(2*calculate_mse_from_matrix(data, predict)/data.nnz)
    # return the rmse
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
    
    # K in the lecture notes
    num_features = K   
    
    # set seed
    np.random.seed(988)

    # initialize the  matrix factorization
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices (using functions of the td)
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    
    
    print("learn the matrix factorization using SGD...")
    # start the gradient descent
    for it in range(num_epochs):
        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        # one iteration of the descent by non_zero indice 
        for d, n in nz_train:
            
            # update user features and item_features
            # stochastic gradient with the function associated to d, n 
            # don't update the other d',n'
            
            # auxiliary variables
            new_user_features = user_features[:,n]
            new_item_features = item_features[:,d]
            
            
            for k in range(num_features):
                
                # update w[k,d] (item features)
                
                # compute the stochastic gradient associated to k-th features
                gradi = -2.0*user_features[k,n]*(train[d,n]-np.dot(item_features[:,d],np.transpose(user_features[:,n])))+lambda_item * item_features[k,d]
                # update the k-th features
                new_item_features[k] = new_item_features[k] - gamma*gradi
                
                # update z[k,n] (user features)
                
                # compute the stochastic gradient associated to k-th features
                gradu = -2.0*item_features[k,d]*(train[d,n]-np.dot(item_features[:,d],np.transpose(user_features[:,n])))+lambda_user*user_features[k,n]
                # update the k-th features
                new_user_features[k] = new_user_features[k] - gamma*gradu

            # store the update 
            user_features[:,n] = new_user_features
            item_features[:,d] = new_item_features
        
        # compute the loss over the training set (rmse) and print it 
        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
    # matrix factorization done
    
    # compute the loss over the test set to measure the accuracy
    rmse_test = compute_error(test, user_features, item_features, nz_test)
    # print it
    print("RMSE on test data: {}.".format(rmse_test))
    
    return user_features, item_features, rmse_test


################################################################################################################################################################# FUNCTIONS FOR THE ALS MATRIX FACTORIZATION #####################################################


def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """
        Subfunction of the Alternative Least Square Matrix Factorization
        This function updates the user feature by optimizing the mse loss function 
        ( computed over non zeros entries of the original matrix only)
        given the item_features.
        The optimization is performed thanks to an analytical formula.
        
        inputs :
        "train" : (item x user) sparse matrix of ints, contains the original ratings, matrix to factorize
        "item_features" : (num_feature x item) matrix of floats, contains the features of the items
        "lambda_user" : float, >0, regularization parameter associated to the user_features matrix frobenius norm
        "nnz_items_per_user" : list of ints, contains for each user the number of items it has rated
        "nz_user_itemindices" : list of list of ints, contains for each user the list of indices it has rated
        
        output :
        "user_features" : (num_feature x user) matrix of floats, contains the new features of the users
    """
    
    # compute the number of user and features
    num_user   = train.shape[1]
    num_feature = item_features.shape[0]
    
    # initialize the new user features
    user_features = np.zeros((num_feature,num_user))
    
    # number of non zero entries in the original ratings matrix
    nnz = train.nnz
    
    # transform the original ratings matrix from a sparse matrix to a dense one (later, inversion impossible otherwise)
    train = train.todense()
    
    # start the update 
    for n in range(num_user):
        # for each user, compute its new features
        # (see analytical formula for more details)
        lambd = lambda_user * nnz
        X = train[nz_user_itemindices[n],n]
        W = item_features[:,nz_user_itemindices[n]]
        gram = np.dot(W,np.transpose(W))
        A = gram + lambd * np.identity(num_feature)
        b = np.dot(W,X)
        
        # conversion of c necessary 
        c= np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            c[i] = b[i,0]
        
        # affect the new features
        user_features[:,n] = np.linalg.solve(A,c)
    
    return user_features
    

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """
        Subfunction of the Alternative Least Square Matrix Factorization
        This function updates the item features by optimizing the mse loss function 
        ( computed over non zeros entries of the original matrix only)
        given the user features.
        The optimization is performed thanks to an analytical formula.
        
        inputs :
        "train" : (item x user) sparse matrix of ints, contains the original ratings, matrix to factorize
        "user_features" : (num_feature x user) matrix of floats, contains the features of the users
        "lambda_item" : float, >0, regularization parameter associated to the item_features matrix frobenius norm
        "nnz_user_per_item" : list of ints, contains for each item the number of users it has been rated by
        "nz_item_userindices" : list of list of ints, contains for each item the list of indices it has been rated by
        
        output :
        "item_features" : (num_feature x user) matrix of floats, contains the new features of the users
    """
    
    # number of items and features
    num_item   = train.shape[0]
    num_feature = user_features.shape[0]
    
    # initialize the new item features matrix
    item_features = np.zeros((num_feature,num_item))
    
    # number of non-zero entries in the original rating matrix
    nnz = train.nnz
    # convert the original rating matrix from sparse to dense (to avoid error with linear algebra functions)
    train = train.todense()
    
    # start the update
    for d in range(num_item):
        
        # for each item, find the new analytical best features 
        lambd = lambda_item * nnz
        X = train[d,nz_item_userindices[d]].transpose()
        Z = user_features[:,nz_item_userindices[d]]
        gram = np.dot(Z,np.transpose(Z))
        A = gram + lambd * np.identity(num_feature)
        b = np.dot(Z,X)
        
        # convert b (necessary to avoid errors)
        c= np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            c[i] = b[i,0]
        
        # affect the new features
        item_features[:,d] = np.linalg.solve(A,c)
        
        
    return item_features


def ALS(train, test,K, lambda_user, lambda_item, stop_criterion, it_max):
    """
        This function performs matrix factorization through alternative least square method. 
        At each iteration, it computes the new features of the items according to the previous one of the users.
        Then, reciprocally. Those features correspond to the minimum of the regularized MSE functions given the 
        previous opposite features. 
        It is then an alternative coordinate descent.
        
        inputs : 
        "train" : (item x user) sparse matrix, contains the ratings of the training set, matrix to factorize.
        "test" : (item x user) sparse matrix, contains the ratings of the test set, used to measure the accuracy of the matrix factorization.
        "K" : int, number of features for the matrix factorization
        "lambda_user " : float, >0, regularization parameter associated to the frobenius norm of the user features
        "lambda_item " : float, >0, regularization parameter associated to the frobenius norm of the item features
        "stop_criterion" : float, >0 when during an iteration the rmse is updated from less than this criterion, the 
        algorithms stops
        "it_max" : number of iterations of the algorithm
        
        outputs :
        "user_features" : (num_features x user) matrix of floats, contains the predicted features associated to the users
        "item_features" : (num_features x item) matrix of floats, contains the predicted features associated to the items
        "rmse_test" : rmse of the factorization matrix found, computed with respect to the entries of the test set
    """
    

    # define parameters
    num_features = K  
    
    # initialize the change of rmse between two iterations
    change = 1
    
    # initialize the list of errors commited 
    error_list = [float("inf")]
    
    # set seed
    np.random.seed(988)

    # initialize matrix factorization through naive initialization function
    user_features, item_features = init_MF(train, num_features)
    
    # initialize the counter of iterations
    it = 0 
    
    # number of users and items in the rating matrix
    num_item, num_user = train.shape
    # non zero entries indices of the training set, (tuples, list per item, list per user)
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    # same for the test set
    nz_test, niut, nuit = build_index_groups(test)
    

    # convert the format
    for i in range(len(nz_item_userindices)):
        nz_item_userindices[i] = nz_item_userindices[i][1]
    for i in range(len(nz_user_itemindices)):
        nz_user_itemindices[i] = nz_user_itemindices[i][1]
    
    # initialize a list with the number of ratings per users
    nnz_items_per_user = np.zeros(num_user)
    
    # compute the entries of this list
    for i in range(num_user):
        nnz_items_per_user[i] = len(nz_user_itemindices[i])
    
    # same thing for the items    
    nnz_users_per_item = np.zeros(num_item)
    for j in range(num_item):
        nnz_users_per_item[j] = len(nz_item_userindices[j])
        
        
    # modification of the initialisation : (more advanced : according to the td)
    # assigning the average rating for the movies as the first row.
    # and other rows : little random numbers 
    im = train.sum(axis =1)
    for i in range(num_item):
        item_features[0,i]=im[i,0]/len(nz_item_userindices[i])
        for k in range(1,num_features):
            item_features[k,i] = item_features[k,i]*np.random.random()/train.nnz  
    
    print("preprocessing done")
    
    # start the alternative coordinate descent 
    while ((change > stop_criterion) &(it<it_max) ):
        # update the number of iterations
        it = it+1 
        print("iteration of the alternative least square : ", it)
        
        # update the user features
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        
        # update the item features
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        # compute the rmse of the matrix factorization (over training set)
        error = compute_error(train, user_features, item_features, nz_train)
        print( "error commited during this iteration : ", error)
        
        # append it to the list of errors
        error_list.append(error)
        change = error_list[-2]-error_list[-1]
        
    # matrix factorization done
    # compute the rmse of this matrix factorization with respect to the test set 
    rmse_test = compute_error(test, user_features, item_features, nz_test)
    
    return user_features, item_features, rmse_test


# function de greg    
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
                    writer.writerow({'Id': 'r'+str(required_rates[i]+1)+'_c'+str(user+1),'Prediction': int(prediction[required_rates[i], user])})
