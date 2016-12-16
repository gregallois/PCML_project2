# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:46:47 2016
run_marc.py
@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
import math 

from itertools import groupby
from helpers import *
from helpers_sgd_matrix_facto import *
from plots import *
import scipy
import scipy.io
import numpy as np
import scipy.sparse as sp
import csv

############################################# Part One : data loading ############################################################

### A) Load the data

# write here the path to the dataset
path_dataset = "datasets/data_train.csv"

# load the data 
ratings = load_data(path_dataset)
print("data loaded")


### B) Split the data

# First get the numbers of ratings per user and ratings per film 
# And plot these values

num_items_per_user, num_users_per_item = plot_raw_data(ratings)

# then select the items and users for which there are enough data
# and split the datan into a training set and a test set

# minimal number of data per user and per item
min_num_ratings = 10 

# fraction of the data set that will be the test set 
p_test = 0.1

valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings, p_test)


########################################## Part Two : Selection of the parameters ###############################################

## Parameters for the matrix factorization

# maximum and minimum numbers of features 
max_K = 9
min_K= 3

# incrementation step for K
step_K = 1

# number of full iterations of the stochastic gradient descent
max_epochs = 30

# regularization parameters (regularization of the loss function in Gradient Descent)
range_lambda_user = np.logspace(-6, -4, 7) 
range_lambda_item = np.logspace(-5, -3, 7)

# descent step size
gamma = 0.1

# initialization of variables refering to the matrix factorization for the best number
best_rmse_test_sgd = float('Inf')
best_k_sgd = 0
best_user_feat_sgd = np.zeros(1)
best_item_feat_sgd = np.zeros(1)
best_lambda_user = 0
best_lambda_item =0

param_rmse = []

# for each lambdas of the grid 
# for each K of the grid, compute the matrix factorization (training set) and the rmse (test set), 
# If it improves the results, update the rmse

for lambda_user in range_lambda_user:
    
    for lambda_item in range_lambda_item: 
        
        for K in range(min_K,max_K,step_K):
    
            print("matrix factorization for the number of features : ", K)

            # compute the stochastic gradient descent matrix factorization
            user_features, item_features, rmse_test, rmse_train = matrix_factorization_SGD(train, test, 
                                                                               K, max_epochs, lambda_user, lambda_item, gamma)
            
            param_rmse.append((lambda_user,lambda_item,K, rmse_test, rmse_train))
            if rmse_test < best_rmse_test_sgd:
                # better rmse => update the references
                best_rmse_test_sgd = rmse_test
                best_k_sgd = K
                best_user_feat_sgd = user_features
                best_item_feat_sgd = item_features
                best_lambda_user = lambda_user
                best_lambda_item = lambda_item

                
########################################## Part Three : Running the process for more full passes ################################


### A) Load the prediction data
path_evaluation = "datasets/sampleSubmission.csv"
ratings_submit = load_data(path_evaluation)
evaluated_on = ratings_submit.toarray()
evaluated_on = evaluated_on.astype(np.float)

# compute the prediction matrix for the best k and all the ratings
# not only those for sufficient amount of data
max_epochs = 80
gamma = 0.1
data_sub = ratings
user_feat_sub, item_feat_sub, rmse_test, rmse_train = matrix_factorization_SGD(data_sub, 
                                                                   data_sub, best_k_sgd, max_epochs, 
                                                                   best_lambda_user, best_lambda_item, gamma)

prediction = np.dot(np.transpose(item_feat_sub), user_feat_sub)
print("prediction matrix computed")


### B) compute the csv file to store the results

filename = "datasets/submission_sgd_test.csv"

matrix2file(filename, prediction, evaluated_on)