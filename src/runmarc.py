# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:46:47 2016
run_marc.py
@author: Marc
"""

from itertools import groupby
from helpers import *
from helpers_marc import *
from plots import *
import scipy
import scipy.io
import numpy as np
import scipy.sparse as sp


#%% 1) LOAD THE DATA 

# write here the path to the dataset
path_dataset = "../datasets/data_train.csv"

# load the data 
ratings = load_data(path_dataset)

#%% 2) SPLIT THE DATA INTO A TEST AND A VALIDATION SET

# First get the numbers of ratings per user and ratings per film 
# And plot these values

num_items_per_user, num_users_per_item = plot_raw_data(ratings)

# then select the items and users for which there are enough data
# and split the datan into a training set and a test set

# minimal number of data per user and per item
min_num_ratings = 10 

# fraction of the data set that will be the test set 
p_test = 0.1
#%% 

valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings, p_test)

print("data split done")
plt.close()
# plot the resulting training and test set 
plot_train_test_data(train, test)

#%% 3) From the training set, compute the matrix factorization of the ratings 
# matrix 

best_rmse = float('Inf')
best_k = 0
best_user_feat = np.zeros(1)
best_item_feat = np.zeros(1)
user_features, item_features, rmse = matrix_factorization_SGD(train, test, K)