import numpy as np
import matplotlib.pyplot as plt
import math 

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


def opt_split_data(ratings, p_test=0.1):
    """
    Take the ratings matrix (item x users)
    Split it into training data and test data.
    
    Inputs:
        ratings : dense matrix item x user containing the ratings
        p_test : fraction of the selected ratings that should be affected to the test set
        
    Outputs :
        train : matrix items x user, copy of ratings from which the values of the test set have been removed
        test : matrix items x user, copy of ratings from which the values of the training set have been removed
        
    """
    # set seed
    np.random.seed(988)
    
    # numbers of items and users
    d, n = ratings.shape
    
    # training and test matrices
    train = np.zeros((d, n))
    test = np.zeros((d, n))
    
    # split the indices 
    test_indices, train_indices = select_indices(d, n,  p_test)
    
    # fill the two matrices 
    for train_i in train_indices:
        train[train_i] = ratings[train_i]
    for test_i in test_indices:
        test[test_i] = ratings[test_i] 
   
    return train, test



def opt_calculate_mse_from_matrix(M, prediction):
    """
        calculate the MSE (restricted to non zeros entries of the original matrix) of the prediction matrix 
        
        inputs : 
        "M" : 2D dense matrix (of ints), original ratings matrix
        "prediction" : 2D matrix of same dimensions, predicted ratings matrix 
        
        outputs : 
        "mse" : float, mean square error of the prediction with respect to the original matrix 
    """
    
    
    # remove from the prediction matrix the predictions associated to zero entries 
    # of the original matrix 
    it,us = np.nonzero(M)

    prediction2 = np.zeros(prediction.shape,dtype='float16')
    
    
    for i in range(len(it)):
        prediction2[it[i],us[i]] = prediction[it[i],us[i]]
        
    mse = (np.linalg.norm((M - prediction2), ord='fro')**2)
    
    return mse


def opt_compute_error(data, user_features, item_features,nnz_data):
    """
    
    function that computes the loss (RMSE) associated to the predicted features (with respect to the original matrix : data)
    
    inputs :
    " data ": (item x user) matrix, original ratings matrix
    " user_features" : (num_features x user) matrix, features predicted for the users
    " item_features" : (num_features x item) matrix, features predicted for the items
    " nnz_data" : number of non zero cells in the matrix data
    
    outputs :
    "rmse" : float, rmse (root mean square error) of the prediction with respect to the original ratings, the rmse is computed over the set of indices associated to non zero original entries
    
    """
    
    # compute the prediction from the features
    predict = np.dot(np.transpose(item_features),user_features)
    # compute the rmse 
    rmse = math.sqrt(2*opt_calculate_mse_from_matrix(data, predict)/nnz_data)
    # return the rmse
    return rmse


def opt_init_MF(train, num_features):
    """
        This function initializes the matrix factorization 
        
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
    #user_features = np.random.rand(num_features,num_users)/(2**5)
    user_features = np.ones((num_features,num_users),dtype='float16')/(2**4)
    #user_features = user_features.astype(dtype ='float128')
    #item_features = np.random.rand(num_features,num_items)/(2**5)
    item_features = np.ones((num_features,num_items),dtype='float16')/(2**4)
    #item_features = item_features.astype(dtype = 'float128')
    
    # fill the matrix with the initialization seen in the lecture : mean of the ratings, then random little numbers
    for i in range(num_users):
        user_features[0,i] = train[:,i].sum()/np.count_nonzero(train[:,i])
    for i in  range(num_items):
        item_features[0,i] = train[i,:].sum()/np.count_nonzero(train[i,:])

    # return the output matrices
    return user_features,item_features



def opt_matrix_factorization_SGD(train, test, K, num_epochs, lambda_user, lambda_item, gamma):
    """
        function that computes the matrix factorization of the training matrix, 
        The Matrix factorization is computed through Stochastic Gradient Descent of the regularized MSE function 
        (MSE with respect to the non-zero entries of train only) 
        It also computes and prints the rmse (over the test set) of this matrix factorization 
        
        inputs : 
        "train" : (item x user) matrix, contains the ratings of the training set, matrix to factorize.
        "test" : (item x user) matrix, contains the ratings of the test set, used to measure the accuracy of the matrix factorization.
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
    np.random.seed()

    # initialize the  matrix factorization
    user_features, item_features = opt_init_MF(train, num_features)
    
    print("initialization of the factorization : done")
                                         
    # compute the tuples of non zero indices of the training matrix 
    nz_train_item,nz_train_users = np.nonzero(train)
    
    # compute the number of such tuples
    nnz_train = np.count_nonzero(train)
    
    # list of ints from 0 to nnztrain -1 (to reindex the tuples later)
    range_nnz_train = np.arange(nnz_train)
    
    print("learn the matrix factorization using SGD...")
    
    # start the gradient descent
    for it in range(num_epochs):
        
        # shuffle the training rating indices
        np.random.shuffle(range_nnz_train)
        
        # decrease step size
        gamma /= 1.2
        
        # one iteration of the descent by non_zero indice 
        for i in range_nnz_train:
            # d,n associated to the index
            d = nz_train_item[i]
            n = nz_train_users[i]
            
            # update user features and item_features
            # stochastic gradient with the function associated to d, n 
            # don't update the other d',n'
            
            # compute the gradient associated to the item d
            gradi = -2.0*user_features[:,n]*(train[d,n] -np.dot(item_features[:,d],np.transpose(user_features[:,n])))+lambda_item * item_features[:,d]
            # store the new values of the features of d in an auxiliary variable
            aux = item_features[:,d] - gamma*gradi
            
            # compute the gradient associated to the user n
            gradu = -2.0*item_features[:,d]*(train[d,n]-np.dot(item_features[:,d],np.transpose(user_features[:,n])))+lambda_user*user_features[:,n]    
               
            # update the features of n and the features of d
            user_features[:,n] = user_features[:,n] - gamma * gradu
            item_features[:,d] = aux
        
        # compute the loss over the training set (rmse) and print it 
        rmse = opt_compute_error(train, user_features, item_features, nnz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
    # matrix factorization done
    nnz_test = np.count_nonzero(test)
    # compute the loss over the test set to measure the accuracy
    rmse_test = opt_compute_error(test, user_features, item_features, nnz_test)
    # print it
    print("RMSE on test data: {}.".format(rmse_test))
    
    return user_features, item_features, rmse_test