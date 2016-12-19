import numpy as np
import csv



K = 5




#ratings = ratings.toarray()

#Max rate given by user
K = 5

def compute_user_rate_matrix(ratings_, user_id):
    
    user_rates = ratings_[:, user_id]
    
    #1 for rated movies, 0 elsewhere
    nz = np.nonzero(user_rates)[0]
    
    #number of rated movies
    m = nz.shape[0]

    #what is the index of rated movies
    movie_index = np.where(user_rates >0)[0]
    rate_matrix = np.zeros((m, K), dtype = np.bool)
    for i in range(m):
        rate_matrix[i , user_rates[movie_index[i]]-1]=1
        
    
    return m, movie_index, rate_matrix




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




"""brief : creates the ratings matrix of a given user (one line for each rated movie, 5 columns corresponding to the possible rates)
   param ratings_ : rate matrix
   param user_id : column corresponding to the user
   return m : number of rated movies
   return movie_index : tab containing the ids of rated movies
   return rate_matrix : array corresponding to the format explained in the brief"""

def compute_user_rate_matrix(ratings_, user_id):
    user_rates = ratings_[:, user_id]
    #1 for rated movies, 0 elsewhere
    nz = np.nonzero(user_rates)[0]
    #number of rated movies
    m = nz.shape[0]
    #what is the index of rated movies
    movie_index = np.where(user_rates >0)[0]
    rate_matrix = np.zeros((m, K), dtype = np.bool)
    for i in range(m):
        rate_matrix[i , user_rates[movie_index[i]]-1]=1 
        
    return m, movie_index, rate_matrix


""" brief : Computes sigmoid values of a matrix
    param t : matrix passed to sigmoid function
    retrun sigma : sigmoid(t) """
def sigmoid(t):
    toSigma = t
    sigma = np.ones(toSigma.shape)/(np.ones(toSigma.shape)+np.exp(-1*toSigma)); 
    return sigma

""" brief : Computes p(vi = k|h) for a single user, for vi corresponding to the movie he has rated
    param w : weight matrix
    param b_v : bias matrix
    param m : number of movies by the user
    param movies_index : indices of movies rated by the user
    return p: square matrix corresponding to  p(vi = k|h)"""
def compute_p_v_h(w, b_v, h, m, movies_index):
    h = h.reshape((h.shape[0],1))
    p = np.zeros((m, K))
    for k in range(K):
        w_sel = w[ movies_index,:,k]
        p[:,k] = np.exp(h.T@w_sel.T + b_v[movies_index,k])
    sum_ = np.sum(p, axis = 1)
    for k in range(K):
        p[:,k] = p[:,k]/sum_

    return p


""" brief : Computes p(hi = 1|v) for a single user, for vi corresponding to the movie he has rated
    param w : weight matrix
    param b_v : bias matrix
    m : number of movies by the user
    movies_index : indices of movies rated by the user"""
def compute_p_h_v(w, b_h,  V, m, movies_index, F):
    p_h_v = np.zeros((F,1))
    for j in range(F):
        sum_ = 0
        w_p = w[movies_index, j ,:]
        sum_ = np.sum(np.einsum('ij,ji->i',V, w_p.T))            
        p_h_v[j] = sum_ +b_h[j]
    
    return sigmoid(p_h_v)


"""brief : CD9 implementation
   param w : weight matrix
   param b_h : bias matrix
   param b_v : bias matrix
   param : number of movies rated by the concerned user
   param movies_index : list of indices correponding to the movies rated by the user
   param rate_matrix : matrix returned by compute_user_rate_matrix
   param  p_h_v : p(hi = 1|v) for the concerned user
   param F : number of hidden units
   return 1 : E(hj x vik)
   return 2 : E(vik)
   return 3 : E(hj)"""
def CD_estimate(w, b_h, b_v, m, movies_index, rate_matrix, p_h_v, F):
    CD_iterations = 1;
    esp_iter = 20

    esp_h_t_v = np.zeros((m, F, K))
    esp_v = np.zeros((m, K))
    esp_h = np.zeros((F,1))

    #To compute esperance, we perform things iter time
    for iter_ in range(esp_iter):
        
        #Initialisation with data training case
        #V = rate_matrix
        p_h_v_iter = p_h_v
        
        for cd_iter in range(CD_iterations):
            #computing h sample
            samples = np.random.ranf(p_h_v.shape)
            h = np.greater(p_h_v_iter, samples)
            #computing new V sample
            V = np.zeros(rate_matrix.shape)
            p_v_h_iter = compute_p_v_h(w, b_v, h, m, movies_index)   
            samples = np.random.ranf((m,1))
            V = np.zeros(rate_matrix.shape)
            acc = np.zeros(samples.shape)
            old_acc = np.zeros(samples.shape)

            for k in range(K):
                acc = acc + np.reshape(p_v_h_iter[:,k], acc.shape)
                V[:,k]= np.reshape(np.greater(acc, samples)*np.greater(samples, old_acc), V[:,k].shape)
                old_acc = acc
            p_h_v_iter = compute_p_h_v(w, b_h,  V, m, movies_index, F)

        esp_h = esp_h + h
        esp_v = esp_v + V  

        add_ = np.zeros(esp_h_t_v.shape)
        for j in range(F):
            for k in range(K):
                add_[:,j,k] = h[j]*V[:,k]              
        esp_h_t_v = esp_h_t_v+add_
        
    return esp_h_t_v/esp_iter, esp_v/esp_iter, esp_h/esp_iter


"""brief : compte delta_w, delta_bh, delta_bv for a given user, corresponding to loglikelihood gradient descent
   param w : weight matrix
   param b_h : bias matrix
   param b_v : bias matrix
   param m : number of movies rated by the concerned user
   param movies_index : list of indices correponding to the movies rated by the user
   param rate_matrix : matrix returned by compute_user_rate_matrix
   param F : number of hidden units
   return 1 : delta_w
   return 2 : delta_bh
   return 3 : delta_bv"""
def update(w, b_h, b_v, m, movie_index, rate_matrix, F):
    p_h_v = compute_p_h_v(w, b_h, rate_matrix, m , movie_index, F)
    cd_estimate_hv, cd_estimate_v, cd_estimate_h = CD_estimate(w, b_h, b_v, m, movie_index, rate_matrix, p_h_v, F)

    u_f_t = np.zeros((rate_matrix.shape[0], F, K))
    
    for k in range(K):
        u_f_t[:,:,k] = (p_h_v*rate_matrix[:,k]).T
                
    grad_w = np.zeros(w.shape)
    grad_h = np.zeros(b_h.shape)
    grad_v = np.zeros(b_v.shape)
    grad_w[movie_index, :,:] = u_f_t - cd_estimate_hv
    grad_v[movie_index, :] = (rate_matrix - cd_estimate_v)
    grad_h = (p_h_v-cd_estimate_h)
    return grad_w, grad_h, grad_v



"""brief : intermediate step of unknown ratings estimation, computes p(vik|h) for a particular movie(i) for the concerned user
   param w : weight matrix
   param b_v : bias matrix
   param p_h_v : p(hj=1) for the studied user
   param movie index : indices of the movies rated by concerned user
   return 1:  computes p(vik=1|h) for the concerned user"""
def compute_p_v_p(w, b_v, p_h_v, movie_index):
    p_h_v = np.reshape(p_h_v, (p_h_v.shape[0],1))
    sum_=0
    p = np.zeros((K,1))
    for k in range(K):
        w_j = w[movie_index, :, k]
        w_j = np.reshape(w_j, (w_j.shape[0],1))
        p[k] = np.exp(p_h_v.T@w_j + b_v[movie_index, k])
        sum_ = sum_+p[k]
    return p/sum_



"""brief : estimates the unknown rate for a particular movie(i) for the concerned user
   param p_v_p : p(vik=1|h) for a given movie(i) and the concerned user
   return 1: rate estimation (expectation over p(vik=1|h)) for movie i for the concerned user"""
def compute_rate_esp(p_v_p):
    sum_ = 0
    for k in range(K):
        sum_ = sum_ + (k+1)*p_v_p[k]
    return sum_


"""brief : estimates missing ratings of a user according to the computed model
   param w: weight matrix
   param b_v : bias matrix
   param b_h : bias matrix
   ratings_ : given dataset
   required_rates : matrix of the size of ratings_ with 1 where we want an estimation
   F : number of hidden units
   return : ratings completed with the estimated rates"""
def predict(w, b_h, b_v, ratings_, user_id, required_rates, F):
    new_ratings = np.copy(ratings_)
    new_ratings = new_ratings.astype(np.float)
    m, movie_index, V = compute_user_rate_matrix(new_ratings, user_id)
    missing_movie_index = np.where(new_ratings[:, user_id] ==0)[0]
    p_h_v = compute_p_h_v(w, b_h, V, m, movie_index, F)
    for i in range(required_rates.shape[0]):
        p_v_p = compute_p_v_p(w, b_v, p_h_v, required_rates[i])
        rate = compute_rate_esp(p_v_p)
        new_ratings[required_rates[i], user_id] = rate
    return new_ratings




def calculate_rmse_from_matrix(M, prediction):
    """
        calculate the RMSE using the original matrix and the prediction matrix
    """
    ones = M>0
    rmse = 0
    N = np.sum(ones)
    rmse = np.sum(np.square((M-prediction)*ones))
    return np.sqrt(rmse/N)


"""brief : estimates missing ratings of the dataset where necessary and compares with the real values
   param w: weight matrix
   param b_v : bias matrix
   param b_h : bias matrix
   ratings_test : matrix of shape of the dataset with non_zeros where we want a prediion
   ratings_train : matrix with knwon ratings
   F : number of hidden units
   return 1: ratings completed with the estimated rates
   return 2: rmse of prediction compared to real values"""
def compute_score(w, b_h, b_v, ratings_test, ratings_train, F):
    nnz  = ratings_test>0
    new_ratings = np.copy(ratings_train)
    new_ratings = new_ratings.astype(np.float)
    for user_id in range(ratings_train.shape[1]):
        if(np.mod(user_id, 200)==0):
            print("Processing user n° {}".format(int(user_id)))
        required_rates = np.where(ratings_test[:, user_id]>0)[0]
        new_ratings = predict(w, b_h, b_v, new_ratings, user_id, required_rates, F)
    return new_ratings, calculate_rmse_from_matrix(ratings_test,new_ratings)


