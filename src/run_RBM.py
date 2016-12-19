import numpy as np
from RBM_helpers import *
from helpers import *



#Loading datasets
print("Please wait while loading dataset....")
path_dataset = "../datasets/data_train.csv"
ratings = load_data(path_dataset)
ratings = ratings.toarray()
ratings = ratings.astype(np.int8)
path_evaluation = "../datasets/sampleSubmission.csv"
evaluated_on = load_data(path_evaluation)
evaluated_on = evaluated_on.toarray()
evaluated_on = evaluated_on.astype(np.int8)
print("Dataset loaded with success !")


# set seed
np.random.seed(988)
    

#Constant definition
step = 0.5 #gradient descent step
max_iter = 5 #number of epochs
F = 10 #number of hidden units
K = 5 #ratings go from 1 to 5


#Initialisation of bias and weights
w = np.asarray(np.random.uniform(
                        low=-4 * np.sqrt(6. / (F + ratings.shape[0])),
                        high=4 * np.sqrt(6. / (F + ratings.shape[0])),
                        size=(ratings.shape[0], F, K)
                    ))

b_h = np.asarray( np.random.uniform(
                        low=-4 * np.sqrt(6. / (F + ratings.shape[0])),
                        high=4 * np.sqrt(6. / (F + ratings.shape[0])),
                        size=(F,1)
                    ))

b_v = np.asarray(np.random.uniform(
                        low=-4 * np.sqrt(6. / (F + ratings.shape[0])),
                        high=4 * np.sqrt(6. / (F + ratings.shape[0])),
                        size=(ratings.shape[0],K)
                    ))



n_users = ratings.shape[1]
grad_ = np.zeros(w.shape)
n_movies = np.sum(ratings>0, axis=0)

#selecting users who have rated more than 150 movies (approximmately 90% of dataset)
goods = n_movies>150
ids = np.where(goods>0)[0]
interesting_users = sum(goods)
        

iter_=0

#number of times we iterate over the dataset
while iter_<max_iter:
    
    #adding some noise to users to avoid overfitting
    #10% chance to have + or -1 for each rating
    rtrain = np.copy(ratings)
    noise = np.random.ranf(rtrain.shape)
    noise = noise*(rtrain>0)
    rtrain = np.maximum(np.minimum(rtrain + (noise>0.9)*(noise<0.95)-(noise>0.95),5*(rtrain>0)),(rtrain>0))

    #variying step size while the number of epochs increases
    if iter_ == 2:
        step = 0.2
    if iter_ == 3:
        step = 0.1
    if iter == 4:
        step = 0.05
    if iter == 4:
        step = 0.01
    
    #users come in a random order
    ids = np.random.permutation(ids)

    for i in range(ids.shape[0]):
        print(i)

        #select random user to train it's RBM
        user_id = ids[i]
                
        #w dropout, 10% of hidden units are off for this training case (avoid overfitting)
        drop_w = np.copy(w)
        drop_bh = np.copy(b_h)
        kept_hidden = np.random.ranf((1,F))<0.9
        drop_w[:, np.where(kept_hidden==0)[1], :] = 0
        drop_bh[np.where(kept_hidden==0)[1], 0] = 0
        
        #training on the selected user
        m, movie_index, rate_matrix = compute_user_rate_matrix(rtrain, user_id)
        delta_w_, delta_h_, delta_bv = update(drop_w, drop_bh, b_v, m, movie_index, rate_matrix, F)
        delta_w_ = delta_w_*(drop_w!=0)
        delta_bh = delta_h_*(drop_bh!=0)
        delta_bv = delta_bv  
        w = w + step * delta_w_
        b_h = b_h + step * delta_bh
        b_v = b_v + step * delta_bv

    iter_ = 1+iter_
        
#making the required predictions 
print("Model trained, making predictions")
prediction_evaluated, error = compute_score(w, b_h, b_v, evaluated_on, ratings, F)
#storing into csv file
print("Predictions done, saving into csv file.")
matrix2file('../datasets/results.csv', prediction_evaluated, evaluated_on)
print("CSV file created, finished")




