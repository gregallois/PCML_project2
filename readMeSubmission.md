Project Recommender System
PCML project 2
Marc Chachuat, Grégoire Gallois-Montbrun, Antoine Sauvage




	1°/ Global description of the folder


As we used three different strategies of recommandation, you will find in this folder three run files. 
Each of them corresponds to a strategy, and generates a submission, using this strategy. 

- run_sgd_matrix_factorization.py 
- TO COMPLETE
- TO COMPLETE 

Important : the run file corresponding to our final submission is : TO COMPLETE

In addition to this run file, the folder contains helpers files. 
More precisely, it contains the helpers provided by the teaching team (that are used by both run files): 

- helpers.py
- plots.py

And the helpers associated to each strategy, and used by the corresponding run file : 

- helpers_sgd_matrix_facto.py
- TO COMPLETE
- TO COMPLETE

Finally, the data are located in the sub-folder : "datasets" where the submission csv files are also generated.




	2°/ Detailed description of the different files 


        a) run_sgd_matrix_factorization.py

    In this file, we use matrix factorization to predict the missing values of the ratings matrix. 
And to compute the matrix factorization of this matrix, we use stochastic gradient descent, with respect to the regularized MSE over a train set. 
    This method depends on a lot of parameters : 
- max_epochs : number of full passes of the stochastic gradient descent
- gamma : step of the gradient descent
- lambda_user, lambda_item : MSE regularization parameters
- K : number of features per item/user

    The strategy developed by this script is to find by grid search the best parameters (regularization, K) for the matrix factorization, with respect to a low number of SGD full passes  (30), and then compute the matrix factorization for a larger number of full passes (hoping that the best parameters for 30 full passes are close from the best parameters for a larger number of iterations). 
    More precisely, this script takes place when the range of research for lambdas and K has already been restricted, through a first rough grid search. Gamma has already been chosen too, low enough to guarantees convergence and big enough to escape the local minima and guarantee correct running time.  
    To define by grid search which parameters are the best, we perform a SGD matrix factorization with respect to each tuple of parameters of our grid, and over the training set, and measure the final rmse over the test set. We select the parameters associated to the lowest rmse over the test set (assuming that the rmse over the test set will approximate the submission rmse).
    Note that during the grid search, we restrain our dataset to significant data, that is, users and items for which at least 10 ratings where available. This trick, suggested in td10, aims to avoid a bias in the selection of parameters, due to meaningless data.
    
    To sum up, the script can be decomposed in parts : 
- part one : load the data
- part two : grid search and selection of the parameters
- part three : final computation for a larger number of full passes.

This script uses methods located in plots.py, helpers.py and helpers_sgd_matrix_facto.py


        b) helpers_sgd_matrix_facto.py

    This file gathers the helper functions needed to compute the matrix factorization through stochastic gradient descent. 
The main function is matrix_factorization_SGD, whose implementation follows the one suggested by the td 10. The function called by matrix_factorization_SGD follow the suggestions of the TD10 as well. 
