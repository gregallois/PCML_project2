def update_user_feature_kr(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices, n_train, verbose = False):
    """update user feature matrix."""
    if verbose: timer_feat = time.time()
    num_user   = train.shape[1]
    num_feature = item_features.shape[0]
    user_features = np.zeros((num_feature,num_user))
    nnz = n_train

    if verbose: timer_feat = timestep(timer_feat,"    Preparation of the update user: ")
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
            #c[i] = b[i,0]
            c[i] = b[i]
        
        user_features[:,n] = np.linalg.solve(A,c)
    if verbose: timer_feat = timestep(timer_feat,"    Loops for update user: ")
    return user_features
    

def update_item_feature_kr(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices, n_train):
    """update item feature matrix."""
    
    num_item   = train.shape[0]
    num_feature = user_features.shape[0]
    item_features = np.zeros((num_feature,num_item))
    nnz = n_train
    #train = train.todense()
    
    for d in range(len(nnz_users_per_item)):
        
        #lambd = lambda_item * nnz_users_per_item[d]
        lambd = lambda_item * nnz
        X = train[d,nz_item_userindices[d]].transpose()
        Z = user_features[:,nz_item_userindices[d]]
        gram = np.dot(Z,np.transpose(Z))
        A = gram + lambd * np.identity(num_feature)
        b = np.dot(Z,X)
        
        c= np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            #c[i] = b[i,0]
            c[i] = b[i]
        
        item_features[:,d] = np.linalg.solve(A,c)
        
        
    return item_features


def kr(train, test,K, lambda_user, lambda_item, stop_criterion, it_max, verbose = False):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = K   # K in the lecture notes
    #lambda_user = 1e-8
    #lambda_item = 1e-8
    #stop_criterion = 1e-4
    change = 1
    error_list = [float("inf")]
    #it_max = 20
    timer = time.time()
    # set seed
    np.random.seed(988)
    
    # init ALS
    user_features, item_features = init_MF(train, num_features)
    if verbose:timer = timestep(timer,"Preparation des données, init MF : ")
    # ALS algorithm
    # counters and variables 
    it = 0 
    
    num_item, num_user = train.shape
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    nz_test, niut, nuit = build_index_groups(test)
    
    for i in range(len(nz_item_userindices)):
        nz_item_userindices[i] = nz_item_userindices[i][1]
    for j in range(len(nz_user_itemindices)):
        nz_user_itemindices[j] = nz_user_itemindices[j][1]
        
    nnz_items_per_user = np.zeros(len(nz_user_itemindices))
    
    for i in range(len(nz_user_itemindices)):
        nnz_items_per_user[i] = len(nz_user_itemindices[i])
        
    nnz_users_per_item = np.zeros(len(nz_item_userindices))

    
    for j in range(len(nz_item_userindices)):
        nnz_users_per_item[j] = len(nz_item_userindices[j])
    if verbose:timer = timestep(timer,"Preparation des données, regroupement des features : ")

    
    n_train = len(nz_train)
    n_test = len(nz_test)
    # modification of the initialisation : 
    # assigning the average rating for the movies as the first row.
    im = train.sum(axis =1)
    #for i in range(num_item):
    for i in range(len(nz_item_userindices)):
        #item_features[0,i] = 0
        #im[i] = 0
        #item_features[0,i]=im[i,0]/len(nz_item_userindices[i])
        item_features[0,i]=im[i]/len(nz_item_userindices[i])
        for k in range(1,num_features):
            item_features[k,i] = item_features[k,i]*np.random.random()/n_train 
    
    #print(item_features)
    if verbose:timer = timestep(timer,"Preparation des données, final step : ")
    while ((change > stop_criterion) &(it<it_max) ): 
        it = it+1 
        #print("Iteration of the alternative least square : ", it)
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices, n_train, verbose)
        if verbose: timer = timestep(timer, "User features update: ")
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices, n_train)
        if verbose: timer = timestep(timer, "Item features update: ")
        error = compute_error(train, user_features, item_features, n_train, verbose)
        #print( "error commited during this iteration : ", error)
        error_list.append(error)
        change = error_list[-2]-error_list[-1]
        if verbose: timer = timestep(timer, "Computation of the indicators: ")
        
    rmse_test = compute_error(test, user_features, item_features, n_train)
    if verbose: timer = timestep(timer, "Final time: ")
    return user_features, item_features, rmse_test