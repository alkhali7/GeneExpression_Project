import logging
import anndata as ad
import numpy as np
import pickle
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
logging.basicConfig(level=logging.INFO)
from sklearn.decomposition import PCA 
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor #used KNN


input_train_gex  = ad.read_h5ad("Gex_processed_training.h5ad") #GEX training
input_train_adt = ad.read_h5ad("Adt_processed_training.h5ad") # Adt training
input_test_gex = ad.read_h5ad("Gex_processed_testing.h5ad") #GEX test
true_test_adt = ad.read_h5ad("Adt_processed_testing.h5ad")  #Adt test


#function to calculate rmse 
def calculate_rmse(true_test_adt, pred_test_adt):
    return  mean_squared_error(true_test_adt.X.toarray(), pred_test_adt.X, squared = False)
    
# function using PCA and KNN 
def baseline_PCKNN(input_train_gex, input_train_adt, input_test_gex):
    '''Baseline method training a linear regressor on the input data'''
    input_gex = ad.concat(
        {"train": input_train_gex, "test": input_test_gex},
        axis = 0,
        join = "outer",
        label = "group",
        fill_value = 0,
        index_unique = "-", 
    )
    
    # Do PCA on the input data
    logging.info('Performing dimensionality reduction on GEX values...')
    embedder_gex = TruncatedSVD(n_components = 70)
    gex_pca = embedder_gex.fit_transform(input_gex.X)
    
    
    # split dimension reduction GEX back up for training
    X_train = gex_pca[input_gex.obs['group'] == 'train']
    X_test = gex_pca[input_gex.obs['group'] == 'test']
    y_train = input_train_adt.X.toarray()
    
    assert len(X_train) + len(X_test) == len(gex_pca)
    
    logging.info('Running KNN regression...')
    
    reg =  KNeighborsRegressor(n_neighbors=50)
    
    # Train the model on the PCA reduced gex 1 and 2 data
    reg.fit(X_train, y_train)
     # Save the model to a file
    # with open('Final_Project_trainedModel.pkl', 'wb') as file:
    #     pickle.dump(reg, file)
    # # predict y (protein) based on the model 
    y_pred = reg.predict(X_test)
    
    # Project the predictions back to the adt feature space
    
    pred_test_adt = ad.AnnData(
        X = y_pred,
        obs = input_test_gex.obs,
        var = input_train_adt.var,
    
    )
    
    # Add the name of the method to the result
    pred_test_adt.uns["method"] = "KNN"
    
    return pred_test_adt

# Run prediction
pred_test_adt = baseline_PCKNN(input_train_gex, input_train_adt, input_test_gex)
# Calculate RMSE
rmse = calculate_rmse(true_test_adt, pred_test_adt)
print("rmse for using PCA and KNN with out normalization of data", rmse)  #0.354899





############## PCA and KNN with normalization , when doing the normalization, the eigenvector of our dataset increased, and we want a stable 
#eigenvalue 
# def baseline_PC_regression2(X_train, Y_train, X_test):
#     '''Baseline method training a linear regressor on the input data'''
#     X_tr = X_train.X.toarray()
#     X_bar = np.mean(X_tr, axis = 0)
#     X_sd = np.std(X_tr, axis=0)
#     X_te = X_test.X.toarray()
#     X_tr1 = (X_tr - X_bar) / X_sd # center and normalized to have variance 1
#     X_te1 = (X_te - X_bar) / X_sd # Normalized testing data
    
    
#     # Do PCA on the X_train data, SVD
#     logging.info('Performing dimensionality reduction on X_train RNA values...')
#     svd = TruncatedSVD(n_components = 70)
#     svd.fit(X_tr1)
#     singular_vectors = svd.components_
#     PC_train = np.dot(X_tr1, singular_vectors.T)
#     PC_test = np.dot(X_te1, singular_vectors.T)
        
#     y_train = Y_train.X.toarray()
    
    
#     logging.info('Running Linear regression...')
    
#     reg = KNeighborsRegressor(n_neighbors=60)
    
#     # Train the model on the PCA reduced modality 1 and 2 data
#     reg.fit(PC_train, y_train)
#     y_pred = reg.predict(PC_test)
    
#     # Project the predictions back to the modality 2 feature space
    
#     pred_Y = ad.AnnData(
#         X = y_pred,
#         obs = X_test.obs,
#         var = Y_train.var
#     )
    
#     # Add the name of the method to the result
#     pred_Y.uns["method"] = "linear"
    
#     return pred_Y

# # Run prediction with normalisation
# pred_test_adt2 = baseline_PC_regression2(input_train_gex, input_train_adt, input_test_gex)
# # Calculate RMSE
# rmse2 = calculate_rmse(true_test_adt, pred_test_adt2)
# print("rmse for using PCA and KNN with normalization ", rmse2)  #0.36603
