import torch
import torch.nn as nn
import torch.optim as optim
# import torch.trainloader
from torch.utils.data import TensorDataset, DataLoader
import logging
import anndata as ad
import numpy as np
 

from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
logging.basicConfig(level=logging.INFO)
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor #used KNN
from tqdm import tqdm
import math
import pickle



print("reading data from files")
Y_train = ad.read_h5ad("Adt_processed_training.h5ad") #gex is gene expression which are RNA
Y_test = ad.read_h5ad("Adt_processed_testing.h5ad") #gex is gene expression which are RNA
X_train = ad.read_h5ad("Gex_processed_training.h5ad") # adt is protein
X_test = ad.read_h5ad("Gex_processed_testing.h5ad") # adt is protein

 

# Perform PCA on training data
print("Preform PCA on training data")
pca = TruncatedSVD(n_components=200)
X_train_pca = pca.fit_transform(X_train.X.toarray())

 

# Transform test data with trained PCA
X_test_pca = pca.transform(X_test.X.toarray())


X_train = X_train_pca
X_test = X_test_pca
 


print("Get data ready for MLP")

idx = np.random.permutation(X_train.shape[0])
valid_ratio = 0.15
train_idx = idx[:-int(X_train.shape[0]*valid_ratio)]
val_idx = idx[-int(X_train.shape[0]*valid_ratio):]
X_train_tensor = torch.Tensor(X_train[train_idx])
Y_train_tensor = torch.Tensor(Y_train.X[train_idx].toarray())
X_valid_tensor = torch.Tensor(X_train[val_idx])
Y_valid_tensor = torch.Tensor(Y_train.X[val_idx].toarray())
X_test_tensor  = torch.Tensor(X_test)
Y_test_tensor  = torch.Tensor(Y_test.X.toarray())

idx = np.random.permutation(X_train.shape[0])
mydata_train = TensorDataset(X_train_tensor, Y_train_tensor)
mydata_val = TensorDataset(X_valid_tensor, Y_valid_tensor)
mydata_test  = TensorDataset(X_test_tensor, Y_test_tensor)

trainloader = torch.utils.data.DataLoader(mydata_train, batch_size = 64, shuffle=True)
validloader = torch.utils.data.DataLoader(mydata_val, batch_size = 64, shuffle=False)
testloader = torch.utils.data.DataLoader(mydata_test, batch_size = 64, shuffle=True)


print(X_train.shape)

input_size = X_train.shape[1]
hidden_sizes = [512, 256, 128]
output_size = 134
MLP = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
   
    nn.Linear(hidden_sizes[2], output_size)
)

print(MLP)



def evaluate(model):
    running_loss = 0
    with torch.no_grad():
        for data, target in validloader:
           
            # Forward pass
            output = MLP(data)
            loss = criterion(output, target)
            running_loss += math.sqrt(loss.item())  # why need to add?
    return running_loss/len(validloader)

# Define the loss function and optimizer
criterion = nn.MSELoss() # Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.Adam(MLP.parameters())



print("Before")

num_epochs = 20 # 15
for epoch in (pbar := tqdm(range(num_epochs))):
    running_loss = 0
    for data, target in trainloader:        
        # Forward pass
        output = MLP(data) #<--- note this line is using the model you set up at the beginning of this section
        loss = criterion(output, target)
        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        running_loss += math.sqrt(loss.item())
        optimizer.step()
        # Print the loss every 5 epochs
    if (epoch+1) % 5 == 0:
        pbar.set_description(f'Epoch {epoch} | Train loss: {running_loss/len(trainloader):.4f} | Valid loss: {evaluate(MLP):.4f}')
         
print("done with mlp training")


pred_Y = MLP(torch.Tensor(X_test))
gpu = pred_Y.detach().numpy()





# KNN model
def baseline_linear(input_train_gex, input_train_adt, input_test_gex):
    '''Baseline method training a KNN regressor on the input data'''
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
    reg =  KNeighborsRegressor(n_neighbors=200)
    

    # Train the model on the PCA reduced gex 1 and 2 data
    reg.fit(X_train, y_train)
    
    # Save the trained KNN model to a file
    with open('mlp_knn_pca.pickle', 'wb') as f:
        pickle.dump(reg, f)

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



# Get Data ready for linear
input_train_gex  = ad.read_h5ad("Gex_processed_training.h5ad") #GEX training
input_train_adt = ad.read_h5ad("Adt_processed_training.h5ad") # Adt training
input_test_gex = ad.read_h5ad("Gex_processed_testing.h5ad") #GEX test
true_test_adt = ad.read_h5ad("Adt_processed_testing.h5ad")  #Adt test

# run linear
pred_test_adt = baseline_linear(input_train_gex, input_train_adt, input_test_gex)


# averaging the mlp and knn 
pcamlpknn = np.mean( np.array([ gpu, pred_test_adt.X ]), axis=0 )




rmse = mean_squared_error(Y_test.X.toarray(), pcamlpknn, squared = False)
    # Print results
print('MLP had a RMSE of ', rmse) #MLP had a RMSE of  0.34448
type(pred_Y)




