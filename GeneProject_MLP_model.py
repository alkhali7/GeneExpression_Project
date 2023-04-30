#!/usr/bin/env python
# coding: utf-8

# # MLP
# 
# The goal for this notebook is to build some DL models

# In[4]:
# you would need to install those libraries below in order for the codes to run

# get_ipython().system('pip install torch_geometric')
# get_ipython().system('pip install packaging')
# get_ipython().system('pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html')
# get_ipython().system('pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html')
# get_ipython().system('pip install -r /Users/shamsalkhalidy/Downloads/pydance/requirements.txt')
# get_ipython().system('pip install -e /Users/shamsalkhalidy/Downloads/pydance')


# # In[5]:


# get_ipython().system('pip install mudata')


# # In[6]:


# get_ipython().system('git clone https://github.com/OmicsML/dance.git pydance')


# In[7]:


import torch
import torch.nn as nn
import anndata as ad
from sklearn.metrics import mean_squared_error


# In[8]:


import torch.optim as optim
# import torch.trainloader
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameters for our network
input_size = 13953
hidden_sizes = [512, 256, 256]
output_size = 134

# Build a feed-forward network
MLP = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[2], output_size))
print(MLP)


# In[9]:


Y_train = ad.read_h5ad("Adt_processed_training.h5ad") #gex is gene expression which are RNA
Y_test = ad.read_h5ad("Adt_processed_testing.h5ad") #gex is gene expression which are RNA
X_train = ad.read_h5ad("Gex_processed_training.h5ad") # adt is protein
X_test = ad.read_h5ad("Gex_processed_testing.h5ad") # adt is protein


# In[11]:


X_train.X.toarray().shape


# In[12]:


import numpy as np
idx = np.random.permutation(X_train.shape[0])
valid_ratio = 0.15
train_idx = idx[:-int(X_train.shape[0]*valid_ratio)]
val_idx = idx[-int(X_train.shape[0]*valid_ratio):]
X_train_tensor = torch.Tensor(X_train.X[train_idx].toarray())
Y_train_tensor = torch.Tensor(Y_train.X[train_idx].toarray())
X_valid_tensor = torch.Tensor(X_train.X[val_idx].toarray())
Y_valid_tensor = torch.Tensor(Y_train.X[val_idx].toarray())
X_test_tensor  = torch.Tensor(X_test.X.toarray())
Y_test_tensor  = torch.Tensor(Y_test.X.toarray())

idx = np.random.permutation(X_train.shape[0])
mydata_train = TensorDataset(X_train_tensor, Y_train_tensor)
mydata_val = TensorDataset(X_valid_tensor, Y_valid_tensor)
mydata_test  = TensorDataset(X_test_tensor, Y_test_tensor)

trainloader = torch.utils.data.DataLoader(mydata_train, batch_size = 64, shuffle=True)
validloader = torch.utils.data.DataLoader(mydata_val, batch_size = 64, shuffle=False)
testloader = torch.utils.data.DataLoader(mydata_test, batch_size = 64, shuffle=True)


# In[13]:


from tqdm import tqdm #tqdm derives from the Arabic word taqaddum (تقدّم)
import math
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


# In[14]:


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
          


# In[15]:


pred_Y = MLP(torch.Tensor(X_test.X.toarray()))
    # Calculate RMSE


# In[19]:


rmse = mean_squared_error(Y_test.X.toarray(), pred_Y.detach().numpy(), squared = False)
    # Print results
print('MLP had a RMSE of ', rmse)
type(pred_Y)


# In[32]:


X_train.obs['cell_type']


# In[33]:


X_test.obs['cell_type']


# In[25]:


cell_type = X_test.obs['cell_type']


cell_type


# In[ ]:


tmp_X.shape


# In[31]:


result = np.zeros(44)
for cell in cell_type:
    id = X_test.obs['cell_type'] == cell
    tmp_X = X_test.X.toarray()[id, :]
    pred_Y = MLP(torch.Tensor(tmp_X))
    rmse = mean_squared_error(Y_test.X.toarray()[id, :], pred_Y.detach().numpy(), squared = False)
    print(cell, ' ', rmse)
    np.append(result, rmse)
    


# In[30]:


result


# ### RMSE Metric
# 
# The metric for task 1 is RMSE on the `adata.X` data.

# In[17]:


def calculate_rmse(true_test_mod2, pred_test_mod2):
    return  mean_squared_error(true_test_mod2.X.toarray(), pred_test_mod2.X, squared=False) # .toarray will turn it to numpy array


# In[74]:


plt.scatter(pred_Y.detach().numpy()[12, :], Y_test.X.toarray()[12, :])


# In[130]:


for method in [baseline_PC_regression2, baseline_mean]:
    # Run prediction
    pred_Y = method(X_train, Y_train, X_test)
    # Calculate RMSE
    rmse = calculate_rmse(Y_test, pred_Y)
    # Print results
    print(f'{pred_Y.uns["method"]} had a RMSE of {rmse:.4f}')


# As expected, the linear model does better than the dummy method. Now the challenge is up to you! Can you do better than this baseline?
