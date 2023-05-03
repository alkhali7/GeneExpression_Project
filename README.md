# Gene Expression Project

## Overview

The goal of this project was to predict protein expression levels using RNA gene expression data obtained through mass cytometry. The dataset was in Anndata format and contained 134 protein features and 13,953 gene expression (GEX RNA) features. The main objective was to beat the RMSE of the baseline PCR linear model, which was 0.38.

## Project Files

- GeneProjectPCA+KNN.py contains the PCA and KNeighborsRegressor model
- GeneProject_MLP_model.py contains PCA, Multi-Layer Perceptron's model combined with a KNN model 
- GeneProject_PC_regression.ipynb contains TruncatedSVD (PCA) feature engineering and Linear Regression model
- Test_MLP.sb contains the sbatch file to submit jobs to a cluster from the MLP model (GeneProject_MLP_model.py)
- You can find the necessary data files for this project on the following website: https://www.dropbox.com/sh/dg10o9wmfmd2cpi/AABWGBng2HeU3g14D1dD20Wia?dl=0 
    - Please note that each file contains high MB and may take some time to download.

## Methods

First, the necessary libraries were imported and the input data was loaded using the "anndata" library to read in the data from the ".h5ad" files. Since the data had a lot of features, the High Performance Computing Center (HPCC) had to be used considering the time it would take to execute python files.

Next, machine learning models were used to predict protein expression levels from gene expression data. Different models such as MLP, Decision Tree model, and KNN were tested, and KNN had the lowest RMSE. The performance of each model was evaluated by varying the number of principal components (n_components) for PCA, the number of neighbors (n_neighbors) for KNN, and the maximum depth and minimum samples per leaf (max_depth and min_samples_leaf) for decision trees.

## Results

Training a linear regression model on the PCA-transformed data with n_components set to 50, results in an RMSE of 0.3813 on the test set. When applying normalization to the data to improve performance, not all models would perform better after, like in our case the PCA-only model. This improved normalized PCR achieved an RMSE of 0.3717. Applying PCA alone to the dataset with n_components set to 10 resulted in an RMSE of 0.41 on the test set. However, this was expected since PCA alone with linear regression can be improved on as stated later.

MLP with PCA and KNN achieved the best performance with an RMSE of 0.346. 

The seconde best model was the PCA_KNN model, which combined PCA with KNN. It had an RMSE of 0.35462 when KNN was applied without normalization to the PCA-transformed data with n_components set to 70 and n_neighbors set to 50. 

In conclusion, our experiments suggest that the MLP model is about 1% better than just TruncatedSVD and KNN alone. Both models achieved a close RMSE, meaning that further analysis can be done to test both models and compare them.
While decision trees with PCA achieved the worst performance with an RMSE of 0.3946068.


## Conclusion

In conclusion, combining PCA with KNN and MLP resulted in the best models for predicting protein expression levels using RNA gene expression data. 
This project have demonstrated the use of machine learning techniques to predict protein expression levels from gene expression data obtained through mass cytometry. PCA and KNN, and MLP can be effective methods for reducing the dimensionality of the data and improving prediction accuracy. The RMSE obtained from the models was 0.354 from KNN-PCA and 0.346 from PCA-MLP-KNN, which is lower than the target RMSE of 0.38. These results show promise for the use of machine learning in the analysis of mass cytometry data and may lead to further advancements in the field of molecular biology.
