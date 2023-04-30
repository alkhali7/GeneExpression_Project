# Gene Expression Project

## Overview

The goal of this project was to predict protein expression levels using RNA gene expression data obtained through mass cytometry. The dataset was in Anndata format and contained 134 protein features and 13,953 gene expression (GEX RNA) features. The main objective was to beat the RMSE of the baseline PCR linear model, which was 0.38.

## Project Files

- GeneProjectPCA+KNN.py contains the PCA and KNeighborsRegressor model
- GeneProject_MLP_model.py contains Multi-Layer Perceptron's model
- GeneProject_PC_regression.ipynb contains TruncatedSVD (PCA) feature engineering and Linear Regression model
- Test_MLP.sb contains the sbatch file to submit jobs to a cluster from the MLP model (GeneProject_MLP_model.py)

## Methods

First, the necessary libraries were imported and the input data was loaded using the "anndata" library to read in the data from the ".h5ad" files. Since the data had a lot of features, the High Performance Computing Center (HPCC) had to be used considering the time it would take to execute python files.

Next, machine learning models were used to predict protein expression levels from gene expression data. Different models such as MLP, Decision Tree model, and KNN were tested, and KNN had the lowest RMSE. The performance of each model was evaluated by varying the number of principal components (n_components) for PCA, the number of neighbors (n_neighbors) for KNN, and the maximum depth and minimum samples per leaf (max_depth and min_samples_leaf) for decision trees.

## Results

Training a linear regression model on the PCA-transformed data with n_components set to 50, results in an RMSE of 0.3813 on the test set. When applying normalization to the data to improve performance, not all models would perform better after, like in our case the PCA-only model. This improved normalized PCR achieved an RMSE of 0.3717. Applying PCA alone to the dataset with n_components set to 10 resulted in an RMSE of 0.41 on the test set. However, this was expected since PCA alone with linear regression can be improved on as stated later.

The best model was the PCA_KNN model, which combined PCA with KNN. It had an RMSE of 0.35462 when KNN was applied without normalization to the PCA-transformed data with n_components set to 70 and n_neighbors set to 50.

## Conclusion

In conclusion, combining PCA with KNN resulted in the best model for predicting protein expression levels using RNA gene expression data. The project shows how machine learning can be applied to molecular biology data to make predictions and gain insights.
