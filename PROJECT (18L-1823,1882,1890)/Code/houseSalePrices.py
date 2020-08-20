#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed May  8 02:28:50 2019



"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

import pandas

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPRegressor


from sklearn.ensemble import RandomForestRegressor

from plotly import graph_objects as go

from yellowbrick.regressor import PredictionError


import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

accuracies = []


# Returns the first 5 rows and the respective 81 columns

train.head()

# Checks for missing values and returns the sum for each column

NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])

NAs[NAs.sum(axis=1) > 0]

train.head()

# Gets the R2 score and Root mean square error

def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
    
    
    

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)
    
train_labels = train.pop('SalePrice')

features = pd.concat([train, test], keys=['train', 'test'])




# I decided to get rid of features that have more than half of missing information or do not correlate to SalePrice
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)
    
train.head()
# MSSubClass as str
features['MSSubClass'] = features['MSSubClass'].astype(str)

# MSZoning NA in pred. filling with most popular values
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

# LotFrontage  NA in all. I suppose NA means 0
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())

# Alley  NA in all. NA means no access
features['Alley'] = features['Alley'].fillna('NOACCESS')

# Converting OverallCond to str
features.OverallCond = features.OverallCond.astype(str)

# MasVnrType NA in all. filling with most popular values
features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBSMT')

# TotalBsmtSF  NA in pred. I suppose NA means 0
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

# KitchenAbvGr to categorical
features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)

# KitchenQual NA in pred. filling with most popular values
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

# FireplaceQu  NA in all. NA means No Fireplace
features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')

# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    features[col] = features[col].fillna('NoGRG')

# GarageCars  NA in pred. I suppose NA means 0
features['GarageCars'] = features['GarageCars'].fillna(0.0)

# SaleType NA in pred. filling with most popular values
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

# Year and Month to categorical
features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)


# Our SalesPrice is skewed right (check plot below). I'm logtransforming it. 
ax = sns.distplot(train_labels)
    

train_labels = np.log(train_labels)



ax = sns.distplot(train_labels)


numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]

numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()

print(numeric_features_standardized.shape)

ax = sns.pairplot(numeric_features_standardized)



# Getting Dummies from Condition1 and Condition2
conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.ix[i, cond] = 1
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

# Getting Dummies from Exterior1st and Exterior2nd
exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])

dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)

for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.ix[i, ext] = 1
    
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)

features.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)

# Getting Dummies from all other categorical vars
for col in features.dtypes[features.dtypes == 'object'].index:
    
    for_dummy = features.pop(col)
    
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)

features_standardized = features.copy()

### Replacing numeric features by standardized values
features_standardized.update(numeric_features_standardized)

    
    
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

print(train_features[0])
### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)


x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)




x_train_st = np.asfarray(x_train_st)

x_test_st = np.asfarray(x_test_st)

y_train_st = np.asfarray(y_train_st)

y_test_st = np.asfarray(y_test_st)


print(x_train_st.shape)

print(x_test_st.shape)

def get_normal_regressor():
    
        
       
    trans_matrix = x_train_st.transpose()
    se = np.matmul(trans_matrix , x_train_st)
    te = np.linalg.pinv(se)
    de = np.matmul(trans_matrix, y_train_st)
    weighted_matrix = np.matmul(te,de)
    
    weighted_matrix = weighted_matrix.reshape(262,)
    predicted = []
    
    total_found_error = 0
    count = 0
    
    for i in range(146):
            sample = x_test_st[i]
            final_result = np.dot(weighted_matrix, sample)
            predicted.append(final_result)
          #  print('Val: ',final_result,'Actual: ' ,y_train_st[i])
            error = abs(y_test_st[i] - final_result)
            total_found_error = total_found_error + (error*error)
            count = count + 1
                
       
       
    rms = np.sqrt(total_found_error/count)
    
    print(rms)
    
    
    
    accuracies.append(100 - rms)
    
    raw_plot(y_test_st,predicted)
    
    return weighted_matrix


    
def get_gradient_regressor():
    
        

    weighted_matrix = np.full((262,), 0.1, dtype= np.float64)
         
    
    
    learning_rate = 0.0000000001
    
    total_val = np.zeros((262,), dtype= np.float64)
    
    

    
    
    for j in range (500):
        
        final_error = np.zeros((262,), dtype= np.float64)
        
        for s in range(1314):
            error = 0.00000000000000
            inter_result = np.dot(weighted_matrix, x_train_st[s])
            expec_result = y_train_st[s]
            error = expec_result - inter_result
            final_error = final_error + np.multiply(error, x_train_st[s])
            
        total_val = np.multiply(learning_rate, final_error)
        weighted_matrix = weighted_matrix + total_val
        
        
        
    total_found_error = 0
    
    predicted = []
    
    count = 0
    
    for i in range(146):
        sample = x_test_st[i]
        final_result = np.dot(weighted_matrix, sample)
        predicted.append(final_result)
     #   print('Val: ',final_result,'Actual: ' ,y_train_st[i])
        error = abs(y_test_st[i] - final_result)
        total_found_error = total_found_error + (error*error)
        count = count + 1
        
    rms = np.sqrt(total_found_error/count)
    print(rms)
   # print('Accuracy: ', 100 - rms)
    accuracies.append(100 - rms)
    
    raw_plot(y_test_st,predicted)
    
    return weighted_matrix

 

def get_stochastic_gradient_regressor():

    
    
    weight_matrix = np.full((262,), 0.1, dtype=float)
    
    
    learning_rate = 0.0000001
    
    
    for j in range (500):
        
        for s in range(1314):
            
            error = 0.000000
            
            inter_result = np.dot( weight_matrix, x_train_st[s])
                
            expec_result = y_train_st[s]
            error = expec_result - inter_result
            error = error * learning_rate
            final_matrix = np.multiply(error, x_train_st[s])
            weight_matrix = weight_matrix + (final_matrix)
        
    
    total_found_error = 0
    count = 0
    
    predicted = []
    
    for i in range(146):
        sample = x_test_st[i]
        final_result = np.dot( weight_matrix, sample)
        predicted.append(final_result)
       # print('Val: ',final_result,'Actual: ' ,y_train_st[i])
        error = abs(y_test_st[i] - final_result)
        total_found_error = total_found_error + (error*error)
        count = count + 1
        
    rms = np.sqrt(total_found_error/count)
    print(rms)
  #  print('Accuracy: ', 100 - rms)
    accuracies.append(100 - rms)
    
    raw_plot(y_test_st,predicted)
                
      
    return  weight_matrix
    




def get_random_forest_regressor():
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    # Train the model on training data
    rf.fit(x_train_st, y_train_st);
    predictions = rf.predict(x_test_st)
    # Calculate the absolute errors
    errors = abs(predictions - y_test_st)
    count = 0
    total_error = 0
    
    for error in errors:
        total_error = total_error + (error * error)
        count = count + 1
        
    rms = np.sqrt(total_error/count)
    
    accuracies.append(100 - rms)
    
    print(rms)
    
    plot_boundary(rf)
    
    # Print out the mean absolute error (mae)
    
  #  print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

def get_ensembled_regressor():
    
    mat1 =  get_normal_regressor()
    mat2 =  get_gradient_regressor()
    mat3 =  get_stochastic_gradient_regressor()
    
    total_found_error = 0
    
    count = 0
    
    predicted = []
    
    for i in range(146):
        
        sample = x_test_st[i]
        
        final_result = float((float(np.dot( mat1, sample)) +  float(np.dot( mat2, sample)) + float(np.dot( mat3, sample)))/3)
        predicted.append(final_result)
       # print('Val: ',final_result,'Actual: ' ,y_train_st[i])
        error = abs(y_test_st[i] - final_result)
        
        total_found_error = total_found_error + (error*error)
        count = count + 1
        
    rms = np.sqrt(total_found_error/count)
    
    print(rms)
    
    raw_plot(y_test_st,predicted)
    
    accuracies.append(100 - rms)
    



def get_ann():
    
    ann_regressor = MLPRegressor(hidden_layer_sizes=(15,10,5,2), random_state = 1,learning_rate_init= 0.01)
    ann_regressor.fit(x_train_st, y_train_st)
    predictions = ann_regressor.predict(x_test_st)
    
    errors = abs(predictions - y_test_st)
    
    count = 0
    
    total_error = 0
    
    for error in errors:
        total_error = total_error + (error * error)
        count = count + 1
    
    rms = np.sqrt(total_error/count)
    
    print(rms)
  #  print('Accuracy: ', 100 - rms)
    accuracies.append(100 - rms)
    
    plot_boundary(ann_regressor)
    
    
def get_kernel_ridge_regressor():
    
    
    kernel_ridge_regressor = KernelRidge(alpha=1.0)
    
    kernel_ridge_regressor.fit(x_train_st, y_train_st) 
    
    predictions = kernel_ridge_regressor.predict(x_test_st)
    
    errors = abs(predictions - y_test_st)
    
    count = 0
    
    total_error = 0
    
    for error in errors:
        total_error = total_error + (error * error)
        count = count + 1
    
    rms = np.sqrt(total_error/count)
    
    print(rms)
   # print('Accuracy: ', 100 - rms)
    accuracies.append(100 - rms)
    
    plot_boundary(kernel_ridge_regressor)



def plot_boundary(mod_passed):
    
    model = mod_passed
  
    visualizer = PredictionError(model)

    visualizer.fit(x_train_st, y_train_st)  # Fit the training data to the visualizer
    
    visualizer.score(x_test_st, y_test_st)  # Evaluate the model on the test data
    
    visualizer.show()                 # Finalize and render the figure
    
def raw_plot(y, predicted):
    
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('y')
    ax.set_ylabel('y-predicted')
    plt.show()
    
     
get_normal_regressor()   
  
get_gradient_regressor()

get_stochastic_gradient_regressor()

get_ensembled_regressor()

get_random_forest_regressor()
    
get_ann()

get_kernel_ridge_regressor()

#fig = go.Figure(go.Funnel(
  #  y = ["Normal", "Batch Gradient", "Stochastic Gradient", "ANN", "Kernel Ridge","Random Forest"],
  #  x = accuracies))    

#fig.show()

#accuracies.append(100)

#print(accuracies)
  


























