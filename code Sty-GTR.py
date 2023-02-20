# A Comprehensive Study on the Styrene–GTR Radical Graft Polymerization: 
# Combination of an Experimental Approach, on Different Scales, with Machine Learning Modeling


#%% 0. Importing of libraries/functions and installation of packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, max_error
from sklearn.model_selection import GridSearchCV, KFold, validation_curve


#%% 1. Choices from the user

scaler = StandardScaler() # standardization method (other scalers: MinMaxScaler(), RobustScaler())
k_out_ML = 5 # k value in k-fold outer loop (train/test split)
random_nb = 221 # random state for k-fold and some ML models
choice_weight_samples = 0 # 1: weights samples with their 1/uncertainty; 0: all samples have same weight
choice_HP = 3 # 1: ML without HPs optimization; 2: Definition of HPs window; 3: ML with HPs optimization

if choice_HP in [2, 3]:
    k_in_ML = 5 # k value in k-fold inner loop used for HPs optimization (train/validation split)

if choice_HP==2: # selection of the ML model and the HP to optimize
    # 'SVR': 'C', 'kernel', 'epsilon', 'gamma'
    # 'RF': 'n_estimators', 'max_features', 'min_samples_split', 'min_samples_leaf', 'max_samples'
    # 'GB': 'learning_rate', 'n_estimators', 'max_features', 'min_samples_split', 'min_samples_leaf', 'subsample'
    # 'MLP': 'activation', 'solver', 'learning_rate', 'learning_rate_init', 'alpha', 'max_iter', 'hidden_layer_sizes'
    choice_mdl = 'SVR'
    choice_mdlHP = 'C'


#%% 2. Data loading

# data loading
data_final = pd.read_excel("data Sty-GTR.xlsx")

# inputs X and output y
X = data_final[['GTR/(GTR+styrene) exp', 'BPO/styrene exp', 'T exp', 't exp']]
y = data_final['Conversion']


#%% 3.1 Machine learning (without HPs optimization)

if choice_HP==1:
    
    print('Screening without HPs optimization:')
    print('-----------------------------------')
    print('')
    
    # k-fold outer loop
    CV_out = KFold(n_splits=k_out_ML, shuffle=True, random_state=random_nb)
    
    # ML Models
    mdl_LR = linear_model.LinearRegression()
    mdl_ridge = linear_model.Ridge(random_state=random_nb)
    mdl_lasso = linear_model.Lasso(random_state=random_nb)
    mdl_SVR = SVR()
    mdl_GPR = GaussianProcessRegressor(random_state=random_nb)
    mdl_kNN = KNeighborsRegressor()
    mdl_DT = DecisionTreeRegressor(random_state=random_nb)
    mdl_RF = RandomForestRegressor(random_state=random_nb)
    mdl_GB = GradientBoostingRegressor(random_state=random_nb)
    mdl_MLP = MLPRegressor(random_state=random_nb)  
    list_mdl = [mdl_LR, mdl_ridge, mdl_lasso, mdl_SVR, mdl_GPR, mdl_kNN, mdl_DT, mdl_RF, mdl_GB, mdl_MLP]
    list_mdl_names = ['Linear Regression', 'Ridge', 'Lasso', 'Support Vector Regression',  'Gaussian Processes', 'k-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Multi-Layer Perceptron']

    # Creation of lists
    list_train_nb = []
    list_train_id = []
    list_test_nb = []
    list_test_id = []
    list_X_train_scaled = []
    list_X_test_scaled = []
    list_y_train = []
    list_y_test = []
    if choice_weight_samples==1:
        list_weight_train = []
        list_weight_test = []
    list_all_R2_train = []
    list_all_RMSE_train = []
    list_all_MeanAE_train = []
    list_all_MedianAE_train = []
    list_all_MaxError_train = []
    list_all_R2_test = []
    list_all_RMSE_test = []
    list_all_MeanAE_test = []
    list_all_MedianAE_test = []
    list_all_MaxError_test = []
    list_all_train_time = []
    list_all_y_train_pred = []
    list_all_y_test_pred = []
    
    k=-1
    for i, j in CV_out.split(X):
        k=k+1
        print('* Split', k+1)
        
        # Saving info on train and test data
        list_train_nb.append(i.shape[0])
        list_train_id.append(data_final.iloc[i,:])
        list_test_nb.append(j.shape[0])
        list_test_id.append(data_final.iloc[j,:])
    
        # Preparation of X_train, X_test, y_train and y_test and scaling of X_train and X_test
        X_train = X.iloc[i,:]
        X_test = X.iloc[j,:]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        list_X_train_scaled.append(X_train_scaled)
        list_X_test_scaled.append(X_test_scaled)
        y_train = data_final.iloc[i,-2]
        y_test = data_final.iloc[j,-2]
        list_y_train.append(y_train)
        list_y_test.append(y_test)
        
        # Preparation of samples weights
        if choice_weight_samples == 1:
            weight_train = data_final.iloc[i,-1]
            weight_test = data_final.iloc[j,-1]
            list_weight_train.append(weight_train)
            list_weight_test.append(weight_test)
             
        # Screening
        train_time_list = []
        list_y_train_pred = []
        list_y_test_pred = []
        R2_train_list = []
        RMSE_train_list = []
        MeanAE_train_list = []
        MedianAE_train_list = []
        MaxError_train_list = []
        R2_test_list = []
        RMSE_test_list = []
        MeanAE_test_list = []
        MedianAE_test_list = []
        MaxError_test_list = []  
        
        # Screening
        p=-1
        for mdl in list_mdl:    
            p = p+1 
            
            t0 = time.time() # starting time
            
            if list_mdl_names[p] in ['Support Vector Regression', 'Random Forest', 'Gradient Boosting']:
                if choice_weight_samples==1:
                    mdl.fit(X_train_scaled,y_train.values.ravel(),sample_weight=weight_train.tolist())
                else:
                    mdl.fit(X_train_scaled,y_train.values.ravel())
            else:
                mdl.fit(X_train_scaled,y_train.values.ravel())
                
            tf = time.time() # ending time
            
            print(list_mdl_names[p], ':', round(tf-t0,3), 's') # training time
            train_time_list.append(tf-t0)
            y_train_pred = mdl.predict(X_train_scaled)
            y_test_pred = mdl.predict(X_test_scaled)               
            list_y_train_pred.append(y_train_pred)
            list_y_test_pred.append(y_test_pred)
            
            # Scores training for one model and one split
            R2_train = r2_score(y_train.values, y_train_pred)
            RMSE_train = mean_squared_error(y_train.values, y_train_pred, squared=False)
            MeanAE_train = mean_absolute_error(y_train.values, y_train_pred)
            MedianAE_train = median_absolute_error(y_train.values, y_train_pred)
            MaxError_train = max_error(y_train.values, y_train_pred)
            
            # Scores training for all models and one split
            R2_train_list.append(R2_train)
            RMSE_train_list.append(RMSE_train)
            MeanAE_train_list.append(MeanAE_train)
            MedianAE_train_list.append(MedianAE_train)
            MaxError_train_list.append(MaxError_train)
                
            # Scores test for one model and one split
            R2_test = r2_score(y_test.values, y_test_pred)
            RMSE_test = mean_squared_error(y_test.values, y_test_pred, squared=False)
            MeanAE_test = mean_absolute_error(y_test.values, y_test_pred)
            MedianAE_test = median_absolute_error(y_test.values, y_test_pred)
            MaxError_test = max_error(y_test.values, y_test_pred)
            
            # Scores test for all models and one split
            R2_test_list.append(R2_test)
            RMSE_test_list.append(RMSE_test)
            MeanAE_test_list.append(MeanAE_test)
            MedianAE_test_list.append(MedianAE_test)
            MaxError_test_list.append(MaxError_test)            
            
        # Scores train/test for all models and all splits
        list_all_R2_train.append(R2_train_list)
        list_all_RMSE_train.append(RMSE_train_list)
        list_all_MeanAE_train.append(MeanAE_train_list)
        list_all_MedianAE_train.append(MedianAE_train_list)
        list_all_MaxError_train.append(MaxError_train_list)
        list_all_R2_test.append(R2_test_list)
        list_all_RMSE_test.append(RMSE_test_list)
        list_all_MeanAE_test.append(MeanAE_test_list)
        list_all_MedianAE_test.append(MedianAE_test_list)
        list_all_MaxError_test.append(MaxError_test_list)
        list_all_train_time.append(train_time_list)
        list_all_y_train_pred.append(list_y_train_pred)
        list_all_y_test_pred.append(list_y_test_pred)
        
        print('')
        
    # Conversion from list of lists to array    
    list_all_R2_train = np.array(list_all_R2_train)
    list_all_RMSE_train = np.array(list_all_RMSE_train)
    list_all_MeanAE_train = np.array(list_all_MeanAE_train)
    list_all_MedianAE_train = np.array(list_all_MedianAE_train)
    list_all_MaxError_train = np.array(list_all_MaxError_train)
    list_all_R2_test = np.array(list_all_R2_test)
    list_all_RMSE_test = np.array(list_all_RMSE_test)
    list_all_MeanAE_test = np.array(list_all_MeanAE_test)
    list_all_MedianAE_test = np.array(list_all_MedianAE_test)
    list_all_MaxError_test = np.array(list_all_MaxError_test)
    list_all_train_time = np.array(list_all_train_time)
    
    # Calculation of mean and std of performances for all splits
    R2_train_mean = np.mean(list_all_R2_train, axis=0)
    RMSE_train_mean = np.mean(list_all_RMSE_train, axis=0)
    MeanAE_train_mean = np.mean(list_all_MeanAE_train, axis=0)
    MedianAE_train_mean = np.mean(list_all_MedianAE_train, axis=0)
    MaxError_train_mean = np.mean(list_all_MaxError_train, axis=0)
    R2_test_mean = np.mean(list_all_R2_test, axis=0)
    RMSE_test_mean = np.mean(list_all_RMSE_test, axis=0)
    MeanAE_test_mean = np.mean(list_all_MeanAE_test, axis=0)
    MedianAE_test_mean = np.mean(list_all_MedianAE_test, axis=0)
    MaxError_test_mean = np.mean(list_all_MaxError_test, axis=0)
    train_time_mean = np.mean(list_all_train_time, axis=0)
    
    R2_train_std = np.std(list_all_R2_train, axis=0)
    RMSE_train_std = np.std(list_all_RMSE_train, axis=0)
    MeanAE_train_std = np.std(list_all_MeanAE_train, axis=0)
    MedianAE_train_std = np.std(list_all_MedianAE_train, axis=0)
    MaxError_train_std = np.std(list_all_MaxError_train, axis=0)
    R2_test_std = np.std(list_all_R2_test, axis=0)
    RMSE_test_std = np.std(list_all_RMSE_test, axis=0)
    MeanAE_test_std = np.std(list_all_MeanAE_test, axis=0)
    MedianAE_test_std = np.std(list_all_MedianAE_test, axis=0)
    MaxError_test_std = np.std(list_all_MaxError_test, axis=0)
    train_time_std = np.std(list_all_train_time, axis=0)    
    
    # Summary of the results
    results_screening = np.empty(shape=(len(list_mdl)+1,22))
    results_screening[:,:] = math.nan
    results_screening[1:1+len(list_mdl),0] = R2_train_mean
    results_screening[1:1+len(list_mdl),1] = R2_train_std
    results_screening[1:1+len(list_mdl),2] = R2_test_mean
    results_screening[1:1+len(list_mdl),3] = R2_test_std
    results_screening[1:1+len(list_mdl),4] = RMSE_train_mean
    results_screening[1:1+len(list_mdl),5] = RMSE_train_std
    results_screening[1:1+len(list_mdl),6] = RMSE_test_mean
    results_screening[1:1+len(list_mdl),7] = RMSE_test_std
    results_screening[1:1+len(list_mdl),8] = MeanAE_train_mean
    results_screening[1:1+len(list_mdl),9] = MeanAE_train_std
    results_screening[1:1+len(list_mdl),10] = MeanAE_test_mean
    results_screening[1:1+len(list_mdl),11] = MeanAE_test_std
    results_screening[1:1+len(list_mdl),12] = MedianAE_train_mean
    results_screening[1:1+len(list_mdl),13] = MedianAE_train_std
    results_screening[1:1+len(list_mdl),14] = MedianAE_test_mean
    results_screening[1:1+len(list_mdl),15] = MedianAE_test_std
    results_screening[1:1+len(list_mdl),16] = MaxError_train_mean
    results_screening[1:1+len(list_mdl),17] = MaxError_train_std
    results_screening[1:1+len(list_mdl),18] = MaxError_test_mean
    results_screening[1:1+len(list_mdl),19] = MaxError_test_std
    results_screening[1:1+len(list_mdl),20] = train_time_mean
    results_screening[1:1+len(list_mdl),21] = train_time_std
    columns_names = ['R2 train mean', 'R2 train std', 'R2 test mean', 'R2 test std', 'RMSE train mean', 'RMSE train std', 'RMSE test mean', 'RMSE test std', 'MeanAE train mean', 'MeanAE train std', 'MeanAE test mean', 'MeanAE test std', 'MedianAE train mean', 'MedianAE train std', 'MedianAE test mean', 'MedianAE test std', 'MaxError train mean', 'MaxError train std', 'MaxError test mean', 'MaxError test std', 'Train time mean (s)', 'Train time std']
    index_names = ['*SCORES FOR ALL SPLITS FOR DIFFERENT MODELS:']
    index_names.extend(list_mdl_names)
    results_screening = pd.DataFrame(results_screening, columns=columns_names, index=index_names)                    
    
    # Parity plots
    fold_min_RMSE_all = pd.DataFrame(list_all_RMSE_test).idxmin()
    for k_mdl in range(len(list_mdl)):
        fold_min_RMSE = fold_min_RMSE_all[k_mdl]      
        plt.figure()
        plt.scatter(list_y_train[fold_min_RMSE], list_all_y_train_pred[fold_min_RMSE][k_mdl], label='train')
        plt.scatter(list_y_test[fold_min_RMSE], list_all_y_test_pred[fold_min_RMSE][k_mdl], label='test')
        plt.plot([0, 1], [0,1], color='black', label='y=x')
        plt.xlabel('Styrene conversion EXPERIMENTAL')                   
        plt.ylabel('Styrene conversion PREDICTED')
        plt.legend()
        plt.title(str(list_mdl_names[k_mdl])+' (fold'+str(fold_min_RMSE+1)+')', fontweight="bold")
    
    
#%% 3.2 Machine learning (definition of HPs boudaries and levels)

if choice_HP==2:
    
    # k-fold outer loop
    CV_out = KFold(n_splits=k_out_ML, shuffle=True, random_state=random_nb)
    
    # k-fold inner loop
    CV_in = KFold(n_splits=k_in_ML, shuffle=True, random_state=random_nb)

    # Models
    mdl_SVR = SVR()
    mdl_RF = RandomForestRegressor(random_state=random_nb)
    mdl_GB = GradientBoostingRegressor(random_state=random_nb)
    mdl_MLP = MLPRegressor(max_iter=800, random_state=random_nb)  
    
    # Evaluation of HPs for train set of each train/test split of the k-fold outer loop
    k=-1
    for i, j in CV_out.split(X):
        k=k+1
            
        # Preparation of X_train, X_test, y_train and y_test and scaling of X_train and X_test
        X_train = X.iloc[i,:]
        X_test = X.iloc[j,:]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train = data_final.iloc[i,-2]
        y_test = data_final.iloc[j,-2]
            
        if choice_mdl=='SVR':
            if choice_mdlHP=='C':
                param_range_values = [0.1, 0.2, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 
                train_score, val_score = validation_curve(mdl_SVR, X_train_scaled, y_train, param_name = 'C', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('C')
                plt.legend()
                plt.title('SVR, outer fold'+str(k+1))
            
            elif choice_mdlHP=='kernel':
                param_range_values = ['linear', 'rbf', 'poly', 'sigmoid']
                train_score, val_score = validation_curve(mdl_SVR, X_train_scaled, y_train, param_name = 'kernel', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('kernel')
                plt.legend()
                plt.title('SVR, outer fold'+str(k+1))

            elif choice_mdlHP=='epsilon':
                param_range_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]
                train_score, val_score = validation_curve(mdl_SVR, X_train_scaled, y_train, param_name = 'epsilon', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('epsilon')
                plt.legend()
                plt.title('SVR, outer fold'+str(k+1))    

            elif choice_mdlHP=='gamma':
                param_range_values = ['scale', 'auto']
                train_score, val_score = validation_curve(mdl_SVR, X_train_scaled, y_train, param_name = 'gamma', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('gamma')
                plt.legend()
                plt.title('SVR, outer fold'+str(k+1)) 
        
        elif choice_mdl=='RF':
            if choice_mdlHP=='n_estimators':
                param_range_values = [10, 50, 100, 150, 200, 500]
                train_score, val_score = validation_curve(mdl_RF, X_train_scaled, y_train, param_name = 'n_estimators', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('n_estimators')
                plt.legend()
                plt.title('RF, outer fold'+str(k+1))
            
            elif choice_mdlHP=='max_features':
                param_range_values = ['sqrt', 'log2', 'auto']
                train_score, val_score = validation_curve(mdl_RF, X_train_scaled, y_train, param_name = 'max_features', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")   
                plt.ylabel('RMSE')
                plt.xlabel('max_features')
                plt.legend()
                plt.title('RF, outer fold'+str(k+1))
                                
            elif choice_mdlHP=='min_samples_split':
                param_range_values = [2, 3, 4, 5, 10]
                train_score, val_score = validation_curve(mdl_RF, X_train_scaled, y_train, param_name = 'min_samples_split', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('min_samples_split')
                plt.legend()
                plt.title('RF, outer fold'+str(k+1))
                                
            elif choice_mdlHP=='min_samples_leaf':
                param_range_values = [1, 2, 3, 4, 5, 10]
                train_score, val_score = validation_curve(mdl_RF, X_train_scaled, y_train, param_name = 'min_samples_leaf', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('min_samples_leaf')
                plt.legend()
                plt.title('RF, outer fold'+str(k+1))
            
            elif choice_mdlHP=='max_samples':
                param_range_values = [int(X_train_scaled.shape[0]*2/5), int(X_train_scaled.shape[0]*3/5), int(X_train_scaled.shape[0]*4/5), int(X_train_scaled.shape[0]*5/5)]
                train_score, val_score = validation_curve(mdl_RF, X_train_scaled, y_train, param_name = 'max_samples', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                   
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('max_samples')
                plt.legend()
                plt.title('RF, outer fold'+str(k+1))
        
        elif choice_mdl=='GB':     
            if choice_mdlHP=='learning_rate':
                param_range_values = [0.001,0.005,0.01,0.05,0.07,0.1,0.2,0.3]
                train_score, val_score = validation_curve(mdl_GB, X_train_scaled, y_train, param_name = 'learning_rate', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")  
                plt.ylabel('RMSE')
                plt.xlabel('learning_rate')
                plt.legend()
                plt.title('GB, outer fold'+str(k+1))
            
            elif choice_mdlHP=='n_estimators':
                param_range_values = [10, 50, 100, 150, 200, 500]
                train_score, val_score = validation_curve(mdl_GB, X_train_scaled, y_train, param_name = 'n_estimators', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                  
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('n_estimators')
                plt.legend()
                plt.title('GB, outer fold'+str(k+1))
            
            elif choice_mdlHP=='max_features':
                param_range_values = ['sqrt', 'log2', 'auto']
                train_score, val_score = validation_curve(mdl_GB, X_train_scaled, y_train, param_name = 'max_features', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                   
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('max_features')
                plt.legend()
                plt.title('GB, outer fold'+str(k+1))
                               
            elif choice_mdlHP=='min_samples_split':
                param_range_values = [2, 3, 4, 5, 10, 20, 30]
                train_score, val_score = validation_curve(mdl_GB, X_train_scaled, y_train, param_name = 'min_samples_split', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('min_samples_split')
                plt.legend()
                plt.title('GB, outer fold'+str(k+1))
                                
            elif choice_mdlHP=='min_samples_leaf':
                param_range_values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
                train_score, val_score = validation_curve(mdl_GB, X_train_scaled, y_train, param_name = 'min_samples_leaf', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('min_samples_leaf')
                plt.legend()
                plt.title('GB, outer fold'+str(k+1))
            
            elif choice_mdlHP=='subsample':
                param_range_values = [1/5,2/5,3/5,4/5,1]
                train_score, val_score = validation_curve(mdl_GB, X_train_scaled, y_train, param_name = 'subsample', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('subsample')
                plt.legend()
                plt.title('GB, outer fold'+str(k+1))
           
        elif choice_mdl=='MLP':
            if choice_mdlHP=='activation':
                param_range_values = ['identity', 'logistic', 'tanh', 'relu']
                train_score, val_score = validation_curve(mdl_MLP, X_train_scaled, y_train, param_name = 'activation', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('activation')
                plt.legend()
                plt.title('MLP, outer fold'+str(k+1))
                            
            elif choice_mdlHP=='solver':
                param_range_values = ['lbfgs', 'sgd', 'adam']
                train_score, val_score = validation_curve(mdl_MLP, X_train_scaled, y_train, param_name = 'solver', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")   
                plt.ylabel('RMSE')
                plt.xlabel('solver')
                plt.legend()
                plt.title('MLP, outer fold'+str(k+1))
                                
            elif choice_mdlHP=='learning_rate':
                param_range_values = ['constant', 'invscaling', 'adaptive']
                train_score, val_score = validation_curve(mdl_MLP, X_train_scaled, y_train, param_name = 'learning_rate', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('learning_rate')
                plt.legend()
                plt.title('MLP, outer fold'+str(k+1))
            
            elif choice_mdlHP=='learning_rate_init':
                param_range_values = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                train_score, val_score = validation_curve(mdl_MLP, X_train_scaled, y_train, param_name = 'learning_rate_init', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('learning_rate_init')
                plt.legend()
                plt.title('MLP, outer fold'+str(k+1))
        
            elif choice_mdlHP=='alpha':
                param_range_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
                train_score, val_score = validation_curve(mdl_MLP, X_train_scaled, y_train, param_name = 'alpha', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('alpha')
                plt.legend()
                plt.title('MLP, outer fold'+str(k+1))
            
            elif choice_mdlHP=='max_iter':
                mdl_MLP = MLPRegressor(random_state=random_nb)
                param_range_values = [200, 500, 800]
                train_score, val_score = validation_curve(mdl_MLP, X_train_scaled, y_train, param_name = 'max_iter', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('max_iter')
                plt.legend()
                plt.title('MLP, outer fold'+str(k+1))
        
            elif choice_mdlHP=='hidden_layer_sizes':
                param_range_values =  [(5,5), (5,10), (5,15), (5,20), (5,25), (5,30), (5,50), 
                                        (10,5), (10,10), (10,15), (10,20), (10,25), (10,30), (10,50), 
                                        (15,5), (15,10), (15,15), (15,20), (15,25), (15,30), (15,50), 
                                        (20,5), (20,10), (20,15), (20,20), (20,25), (20,30), (20,50), 
                                        (25,5), (25,10), (25,15), (25,20), (25,25), (25,30), (25,50), 
                                        (30,5), (30,10), (30,15), (30,20), (30,25), (30,30), (30,50), 
                                        (50,5), (50,10), (50,15), (50,20), (50,25), (50,30), (50,50), 
                                        (25,), (50,), (75,), (100,), (125,), (150,)]
                param_range_values_bis = ['(5,5)', '(5,10)', '(5,15)', '(5,20)', '(5,25)', '(5,30)', '(5,50)',
                                        '(10,5)', '(10,10)', '(10,15)', '(10,20)', '(10,25)', '(10,30)', '(10,50)', 
                                        '(15,5)', '(15,10)', '(15,15)', '(15,20)', '(15,25)', '(15,30)', '(15,50)', 
                                        '(20,5)', '(20,10)', '(20,15)', '(20,20)', '(20,25)', '(20,30)', '(20,50)', 
                                        '(25,5)', '(25,10)', '(25,15)', '(25,20)', '(25,25)', '(25,30)', '(25,50)',
                                        '(30,5)', '(30,10)', '(30,15)', '(30,20)', '(30,25)', '(30,30)', '(30,50)', 
                                        '(50,5)', '(50,10)', '(50,15)', '(50,20)', '(50,25)', '(50,30)', '(50,50)', 
                                        '(25,)', '(50,)', '(75,)', '(100,)', '(125,)', '(150,)']
                train_score, val_score = validation_curve(mdl_MLP, X_train_scaled, y_train, param_name = 'hidden_layer_sizes', param_range = param_range_values, cv = k_in_ML, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
                plt.figure()
                plt.plot(param_range_values_bis, -train_score.mean(axis=1), label='train')
                plt.plot(param_range_values_bis, -val_score.mean(axis=1), label='validation')                    
                plt.fill_between(param_range_values_bis, -train_score.mean(axis=1) - train_score.std(axis=1),
                                  -train_score.mean(axis=1) + train_score.std(axis=1), alpha=0.2, color="blue")
                plt.fill_between(param_range_values_bis, -val_score.mean(axis=1) - val_score.std(axis=1),
                                  -val_score.mean(axis=1) + val_score.std(axis=1), alpha=0.2, color="orange")    
                plt.ylabel('RMSE')
                plt.xlabel('hidden_layer_sizes')
                plt.xticks(rotation = 90)
                plt.legend()
                plt.title('MLP, outer fold'+str(k+1))
                               
        
#%% 3.3 Machine learning (with HPs optimization)    
    
if choice_HP==3:
    
    print('Screening with HPs optimization:')
    print('--------------------------------')
    print('')
    
    # k-fold outer loop
    CV_out = KFold(n_splits=k_out_ML, shuffle=True, random_state=random_nb)
    
    # k-fold inner loop
    CV_in = KFold(n_splits=k_in_ML, shuffle=True, random_state=random_nb)

    # Models
    mdl_SVR = SVR()
    mdl_RF = RandomForestRegressor(random_state=random_nb, n_jobs=-1)
    mdl_GB = GradientBoostingRegressor(random_state=random_nb)
    mdl_MLP = MLPRegressor(max_iter=800, random_state=random_nb)  
    list_mdl = [mdl_SVR, mdl_RF, mdl_GB, mdl_MLP]
    list_mdl_names = ['Support Vector Regression', 'Random Forest', 'Gradient Boosting', 'Multi-Layer Perceptron']
    
    # Grid of hyperparameters for GridSearchCV
    param_grid_SVR = {'C': [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10],
                         'epsilon': [0.01, 0.5, 0.1]}
    param_grid_RF = {'n_estimators': [50, 100, 150],
                     'max_features': ['log2', 'sqrt']}
    param_grid_GB = {'n_estimators': [50, 100, 150],
                     'max_features': ['log2', 'sqrt'],
                     'min_samples_leaf': [1, 5, 10, 15],
                     'subsample': [1/5, 2/5, 3/5, 4/5, 1]}
    param_grid_MLP = {'hidden_layer_sizes': [(5,5), (5,10), (5,15), (5,20), (5,25), (5,30), (5,50),
                                             (10,5), (10,10), (10,15), (10,20), (10,25), (10,30), (10,50), 
                                             (15,5), (15,10), (15,15), (15,20), (15,25), (15,30), (15,50), 
                                             (20,5), (20,10), (20,15), (20,20), (20,25), (20,30), (20,50), 
                                             (25,5), (25,10), (25,15), (25,20), (25,25), (25,30), (25,50), 
                                             (30,5), (30,10), (30,15), (30,20), (30,25), (30,30), (30,50), 
                                             (50,5), (50,10), (50,15), (50,20), (50,25), (50,30), (50,50), 
                                             (25,), (50,), (75,), (100,), (125,), (150,)],
                      'solver': ['lbfgs', 'adam'], 
                      'learning_rate_init': [0.001,0.005, 0.01, 0.04, 0.07]}
    param_grid_list = [param_grid_SVR, param_grid_RF, param_grid_GB, param_grid_MLP]
    
    # Creation of lists
    list_train_nb = []
    list_train_id = []
    list_test_nb = []
    list_test_id = []    
    list_X_train_scaled = []
    list_X_test_scaled = []
    list_y_train = []
    list_y_test = []     
    if choice_weight_samples==1:
        list_weight_train = []
        list_weight_test = []    
    list_all_R2_train = []
    list_all_RMSE_train = []
    list_all_MeanAE_train = []
    list_all_MedianAE_train = []
    list_all_MaxError_train = []
    list_all_R2_test = []
    list_all_RMSE_test = []
    list_all_MeanAE_test = []
    list_all_MedianAE_test = []
    list_all_MaxError_test = []
    list_all_train_time = []
    list_all_HPopt_time = []    
    list_all_y_train_pred = []
    list_all_y_test_pred = []        
    
    k=-1
    for i, j in CV_out.split(X):
        k=k+1
        print('* Split', k+1)
        
        # Save info on train and test data
        list_train_nb.append(i.shape[0])
        list_train_id.append(data_final.iloc[i,:])
        list_test_nb.append(j.shape[0])
        list_test_id.append(data_final.iloc[j,:])
        
        # Preparation of X_train, X_test, y_train and y_test and scaling of X_train and X_test
        X_train = X.iloc[i,:]
        X_test = X.iloc[j,:]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        list_X_train_scaled.append(X_train_scaled)
        list_X_test_scaled.append(X_test_scaled)
        y_train = data_final.iloc[i,-2]
        y_test = data_final.iloc[j,-2]
        list_y_train.append(y_train)
        list_y_test.append(y_test)
        
        # Preparation of samples weights
        if choice_weight_samples == 1:
            weight_train = data_final.iloc[i,-1]
            weight_test = data_final.iloc[j,-1]
            list_weight_train.append(weight_train)
            list_weight_test.append(weight_test)
        
        # Screening
        HPopt_time_list = []
        train_time_list = []
        list_y_train_pred = []
        list_y_test_pred = []
        R2_train_list = []
        RMSE_train_list = []
        MeanAE_train_list = []
        MedianAE_train_list = []
        MaxError_train_list = []
        R2_test_list = []
        RMSE_test_list = []
        MeanAE_test_list = []
        MedianAE_test_list = []
        MaxError_test_list = []
        
        p=-1
        for mdl in list_mdl:    
            p = p+1       
            print('')
            print(list_mdl_names[p], ':')
            print('--------------------')
            
            # HP optimization
            t0hp = time.time() # starting time
            grid = GridSearchCV(list_mdl[p], param_grid = param_grid_list[p], scoring = 'neg_root_mean_squared_error', cv= k_in_ML, n_jobs=-1)        
            if list_mdl_names[p] in ['Support Vector Regression', 'Random Forest', 'Gradient Boosting']:
                if choice_weight_samples==1:
                    grid.fit(X_train_scaled, y_train, sample_weight=weight_train.tolist())
                else:
                    grid.fit(X_train_scaled, y_train)
            else:
                grid.fit(X_train_scaled, y_train)
            tfhp = time.time() # ending time
            print('HPs optimization time:',  round(tfhp-t0hp,3), 's')
            print('best score (lowest RMSE):', -grid.best_score_)
            print('best params:', grid.best_params_)
            mdl_opt = grid.best_estimator_
            HPopt_time_list.append(tfhp-t0hp)
            
            # Training and prediction with best HP
            t0 = time.time() # starting time
            if list_mdl_names[p] in ['Support Vector Regression', 'Random Forest', 'Gradient Boosting']:
                if choice_weight_samples==1:
                    mdl_opt.fit(X_train_scaled,y_train.values.ravel(), sample_weight=weight_train.tolist())
                else:
                    mdl_opt.fit(X_train_scaled,y_train.values.ravel())
            else:
                mdl_opt.fit(X_train_scaled,y_train.values.ravel())
            tf = time.time() # ending time
            print('training time:', round(tf-t0,3), 's')
            train_time_list.append(tf-t0)
            y_train_pred = mdl_opt.predict(X_train_scaled)
            y_test_pred = mdl_opt.predict(X_test_scaled)               
            list_y_train_pred.append(y_train_pred)
            list_y_test_pred.append(y_test_pred)            
            
            # Scores training for one model and one split
            R2_train = r2_score(y_train.values, y_train_pred)
            RMSE_train = mean_squared_error(y_train.values, y_train_pred, squared=False)
            MeanAE_train = mean_absolute_error(y_train.values, y_train_pred)
            MedianAE_train = median_absolute_error(y_train.values, y_train_pred)
            MaxError_train = max_error(y_train.values, y_train_pred)
            
            # Scores training for all models and one split
            R2_train_list.append(R2_train)
            RMSE_train_list.append(RMSE_train)
            MeanAE_train_list.append(MeanAE_train)
            MedianAE_train_list.append(MedianAE_train)
            MaxError_train_list.append(MaxError_train)
                
            # Scores test for one model and one split
            R2_test = r2_score(y_test.values, y_test_pred)
            RMSE_test = mean_squared_error(y_test.values, y_test_pred, squared=False)
            MeanAE_test = mean_absolute_error(y_test.values, y_test_pred)
            MedianAE_test = median_absolute_error(y_test.values, y_test_pred)
            MaxError_test = max_error(y_test.values, y_test_pred)
            
            # Scores test for all models and one split
            R2_test_list.append(R2_test)
            RMSE_test_list.append(RMSE_test)
            MeanAE_test_list.append(MeanAE_test)
            MedianAE_test_list.append(MedianAE_test)
            MaxError_test_list.append(MaxError_test)
                                
        # Scores train/test for all models and all splits
        list_all_R2_train.append(R2_train_list)
        list_all_RMSE_train.append(RMSE_train_list)
        list_all_MeanAE_train.append(MeanAE_train_list)
        list_all_MedianAE_train.append(MedianAE_train_list)
        list_all_MaxError_train.append(MaxError_train_list)
        list_all_R2_test.append(R2_test_list)
        list_all_RMSE_test.append(RMSE_test_list)
        list_all_MeanAE_test.append(MeanAE_test_list)
        list_all_MedianAE_test.append(MedianAE_test_list)
        list_all_MaxError_test.append(MaxError_test_list)
        list_all_train_time.append(train_time_list)
        list_all_HPopt_time.append(HPopt_time_list)
        list_all_y_train_pred.append(list_y_train_pred)
        list_all_y_test_pred.append(list_y_test_pred)
        
        print('')
        
    # Conversion from list of lists to array    
    list_all_R2_train = np.array(list_all_R2_train)
    list_all_RMSE_train = np.array(list_all_RMSE_train)
    list_all_MeanAE_train = np.array(list_all_MeanAE_train)
    list_all_MedianAE_train = np.array(list_all_MedianAE_train)
    list_all_MaxError_train = np.array(list_all_MaxError_train)
    list_all_R2_test = np.array(list_all_R2_test)
    list_all_RMSE_test = np.array(list_all_RMSE_test)
    list_all_MeanAE_test = np.array(list_all_MeanAE_test)
    list_all_MedianAE_test = np.array(list_all_MedianAE_test)
    list_all_MaxError_test = np.array(list_all_MaxError_test)
    list_all_train_time = np.array(list_all_train_time)
    
    # Calculation of mean and std of performances for all splits
    R2_train_mean = np.mean(list_all_R2_train, axis=0)
    RMSE_train_mean = np.mean(list_all_RMSE_train, axis=0)
    MeanAE_train_mean = np.mean(list_all_MeanAE_train, axis=0)
    MedianAE_train_mean = np.mean(list_all_MedianAE_train, axis=0)
    MaxError_train_mean = np.mean(list_all_MaxError_train, axis=0)
    R2_test_mean = np.mean(list_all_R2_test, axis=0)
    RMSE_test_mean = np.mean(list_all_RMSE_test, axis=0)
    MeanAE_test_mean = np.mean(list_all_MeanAE_test, axis=0)
    MedianAE_test_mean = np.mean(list_all_MedianAE_test, axis=0)
    MaxError_test_mean = np.mean(list_all_MaxError_test, axis=0)
    train_time_mean = np.mean(list_all_train_time, axis=0)
    
    R2_train_std = np.std(list_all_R2_train, axis=0)
    RMSE_train_std = np.std(list_all_RMSE_train, axis=0)
    MeanAE_train_std = np.std(list_all_MeanAE_train, axis=0)
    MedianAE_train_std = np.std(list_all_MedianAE_train, axis=0)
    MaxError_train_std = np.std(list_all_MaxError_train, axis=0)
    R2_test_std = np.std(list_all_R2_test, axis=0)
    RMSE_test_std = np.std(list_all_RMSE_test, axis=0)
    MeanAE_test_std = np.std(list_all_MeanAE_test, axis=0)
    MedianAE_test_std = np.std(list_all_MedianAE_test, axis=0)
    MaxError_test_std = np.std(list_all_MaxError_test, axis=0)
    train_time_std = np.std(list_all_train_time, axis=0)
        
    # Summary of the results
    results_screening = np.empty(shape=(len(list_mdl)+1,22))
    results_screening[:,:] = math.nan
    results_screening[1:1+len(list_mdl),0] = R2_train_mean
    results_screening[1:1+len(list_mdl),1] = R2_train_std
    results_screening[1:1+len(list_mdl),2] = R2_test_mean
    results_screening[1:1+len(list_mdl),3] = R2_test_std
    results_screening[1:1+len(list_mdl),4] = RMSE_train_mean
    results_screening[1:1+len(list_mdl),5] = RMSE_train_std
    results_screening[1:1+len(list_mdl),6] = RMSE_test_mean
    results_screening[1:1+len(list_mdl),7] = RMSE_test_std
    results_screening[1:1+len(list_mdl),8] = MeanAE_train_mean
    results_screening[1:1+len(list_mdl),9] = MeanAE_train_std
    results_screening[1:1+len(list_mdl),10] = MeanAE_test_mean
    results_screening[1:1+len(list_mdl),11] = MeanAE_test_std
    results_screening[1:1+len(list_mdl),12] = MedianAE_train_mean
    results_screening[1:1+len(list_mdl),13] = MedianAE_train_std
    results_screening[1:1+len(list_mdl),14] = MedianAE_test_mean
    results_screening[1:1+len(list_mdl),15] = MedianAE_test_std
    results_screening[1:1+len(list_mdl),16] = MaxError_train_mean
    results_screening[1:1+len(list_mdl),17] = MaxError_train_std
    results_screening[1:1+len(list_mdl),18] = MaxError_test_mean
    results_screening[1:1+len(list_mdl),19] = MaxError_test_std
    results_screening[1:1+len(list_mdl),20] = train_time_mean
    results_screening[1:1+len(list_mdl),21] = train_time_std   
    columns_names = ['R2 train mean', 'R2 train std', 'R2 test mean', 'R2 test std', 'RMSE train mean', 'RMSE train std', 'RMSE test mean', 'RMSE test std', 'MeanAE train mean', 'MeanAE train std', 'MeanAE test mean', 'MeanAE test std', 'MedianAE train mean', 'MedianAE train std', 'MedianAE test mean', 'MedianAE test std', 'MaxError train mean', 'MaxError train std', 'MaxError test mean', 'MaxError test std', 'Train time mean (s)', 'Train time std']
    index_names = ['*SCORES FOR ALL SPLITS FOR DIFFERENT MODELS:']
    index_names.extend(list_mdl_names)
    results_screening = pd.DataFrame(results_screening, columns=columns_names, index=index_names)                 
    
    # Parity plots
    fold_min_RMSE_all = pd.DataFrame(list_all_RMSE_test).idxmin()
    for k_mdl in range(len(list_mdl)):
        fold_min_RMSE = fold_min_RMSE_all[k_mdl]        
        plt.figure()
        plt.scatter(list_y_train[fold_min_RMSE], list_all_y_train_pred[fold_min_RMSE][k_mdl], label='train')
        plt.scatter(list_y_test[fold_min_RMSE], list_all_y_test_pred[fold_min_RMSE][k_mdl], label='test')
        plt.plot([0, 1], [0,1], color='black', label='y=x')
        
        plt.xlabel('Styrene conversion EXPERIMENTAL')                   
        plt.ylabel('Styrene conversion PREDICTED')
    
        plt.legend()
        plt.title(str(list_mdl_names[k_mdl])+' OPT (fold'+str(fold_min_RMSE+1)+')', fontweight="bold")