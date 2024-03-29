# Functions to be used for modeling and training
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def reg_metrics(y_train,pred_train,y_test,pred_test,transform = True):
  '''
  Function to calculate rmse and R^2 on training and out of sample data.
  Possibility to also transform data back to regular form since features
  and targets log transformed for training.
  '''
  from sklearn.metrics import mean_squared_error, r2_score
  
  print("Log(Price) Metrics:")
  rmse_train = np.sqrt(mean_squared_error(y_train, pred_train)) #calculate rmse train
  r2_train = r2_score(y_train, pred_train) #calculate r2 train
  print(f"Training RMSE:{rmse_train:03f}")
  print(f"Training R²: {r2_train:03f}")

  rmse_test = np.sqrt(mean_squared_error(y_test, pred_test)) #calculate rmse test
  r2_test = r2_score(y_test, pred_test) #calculate r2 test
  print(f"Out-of-sample RMSE: {rmse_test:03f}" )
  print(f"Out-of-sample R²: {r2_test:03f}")

  r2_train_log = None
  r2_test_log = None

  if transform == True: #if price requires inverse transform
    r2_train_log = r2_train #retain r2 train from log price
    r2_test_log = r2_test #retain r2 test from log price
    y_train,pred_train,y_test,pred_test = np.e**y_train,np.e**pred_train,np.e**y_test,np.e**pred_test #inverse log transform values
    print("Price Metrics:")
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    r2_train = r2_score(y_train, pred_train)
    print(f"Training RMSE:{rmse_train:03f}")
    print(f"Training R²: {r2_train:03f}")

    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    r2_test = r2_score(y_test, pred_test)
    print(f"Out-of-sample RMSE: {rmse_test:03f}")
    print(f"Out-of-sample R²: {r2_test:03f}")

    print("\n")
  return (rmse_train, r2_train, rmse_test, r2_test, r2_train_log,r2_test_log) #return tuple with all metrics

def update_scores(summary_dict, train_summary,model_name = ''):
  '''Function to store the scores obtained from reg_metrics into a dictionary 
  Parameters:
  -----------------
  summary_dict - a dictionary storing the summary metrics obtained from reg_metrics
  train_summary - summary statistics obtained from reg_metrics
  model_name - the name of the model trained to obtain train_summary

  Output:
  ---------------
  updated_dict - an updated summary_dict with the new metrics and model name
  '''
  updated_dict = summary_dict.copy()
  if train_summary == None:
    return updated_dict
  
  updated_dict['model'].append(model_name) #append name of model trained
  for i, key in enumerate(list(updated_dict.keys())[1:]): #append summary statistics from reg_metrics
    if train_summary[i] != None:
      updated_dict[key].append(round(train_summary[i],3))
    else:
      updated_dict[key].append(train_summary[i])
  
  return updated_dict

def train_linear_reg(X_train, y_train,X_test, y_test,cross_val = False, transform = False):
  from sklearn.model_selection import cross_val_score, KFold
  from sklearn.linear_model import LinearRegression

  lr = LinearRegression()
  summary_train = None

  if cross_val == True: #if cross_validation required
    kfold = KFold(n_splits=10)
    print("Average kFold CV Score: {}".format(np.mean(cross_val_score(lr, X_train, y_train,
                                      scoring='r2', cv=kfold, n_jobs=-1)))) #perform kfold cross validation and return mean CV R^2

  else:
    lr.fit(X_train,y_train) #fit Linear Regression model
    summary_train = reg_metrics(y_train,lr.predict(X_train),y_test,lr.predict(X_test),transform = transform) #obtain summary statistics

  return lr, summary_train

def train_decision_tree(X_train,y_train,X_test,y_test,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False,transform = False):
  '''
  :param max_depth: max_depth of tree
  :param min_samples_split: min samples required to split a node
  :param transform: Boolean, indicating whether to inverse transform the target variable or not
  :param cross_val: Boolean, indicating whether cross validation to be performed
  :param grid_search: Boolean, indicating whether to perform a gridsearch
  :return: the trained RF regressor object and the metrics from reg_metrics
  '''
  from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
  from sklearn.tree import DecisionTreeRegressor

  clf = DecisionTreeRegressor(max_depth = max_depth,min_samples_split = min_samples_split) #parameters investigated are max_depth and min_samples_split
  summary_train = None

  if cross_val == True: # if true perform kfold CV and return mean R^2
    kfold = KFold(n_splits=10)
    print("Average kFold CV Score: {}".format(np.mean(cross_val_score(clf, X_train, y_train,
                                      scoring='r2', cv=kfold, n_jobs=-1))))
  elif grid_search == True: #if true perform grid search - below grid search parameters were the ones performed in final run
    pgrid = {"max_depth": [ 7, 10, 15 ],
      "min_samples_split": [ 10, 20]}

    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid=pgrid, cv=10) #run grid search
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_) #orints best parameters and CV score

  else:
    clf.fit(X_train,y_train)
    summary_train = reg_metrics(y_train,clf.predict(X_train),y_test,clf.predict(X_test),transform = transform) #return summary stats

  return clf, summary_train


def train_random_forest(X_train,y_train,X_test,y_test,estimators = 100,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False,transform = False):
  '''
  :param estimators: number of estimators
  :param max_depth: max_depth of tree
  :param min_samples_split: min samples required to split a node
  :param transform: Boolean, indicating whether to inverse transform the target variable or not
  :param cross_val: Boolean, indicating whether cross validation to be performed
  :param grid_search: Boolean, indicating whether to perform a gridsearch
  :return: the trained RF regressor object and the metrics from reg_metrics
  '''
  from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
  from sklearn.ensemble import RandomForestRegressor

  rf = RandomForestRegressor(n_estimators = estimators,max_depth = max_depth,min_samples_split = min_samples_split) #parameters investigated are max_depth and min_samples_split
  summary_train = None

  if cross_val == True: 
    kfold = KFold(n_splits=10)
    print(np.mean(cross_val_score(rf, X_train, y_train,
                                      scoring='r2', cv=kfold, n_jobs=-1)))
  elif grid_search == True:
    pgrid = {"max_depth": [7, 10 , 11, 20], #grid search performed in final grid search trial
      "min_samples_split": [2, 5, 10, 20]}

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid=pgrid, cv=10)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

  else:
    rf.fit(X_train,y_train)
    summary_train = reg_metrics(y_train,rf.predict(X_train),y_test,rf.predict(X_test),transform = transform)
  
  return rf, summary_train

def train_ensemble_models(X_train, y_train):
  '''
  Training an ensemble of several models to find best performing cv performance from:
  Linear Regression
  Random Forest Regressor
  Decision Tree Regressor
  Support Vector Machine Regressor
  K Neighbors Reggresor
  Gradient Boosting Regressor
  Extra Trees Regressor
  '''
  from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.svm import SVR
  from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

  kfold = KFold(n_splits=10)

  regressors = []
  regressors.append(SVR()) 
  regressors.append(DecisionTreeRegressor()) 
  regressors.append(RandomForestRegressor())
  regressors.append(ExtraTreesRegressor())
  regressors.append(GradientBoostingRegressor())
  regressors.append(KNeighborsRegressor())
  regressors.append(LinearRegression())

  cv_results = []
  for regressor in regressors:
      cv_results.append(cross_val_score(regressor, X_train, y_train.values.ravel(),
                                        scoring='r2', cv=kfold, n_jobs=-1))
      print("{} Regressor CV complete!".format(regressor))
  cv_means = []
  cv_std = []
  for cv_result in cv_results:
      cv_means.append(cv_result.mean())
      cv_std.append(cv_result.std())

  cvResDf = pd.DataFrame({'cv_mean': cv_means,
                          'cv_std': cv_std,
                          'algorithm': ['SVR', 'DecisionTreeReg', 'RandomForestReg', 'ExtraTreesReg',
                                        'GradientBoostingReg', 'KNN', 'LinearReg']})
  
  cvResFacet=sns.FacetGrid(cvResDf.sort_values(by='cv_mean',ascending=False),sharex=False,
            sharey=False,aspect=2)
  cvResFacet.map(sns.barplot,'cv_mean','algorithm',**{'xerr':cv_std},
               palette='muted')
  cvResFacet.add_legend()
  plt.show()

  return cvResDf


def train_lgbm(X_train,y_train,X_test,y_test,estimators = 200, lr = 0.07, n_jobs = -1, rs = 42,
               transform = False):
  
  '''
  :param estimators: number of estimators
  :param lr: learning rate
  :param rs: random seed of the model
  :param transform: Boolean, indicating whether to inverse transform the target variable or not
  :return: the trained LGBM regressor object and the metrics from reg_metrics
  '''
  from lightgbm import LGBMRegressor #Light Gradient Boosting Machine Learning Regressor
  import lightgbm as lgb

  lgbm_reg = LGBMRegressor(n_estimators = estimators, learning_rate = lr, n_jobs = n_jobs, random_state = rs)
  lgbm_reg.fit(X_train, y_train)

  # Predict log_SalePrice for the variables in the training set
  y_pred_train = lgbm_reg.predict(X_train)
  summary_train = reg_metrics(y_train,lgbm_reg.predict(X_train),y_test,lgbm_reg.predict(X_test),transform = transform)

  lgb.plot_importance(lgbm_reg,max_num_features= 25, title = "Top 25 features") #plots 25 most important features on the basis of "split" (the number of times the feature was used to split the model)
  plt.show()

  return lgbm_reg, summary_train


## training xgboost
def train_xgb(X_train, y_train, X_test, y_test, estimators = 200, lr = 0.07, rs = 42, transform = False,
              device = 'cpu', max_depth = 6):
  '''
  :param estimators: number of estimators
  :param lr: learning rate
  :param rs: random seed of the model
  :param transform: Boolean, indicating whether to inversentransform the target variable or not
  :param device: device to use for training (requires suitable version of xgboost installed if cuda)
  :param max_depth: maxim depth for the decision tree model
  :return: the trained XGBoost regressor object
  '''
  import xgboost as xgb

  if device == 'cuda':
    xgb_reg = xgb.XGBRegressor(n_estimators=estimators, learning_rate=lr, random_state= rs,
                            tree_method = 'gpu_hist', max_depth = max_depth)
  else:
    xgb_reg = xgb.XGBRegressor(n_estimators=estimators, learning_rate=lr, random_state= rs)


  # Fit model to training set
  xgb_reg.fit(X_train, y_train)

  y_pred_train = xgb_reg.predict(X_train)
  train_summary = reg_metrics(y_train,xgb_reg.predict(X_train),y_test,xgb_reg.predict(X_test),transform = transform)

  xgb.plot_importance(xgb_reg, max_num_features = 25, title = "Top 25 features") #Top 25 feature importances on the basis of how many times feature appears in tree
  plt.show()

  return xgb_reg, train_summary


## training catboost
def train_catboost(X_train, y_train, X_test, y_test, estimators=3000, lr=1 / 10, max_depth=6,
                   l2=5, eval_metric="R2", one_hot_max_size=1000, od_type= None, od_wait= None,
                   transform=False, verbose=False, data_in_leaf=1, cat_features = []):
  '''
  :param estimators: number of estimators
  :param lr: learning rate
  :param max_depth: maximum depth of the decision trees in the model
  :param l2: L2 regularization coefficient for the model
  :param eval_metric: evaluation metric used to assess model performance
  :param one_hot_max_size: maximum size of a categorical feature that will be converted to one-hot encoding
  :param od_type: type of overfitting detector to use
  :param od_wait:  number of iterations to wait before stopping the training process if no improvement in the evaluation metric is observed
  :param transform: Boolean, whether to inverse transform the target variable or not
  :param verbose: Boolean, whether to display training progress information or not
  :param data_in_leaf: the minimum number of data samples allowed in a leaf nod
  :param cat_feature: specify the features that are categorical variables
  :return: the trained CatBoost regressor object
  '''
  from catboost import CatBoostRegressor, Pool
  from sklearn.model_selection import train_test_split
  # CatBoost Machine Learning Regressor that solves for Categorical feature using a permutation driven alternative
  
  #Data split into a train/cross_val pool
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state = 123)

  # Defition of training plot
  train_dataset = Pool(data=X_train,
                       label=y_train,
                       cat_features=cat_features)

  # Definition of Cross-Validation Pool
  eval_dataset = Pool(data=X_val,
                      label=y_val,
                      cat_features=cat_features)

  # Initialize CatBoostClassifier
  cat_reg = CatBoostRegressor(n_estimators=estimators,
                              learning_rate=lr, max_depth=max_depth,
                              l2_leaf_reg=l2, eval_metric=eval_metric,
                              one_hot_max_size=one_hot_max_size, od_type=od_type, od_wait=od_wait,
                              min_data_in_leaf=data_in_leaf)

  cat_reg.fit(train_dataset, eval_set=eval_dataset, plot=False, verbose=verbose)

  train_summary = reg_metrics(y_train, cat_reg.predict(X_train), y_test, cat_reg.predict(X_test), transform=transform)

  print(cat_reg.get_feature_importance(prettified=True).iloc[0:25]) #plot 25 most important features based on how much on average the prediction changes if the feature value changes.

  return cat_reg, train_summary
