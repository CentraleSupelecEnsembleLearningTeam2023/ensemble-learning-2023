# Functions to be used for modeling and training
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def reg_metrics(y_train,pred_train,y_test,pred_test,transform = False):
  '''
  Function to calculate rmse and R^2 on training and out of sample data.
  Possibility to also transform data back to regular form since features
  and targets log transformed for training.
  '''
  from sklearn.metrics import mean_squared_error, r2_score

  print("Log(Price) Metrics:")
  rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
  r2_train = r2_score(y_train, pred_train)
  print("Training RMSE:", rmse_train)
  print("Training R²:", r2_train)

  rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
  r2_test = r2_score(y_test, pred_test)
  print("Out-of-sample RMSE:", rmse_test)
  print("Out-of-sample R²:", r2_test)

  print("\n")

  if transform == True:
    y_train,pred_train,y_test,pred_test = np.e**y_train,np.e**pred_train,np.e**y_test,np.e**pred_test
    print("Price Metrics:")
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    r2_train = r2_score(y_train, pred_train)
    print("Training RMSE:", rmse_train)
    print("Training R²:", r2_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    r2_test = r2_score(y_test, pred_test)
    print("Out-of-sample RMSE:", rmse_test)
    print("Out-of-sample R²:", r2_test)

    print("\n")

def train_linear_reg(X_train,y_train,X_test,y_test,cross_val = False, transform = False):
  from sklearn.model_selection import cross_val_score, KFold
  from sklearn.linear_model import LinearRegression

  lr = LinearRegression()

  if cross_val == True:
    kfold = KFold(n_splits=10)
    print("Average kFold CV Score: {}".format(np.mean(cross_val_score(lr, X_train, y_train,
                                      scoring='r2', cv=kfold, n_jobs=-1))))

  else:
    lr.fit(X_train,y_train)
    reg_metrics(y_train,lr.predict(X_train),y_test,lr.predict(X_test),transform = transform)


def train_decision_tree(X_train,y_train,X_test,y_test,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False,transform = False):
  from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
  from sklearn.tree import DecisionTreeRegressor

  clf = DecisionTreeRegressor(max_depth = max_depth,min_samples_split = min_samples_split)
  
  if cross_val == True:
    kfold = KFold(n_splits=10)
    print("Average kFold CV Score: {}".format(np.mean(cross_val_score(clf, X_train, y_train,
                                      scoring='r2', cv=kfold, n_jobs=-1))))
  elif grid_search == True:
    pgrid = {"max_depth": [1, 2, 3, 4, 5, 6, 7],
      "min_samples_split": [2, 3, 5, 10, 15, 20]}

    grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid=pgrid, cv=10)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_.score(X_test, y_test))

  else:
    clf.fit(X_train,y_train)
    reg_metrics(y_train,clf.predict(X_train),y_test,clf.predict(X_test),transform = transform)


def train_random_forest(X_train,y_train,X_test,y_test,estimators = 100,max_depth= None,min_samples_split = 2,cross_val = False,grid_search = False,transform = False):
  from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
  from sklearn.ensemble import RandomForestRegressor

  rf = RandomForestRegressor(n_estimators = estimators,max_depth = max_depth,min_samples_split = min_samples_split)
  
  if cross_val == True:
    kfold = KFold(n_splits=10)
    print(np.mean(cross_val_score(rf, X_train, y_train,
                                      scoring='r2', cv=kfold, n_jobs=-1)))
  elif grid_search == True:
    pgrid = {"max_depth": [1, 2, 3, 4, 5, 6, 7],
      "min_samples_split": [2, 3, 5, 10, 15, 20]}

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid=pgrid, cv=10)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_.score(X_test, y_test))

  else:
    rf.fit(X_train,y_train)
    reg_metrics(y_train,rf.predict(X_train),y_test,rf.predict(X_test),transform = transform)

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
      cv_results.append(cross_val_score(regressor, X_train, y_train,
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

  return cvResDf


def train_lgbm(X_train,y_train,X_test,y_test,estimators = 200, lr = 0.07, n_jobs = -1, rs = 42,
               transform = False):
  from lightgbm import LGBMRegressor #Light Gradient Boosting Machine Learning Regressor
  import lightgbm as lgb

  lgbm_reg = LGBMRegressor(n_estimators = estimators, learning_rate = lr, n_jobs = n_jobs, random_state = rs)
  lgbm_reg.fit(X_train, y_train)

  # Predict log_SalePrice for the variables in the training set
  y_pred_train = lgbm_reg.predict(X_train)
  reg_metrics(y_train,lgbm_reg.predict(X_train),y_test,lgbm_reg.predict(X_test),transform = transform)

  lgb.plot_importance(lgbm_reg,max_num_features= 25)
  plt.show()

  return lgbm_reg

## training xgboost
def train_xgb(X_train, y_train, X_test, y_test, estimators = 200, lr = 0.07, rs = 42, transform = False,
              device = 'cpu', max_depth = 6):
  '''
  :param estimators: number of estimators
  :param lr: learning rate
  :param rs: random seed of the model
  :param transform: Boolean, indicating whether to transform the target variable or not
  :param device: device to use for training
  :param max_depth: maxim depth for the decision tree model
  :return: the trained XGBoost regressor object
  '''
  import xgboost as xgb

  if device == 'cuda':
    xgb_reg = xgb.XGBRegressor(n_estimators=estimators, learning_rate=lr, random_state= rs,
                            tree_method = 'gpu_hist', max_depth = max_depth)
  else:
    xgb_reg = xgb.XGBRegressor(n_estimators=estimators, learning_rate=lr, random_state= rs,
                               max_depth = max_depth)


  # Fit model to training set
  xgb_reg.fit(X_train, y_train)

  reg_metrics(y_train,xgb_reg.predict(X_train),y_test,xgb_reg.predict(X_test),transform = transform)

  xgb.plot_importance(xgb_reg, max_num_features = 25, title = "Top 25 features")
  plt.show()

  return xgb_reg

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
  :param transform: Boolean, whether to transform the target variable or not
  :param verbose: Boolean, whether to display training progress information or not
  :param data_in_leaf: the minimum number of data samples allowed in a leaf nod
  :return: the trained CatBoost regressor object
  '''
  from catboost import CatBoostRegressor, Pool
  # CatBoost Machine Learning Regressor that solves for Categorical feature using a permutation driven alternative

  # Defition of training plot
  train_dataset = Pool(data=X_train,
                       label=y_train,
                       cat_features=cat_features)

  # Definition of Cross-Validation Pool
  eval_dataset = Pool(data=X_test,
                      label=y_test,
                      cat_features=cat_features)

  # Initialize CatBoostClassifier
  cat_reg = CatBoostRegressor(n_estimators=estimators,
                              learning_rate=lr, max_depth=max_depth,
                              l2_leaf_reg=l2, eval_metric=eval_metric,
                              one_hot_max_size=one_hot_max_size, od_type=od_type, od_wait=od_wait,
                              min_data_in_leaf=data_in_leaf)

  cat_reg.fit(train_dataset, eval_set=eval_dataset, plot=False, verbose=verbose)

  reg_metrics(y_train, cat_reg.predict(X_train), y_test, cat_reg.predict(X_test), transform=transform)

  print(cat_reg.get_feature_importance(prettified=True).iloc[0:40])

  return cat_reg