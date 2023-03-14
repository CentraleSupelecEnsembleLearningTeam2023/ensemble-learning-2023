# Functions to be used for modeling and training
import numpy as np

def reg_metrics(y_train,pred_train,y_test,pred_test,transform = False):
  '''
  Function to calculate rmse and R^2 on training and out of sample data.
  Possibility to also transform data back to regular form since features
  and targets log transformed for training
  '''
  from sklearn.metrics import mean_squared_error, r2_score

  if transform == True:
    y_train,pred_train,y_test,pred_test = np.e**y_train,np.e**pred_train,np.e**y_test,np.e**pred_test

  rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
  r2_train = r2_score(y_train, pred_train)
  print("Training RMSE:", rmse_train)
  print("Training R²:", r2_train)

  print("\n")

  rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
  r2_test = r2_score(y_test, pred_test)
  print("Out-of-sample RMSE:", rmse_test)
  print("Out-of-sample R²:", r2_test)

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
