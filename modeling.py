# Functions to be used for modeling and training

def reg_metrics(y_train,pred_train,y_test,pred_test):
  '''
  Function to calculate rmse and R^2 on training and out of sample data.
  '''
  from sklearn.metrics import mean_squared_error, r2_score

  rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
  r2_train = r2_score(y_train, pred_train)
  print("Training RMSE:", rmse_train)
  print("Training R²:", r2_train)

  print("\n")

  rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
  r2_test = r2_score(y_test, pred_test)
  print("Out-of-sample RMSE:", rmse_test)
  print("Out-of-sample R²:", r2_test)

def train_linear_reg(X_train,y_train,X_test,y_test):
  '''
  Function to conduct linear regression model.
  '''
  from sklearn.model_selection import cross_val_score, KFold
  from sklearn.linear_model import LinearRegression

  lr = LinearRegression()

  lr.fit(X_train,y_train)
  reg_metrics(y_train,lr.predict(X_train),y_test,lr.predict(X_test))

def train_decision_tree(X_train,y_train,X_test,y_test,max_depth= None,min_samples_split = 2):
  '''
  Function to conduct decision tree model.
  '''
  from sklearn.tree import DecisionTreeRegressor

  clf = DecisionTreeRegressor(max_depth = max_depth,min_samples_split = min_samples_split)

  clf.fit(X_train,y_train)
  reg_metrics(y_train,clf.predict(X_train),y_test,clf.predict(X_test))


def train_random_forest(X_train,y_train,X_test,y_test,estimators = 100,max_depth= None,min_samples_split = 2):
  from sklearn.ensemble import RandomForestRegressor

  rf = RandomForestRegressor(n_estimators = estimators,max_depth = max_depth,min_samples_split = min_samples_split)

  rf.fit(X_train,y_train)
  reg_metrics(y_train,rf.predict(X_train),y_test,rf.predict(X_test))