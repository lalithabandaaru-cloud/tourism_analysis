#for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import mlflow
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_Tourism_Experiment_Analysis")

api=HfApi()

Xtrain_path="hf://datasets/LalithaRB/tourism-analysis/Xtrain.csv"
Xtest_path="hf://datasets/LalithaRB/tourism-analysis/Xtest.csv"
ytrain_path="hf://datasets/LalithaRB/tourism-analysis/Ytrain.csv"
ytest_path="hf://datasets/LalithaRB/tourism-analysis/Ytest.csv"

X_train=pd.read_csv(Xtrain_path)
X_test=pd.read_csv(Xtest_path)
y_train=pd.read_csv(ytrain_path)
y_test=pd.read_csv(ytest_path)

#define numeric and categorical features
numeric_features=['Age','NumberOfPersonVisiting','PreferredPropertyStar','NumberOfTrips','NumberOfFollowups','DurationOfPitch','Passport','OwnCar']
categorical_features=['TypeofContact','CityTier','Occupation','Gender','MaritalStatus','Designation','ProductPitched']

#Preprocessor
preprocessor=make_column_transformer(
    (StandardScaler(),numeric_features),
    (OneHotEncoder(handle_unknown='ignore'),categorical_features)
)
#Define base XGBoost Regressor
xgb_model=xgb.XGBRegressor(random_state=42,n_jobs=-1)

#Define Random Forest Regressor
rf_model=RandomForestRegressor(random_state=42, n_jobs=-1)

#Hyperparameter grid
param_grid={
    'xgbregressor__n_estimators':[50,100,200],
    'xgbregressor__learning_rate':[0.01,0.1,0.2],
    'xgbregressor__max_depth':[3,4,5],
    'xgbregressor__subsample':[0.8,0.9,1.0],
    'xgbregressor__colsample_bytree':[0.8,0.9,1.0],
    'xgbregressor__reg_alpha':[0,0.1,1],
    'xgbregressor__reg_lambda':[0,0.1,1]
}

#Create pipeline
model_pipeline=make_pipeline(preprocessor,xgb_model)

with mlflow.start_run():
  #Train the model
  grid_search=GridSearchCV(model_pipeline,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
  grid_search.fit(X_train,y_train)

  #Log parameters sets
  results=grid_search.cv_results_
  for i in range(len(results['params'])):
    param_set=results['params'][i]
    mean_score=results['mean_test_score'][i]

    with mlflow.start_run(nested=True):
      mlflow.log_params(param_set)
      mlflow.log_metric('mean_neg_mse',mean_score)

  #Best model
  mlflow.log_param('best_params',grid_search.best_params_)
  best_model=grid_search.best_estimator_

  #Predictions
  y_pred_train=best_model.predict(X_train)
  y_pred_test=best_model.predict(X_test)

  #Metrics
  train_rmse=np.sqrt(mean_squared_error(y_train,y_pred_train))
  test_rmse=np.sqrt(mean_squared_error(y_test,y_pred_test))

  train_mae=mean_absolute_error(y_train,y_pred_train)
  test_mae=mean_absolute_error(y_test,y_pred_test)

  train_r2=r2_score(y_train,y_pred_train)
  test_r2=r2_score(y_test,y_pred_test)

  #Log metrics
  mlflow.log_metrics({
      "train_rmse":train_rmse,
      "test_rmse":test_rmse,
      "train_mae":train_mae,
      "test_mae":test_mae,
      "train_r2":train_r2,
      "test_r2":test_r2
  })

  #save the model locally
  model_path="Tourism_Way_to_Use.joblib"
  joblib.dump(best_model,model_path)
  print("Model saved locally")

  #Log the model locally
  mlflow.log_artifact(model_path, artifact_path="model")
  print(f"Model saved as artifact at: {model_path}")

  #upload the model to Hugging Face
  repo_id="LalithaRB/tourism-analysis"
  repo_type="model"
  try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists Using it.")
  except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space..") 
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created successfully")

  api.upload_file(
      path_or_fileobj=model_path,
      path_in_repo=model_path,
      repo_id=repo_id,
      repo_type=repo_type,
  )
