# import libraries required
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api= HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/LalithaRB/tourism-analysis/tourism.csv"
df=pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")
df.drop(columns=['CustomerID'], inplace=True)


label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])
df['ProdTaken'] = label_encoder.fit_transform(df['ProdTaken'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['CityTier'] = label_encoder.fit_transform(df['CityTier'])
df['Passport'] = label_encoder.fit_transform(df['Passport'])
df['OwnCar'] = label_encoder.fit_transform(df['OwnCar'])
#Define target variable
target_col='MonthlyIncome'
#Split into x and y
x=df.drop(['CustomerID'])
y=df[target_col]

#Perform train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.to_csv("Xtrain.csv", index=False)
x_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("Ytrain.csv", index=False)
y_test.to_csv("Ytest.csv", index=False)

files=["Xtrain.csv","Xtest.csv","Ytrain.csv","Ytest.csv"]

for file_path in files:
  api.upload_file(
      path_or_fileobj=file_path,
      path_in_repo=file_path.split("/")[-1],
      repo_id="LalithaRB/tourism-analysis",
      repo_type="dataset",
  )

