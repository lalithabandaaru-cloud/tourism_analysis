import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

#Download and load the trained model
model_path="LalithaRB/tourism-analysis"
model_path=hf_hub_download(repo_id=model_path,filename="Tourism_Way_to_Use.joblib")
model=joblib.load(model_path)

#stramlit UI
st.title("Tourism Income Prediction")
st.write("""
This application predicts the expected revenue of a Way to Use Tourism"""
)

#User input
app_category=st.Selectbox("App Category", ["TypeofContact","CityTier","Occupation","Gender","MaritalStatus","Designation","ProductPitched"])
age=st.number_input("Age", min_value=18, max_value=100, value=30)
number_of_person_visiting=st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
preferred_property_star=st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)

number_of_trips=st.number_input("Number of Trips", min_value=1, max_value=10, value=2)
number_of_followups=st.number_input("Number of Followups", min_value=1, max_value=10, value=2)
duration_of_pitch=st.number_input("Duration of Pitch", min_value=1, max_value=10, value=2)
passport=st.number_input("Passport", min_value=0, max_value=1, value=0)
own_car=st.number_input("Own Car", min_value=0, max_value=1, value=0)

#Assemble input into Dataframe
input_data=pd.DataFrame({
    'app_category':[app_category],
    "Age":[age],
    "NumberOfPersonVisiting":[number_of_person_visiting],
    "PreferredPropertyStar":[preferred_property_star],
    "NumberOfTrips":[number_of_trips],
    "NumberOfFollowups":[number_of_followups],
    "DurationOfPitch":[duration_of_pitch],
    "Passport":[passport],
    "OwnCar":[own_car]
})

#Prediction button
if st.button("Predict Revenue"):
  prediction=model.predict(input_data)[0]
  st.subheader("Prediction Result of Tourism:")
  st.success(f"The predicted revenue is **${prediction:,.2f}**")
