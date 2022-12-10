# 1. Library imports
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
from fastapi import FastAPI
import uvicorn
import pickle
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
import numpy as np


app = FastAPI()
model = pickle.load(open('LRM.sav', 'rb'))

# 2. Class which describes a single flower measurements
class IrisSpecies(BaseModel):
    host_is_superhost : int
    accommodates : int
    bedrooms : int
    beds : int
    bathrooms : int
    number_of_reviews : int

# 5. Make a prediction based on the user-entered data
# # Returns the predicted species with its respective probability

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Airbnb Dynamic Price Predictor FastAPI"}

@app.post('/predict_dynamic_price')
def predict_species(host_is_superhost,accommodates, bedrooms, beds, bathrooms, number_of_reviews):
    data_in = [[host_is_superhost,accommodates, bedrooms, beds, bathrooms, number_of_reviews]]
    prediction = model.predict(data_in)
    return prediction[0]