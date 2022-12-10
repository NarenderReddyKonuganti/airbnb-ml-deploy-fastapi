# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load
import pickle
from pydantic import BaseModel

# Load the model
model = pickle.load(open('models/LRM.sav', 'rb'))

# Initialize an instance of FastAPI
app = FastAPI()

# Class which describes input variables
class IrisSpecies(BaseModel):
    host_is_superhost : int
    accommodates : int
    bedrooms : int
    beds : int
    bathrooms : int
    number_of_reviews : int

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Airbnb Dynamic Price Predictor FastAPI"}

# Define the route to the price predictor
@app.post('/predict_dynamic_price')

# Make a prediction based on the user-entered data
# Returns the predicted price with its respective probability
def predict_price(host_is_superhost,accommodates, bedrooms, beds, bathrooms, number_of_reviews):
    data_in = [[host_is_superhost,accommodates, bedrooms, beds, bathrooms, number_of_reviews]]
    prediction = model.predict(data_in)
    return prediction[0]