# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load

# Load the model
model = pickle.load(open('LRM.sav', 'rb'))


# Initialize an instance of FastAPI
app = FastAPI()

# Class which describes a single flower measurements
class IrisSpecies(BaseModel):
    host_is_superhost : int
    accommodates : int
    bedrooms : int
    beds : int
    bathrooms : int
    number_of_reviews : int

# Make a prediction based on the user-entered data
# Returns the predicted price with its respective probability

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Airbnb Dynamic Price Predictor FastAPI"}

# Define the route to the price predictor
@app.post('/predict_dynamic_price')
def predict_species(host_is_superhost,accommodates, bedrooms, beds, bathrooms, number_of_reviews):
    data_in = [[host_is_superhost,accommodates, bedrooms, beds, bathrooms, number_of_reviews]]
    prediction = model.predict(data_in)
    return prediction[0]