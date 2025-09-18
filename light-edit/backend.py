from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# ----------------------------
# Load Model, Encoders, Scaler
# ----------------------------
with open("random_forest_agri_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("crop_encoder.pkl", "rb") as f:
    crop_encoder = pickle.load(f)

with open("irrigation_encoder.pkl", "rb") as f:
    irrigation_encoder = pickle.load(f)

with open("soil_encoder.pkl", "rb") as f:
    soil_encoder = pickle.load(f)

with open("season_encoder.pkl", "rb") as f:
    season_encoder = pickle.load(f)

with open("scaler_agri.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns_order_agri.pkl", "rb") as f:
    columns_order = pickle.load(f)

# ----------------------------
# FastAPI app initialization
# ----------------------------
app = FastAPI()

# ----------------------------
# Define Input Schema
# ----------------------------
class CropInput(BaseModel):
    Crop_Type: str
    Irrigation: str
    Soil_Type: str
    Season: str
    Farm_Area_acres: float
    Fertilizer_Used_tons: float
    Pesticide_Used_kg: float
    Water_Usage_cubic_meters: float

# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def home():
    return {"message": "🌱 Crop Yield Prediction API is running 🚀"}

@app.post("/predict")
def predict_yield(data: CropInput):
    # Convert input to DataFrame
    user_data = pd.DataFrame([{
        "Crop_Type": data.Crop_Type,
        "Irrigation": data.Irrigation,
        "Soil_Type": data.Soil_Type,
        "Season": data.Season,
        "Farm_Area(acres)": data.Farm_Area_acres,
        "Fertilizer_Used(tons)": data.Fertilizer_Used_tons,
        "Pesticide_Used(kg)": data.Pesticide_Used_kg,
        "Water_Usage(cubic meters)": data.Water_Usage_cubic_meters
    }])

    # Encode categorical values
    user_data["Crop_Type"] = crop_encoder.transform(user_data["Crop_Type"])
    user_data["Irrigation"] = irrigation_encoder.transform(user_data["Irrigation"])
    user_data["Soil_Type"] = soil_encoder.transform(user_data["Soil_Type"])
    user_data["Season"] = season_encoder.transform(user_data["Season"])

    # Scale numerical values
    num_cols = ["Farm_Area(acres)", "Fertilizer_Used(tons)", "Pesticide_Used(kg)", "Water_Usage(cubic meters)"]
    user_data[num_cols] = scaler.transform(user_data[num_cols])

    # Reorder columns to match training
    user_data = user_data.reindex(columns=columns_order, fill_value=0)

    # Make prediction
    prediction = model.predict(user_data)[0]

    return {
        "Predicted_Yield(tons)": round(float(prediction), 2)
    }
