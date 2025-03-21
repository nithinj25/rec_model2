from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import uvicorn
import pandas as pd
import io
import os
from model import train_recommendation_model, get_recommendations
import pickle

app = FastAPI(
    title="Insurance Policy Recommendation API",
    description="API for recommending insurance policies based on user information",
    version="1.0.0"
)

class UserInfo(BaseModel):
    Full_Name: str
    Insurance_Company: str = "Unknown"
    Policy_Name: str = "Unknown"
    Age: int
    Budget_in_INR: float
    Zip_Code: str
    Email: EmailStr
    Emergency_Services: bool
    Preventive_Care_and_Screening: bool
    Hospital_Stays_and_Treatments: bool
    Prescription_Medication: bool
    Smoking_Status: str
    Pre_existing_Health_Conditions: bool

class PolicyRecommendation(BaseModel):
    Full_Name: str
    Insurance_Company: str
    Policy_Name: str
    Budget_in_INR: float
    Age: int
    Features: dict
    Smoking_Status: str
    Similarity_Score: float

# Global variable to store the trained model
model_info = None
MODEL_PATH = "model.pkl"

def load_model():
    global model_info
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_info = pickle.load(f)

def save_model(model_data):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

# Load model on startup
load_model()

@app.get("/")
async def root():
    return {
        "message": "Welcome to Insurance Policy Recommendation API",
        "endpoints": {
            "/upload": "POST - Upload CSV data to train the model",
            "/recommend": "POST - Get insurance policy recommendations",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_trained": model_info is not None
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global model_info
    try:
        # Read the CSV file
        contents = await file.read()
        csv_data = contents.decode('utf-8')
        
        # Train the model with the uploaded data
        model_info = train_recommendation_model(pd.read_csv(io.StringIO(csv_data)))
        
        # Save the model
        save_model(model_info)
        
        return {
            "message": "Model trained successfully",
            "num_records": model_info['model_info']['num_records'],
            "num_companies": model_info['model_info']['num_companies'],
            "num_policies": model_info['model_info']['num_policies']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=List[PolicyRecommendation])
async def get_recommendations_endpoint(user_info: UserInfo):
    if model_info is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please upload training data first using the /upload endpoint."
        )
    
    try:
        # Convert Pydantic model to dictionary
        user_dict = user_info.dict()
        
        # Convert keys to match the model's expected format
        user_dict = {
            'Full Name': user_dict['Full_Name'],
            'Insurance Company': user_dict['Insurance_Company'],
            'Policy Name': user_dict['Policy_Name'],
            'Budget in INR': user_dict['Budget_in_INR'],
            'Age': user_dict['Age'],
            'Zip Code': user_dict['Zip_Code'],
            'Email': user_dict['Email'],
            'Emergency Services': user_dict['Emergency_Services'],
            'Preventive Care and Screening': user_dict['Preventive_Care_and_Screening'],
            'Hospital Stays and Treatments': user_dict['Hospital_Stays_and_Treatments'],
            'Prescription Medication': user_dict['Prescription_Medication'],
            'Smoking Status': user_dict['Smoking_Status'],
            'Pre-existing Health Conditions': user_dict['Pre_existing_Health_Conditions']
        }
        
        # Get recommendations
        recommendations = get_recommendations(model_info, user_dict)
        
        # Convert recommendations to match the response model
        formatted_recommendations = []
        for rec in recommendations:
            formatted_rec = PolicyRecommendation(
                Full_Name=rec['Full Name'],
                Insurance_Company=rec['Insurance Company'],
                Policy_Name=rec['Policy Name'],
                Budget_in_INR=rec['Budget in INR'],
                Age=rec['Age'],
                Features=rec['Features'],
                Smoking_Status=rec['Smoking Status'],
                Similarity_Score=rec['Similarity_Score']
            )
            formatted_recommendations.append(formatted_rec)
        
        return formatted_recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 