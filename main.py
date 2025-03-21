from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="Insurance Policy Recommendation API",
    description="Simple API to get insurance policies based on user parameters",
    version="1.0.0"
)

# Add a root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Insurance Policy Recommendation API",
        "endpoints": {
            "/recommend": "POST - Get insurance policy recommendations",
            "/docs": "GET - API documentation"
        }
    }

# Define the input model based on required parameters
class UserInput(BaseModel):
    Full_Name: str
    Age: int
    Budget_in_INR: float
    Zip_Code: str
    Email: str
    Emergency_Services: bool
    Preventive_Care_and_Screening: bool
    Hospital_Stays_and_Treatments: bool
    Prescription_Medication: bool
    Smoking_Status: str  # "Smoker" or "Non-Smoker"
    Pre_existing_Health_Conditions: bool

@app.post("/recommend")
async def recommend_policies(user: UserInput):
    try:
        # Here you would add your logic to read from your existing CSV
        # and return matching policies based on user parameters
        
        # Example response format
        return {
            "recommendations": [
                {
                    "Insurance_Company": "Example Insurance Co",
                    "Policy_Name": "Example Policy",
                    "Budget_in_INR": user.Budget_in_INR,
                    "Features": {
                        "Emergency_Services": user.Emergency_Services,
                        "Preventive_Care_and_Screening": user.Preventive_Care_and_Screening,
                        "Hospital_Stays_and_Treatments": user.Hospital_Stays_and_Treatments,
                        "Prescription_Medication": user.Prescription_Medication
                    }
                }
                # Add more recommendations as needed
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 