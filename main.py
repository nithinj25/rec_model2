from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd

app = FastAPI(
    title="Insurance Policy Recommendation API",
    description="Insurance policy recommendations based on complete dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your complete dataset
try:
    # Replace 'your_complete_dataset.csv' with your actual CSV filename
    df = pd.read_csv('your_complete_dataset.csv')
    print(f"Loaded dataset with {len(df)} records")
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()

@app.get("/recommend")
async def get_recommendations(
    Full_Name: str = Query(..., description="Full name of the user"),
    Age: int = Query(..., description="Age of the user", ge=0),
    Budget_in_INR: float = Query(..., description="Budget in INR", ge=0),
    Zip_Code: str = Query(..., description="Zip code"),
    Email: str = Query(..., description="Email address"),
    Emergency_Services: str = Query(..., description="Emergency Services (true/false)"),
    Preventive_Care_and_Screenings: str = Query(..., description="Preventive Care and Screenings (true/false)"),
    Hospital_Stays_and_Treatments: str = Query(..., description="Hospital Stays and Treatments (true/false)"),
    Prescription_Medication: str = Query(..., description="Prescription Medication (true/false)"),
    Smoking_Status: str = Query(..., description="Smoking Status (Smoker/Non-Smoker)"),
    Pre_existing_Health_Conditions: str = Query(..., description="Pre-existing Health Conditions (true/false)")
):
    try:
        # Convert string boolean values to actual booleans
        emergency = Emergency_Services.lower() == 'true'
        preventive = Preventive_Care_and_Screenings.lower() == 'true'
        hospital = Hospital_Stays_and_Treatments.lower() == 'true'
        prescription = Prescription_Medication.lower() == 'true'
        pre_existing = Pre_existing_Health_Conditions.lower() == 'true'

        # Find matching policies from the complete dataset
        matching_policies = df[
            (df['Budget in INR'] <= Budget_in_INR) &
            (df['Emergency Services'] == emergency) &
            (df['Preventive Care and Screening'] == preventive) &
            (df['Hospital Stays and Treatments'] == hospital) &
            (df['Prescription Medication'] == prescription) &
            (df['Smoking Status'] == Smoking_Status)
        ]

        # Sort by budget to get the most cost-effective options
        matching_policies = matching_policies.sort_values('Budget in INR')

        # Convert matching policies to list of recommendations
        recommendations = []
        for _, policy in matching_policies.iterrows():
            recommendations.append({
                'Policy_Name': policy['Policy Name'],
                'Insurance_Company': policy['Insurance Company'],
                'Budget_in_INR': float(policy['Budget in INR']),
                'Features': {
                    'Emergency_Services': bool(policy['Emergency Services']),
                    'Preventive_Care_and_Screening': bool(policy['Preventive Care and Screening']),
                    'Hospital_Stays_and_Treatments': bool(policy['Hospital Stays and Treatments']),
                    'Prescription_Medication': bool(policy['Prescription Medication'])
                }
            })

        return {'recommendations': recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a root endpoint that shows API info
@app.get("/")
async def root():
    return {
        "message": "Welcome to Insurance Policy Recommendation API",
        "usage": "Use /recommend endpoint with query parameters",
        "example": "/recommend?Full_Name=John%20Doe&Age=35&Budget_in_INR=150000&Zip_Code=400001&Email=john@example.com&Emergency_Services=true&Preventive_Care_and_Screenings=true&Hospital_Stays_and_Treatments=true&Prescription_Medication=true&Smoking_Status=Non-Smoker&Pre_existing_Health_Conditions=false"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 