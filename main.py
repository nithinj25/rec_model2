from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os

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

# Load the dataset with better error handling
try:
    # Print debugging information
    print("Current directory:", os.getcwd())
    print("Available files:", os.listdir())
    
    df = pd.read_csv('indian_insurance_data.csv')  # Updated filename
    print("Dataset loaded successfully")
    print("Number of records:", len(df))
    print("Available columns:", df.columns.tolist())
except Exception as e:
    print(f"Error loading CSV: {e}")
    raise Exception(f"Failed to load dataset: {e}")

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
        # Debug print statements
        print("Received request with parameters:")
        print(f"Budget: {Budget_in_INR}")
        print(f"Emergency Services: {Emergency_Services}")
        
        # Convert string boolean values to actual booleans
        emergency = Emergency_Services.lower() == 'true'
        preventive = Preventive_Care_and_Screenings.lower() == 'true'
        hospital = Hospital_Stays_and_Treatments.lower() == 'true'
        prescription = Prescription_Medication.lower() == 'true'
        pre_existing = Pre_existing_Health_Conditions.lower() == 'true'

        # Print first few rows of dataframe for debugging
        print("First few rows of dataset:")
        print(df.head())

        # Find matching policies with more detailed error handling
        try:
            matching_policies = df[
                (df['Budget in INR'] <= Budget_in_INR) &
                (df['Emergency Services'] == emergency) &
                (df['Preventive Care and Screening'] == preventive) &
                (df['Hospital Stays and Treatments'] == hospital) &
                (df['Prescription Medication'] == prescription) &
                (df['Smoking Status'] == Smoking_Status)
            ]
            print(f"Found {len(matching_policies)} matching policies")
        except KeyError as e:
            print(f"Column not found: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid column name: {e}")

        # Create recommendations
        recommendations = {
            'user_info': {
                'Full_Name': Full_Name,
                'Age': Age,
                'Budget_in_INR': Budget_in_INR,
                'Email': Email,
                'Requirements': {
                    'Emergency_Services': emergency,
                    'Preventive_Care_and_Screening': preventive,
                    'Hospital_Stays_and_Treatments': hospital,
                    'Prescription_Medication': prescription,
                    'Smoking_Status': Smoking_Status,
                    'Pre_existing_Health_Conditions': pre_existing
                }
            },
            'recommendations': []
        }

        # Add matching policies to recommendations
        for _, policy in matching_policies.iterrows():
            recommendations['recommendations'].append({
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

        # Save recommendations to JSON file
        filename = f"recommendations_{Full_Name.replace(' ', '_')}_{Age}.json"
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=4)

        return recommendations

    except Exception as e:
        print(f"Error in get_recommendations: {e}")
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