from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(
    title="Insurance Policy Recommendation API",
    description="Insurance policy recommendations based on CSV reference data",
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

# Load the reference CSV file
df = pd.read_csv('mock_insurance_data_100.csv')

@app.get("/")
async def get_recommendations(
    Full_Name: str,
    Age: int,
    Budget_in_INR: float,
    Zip_Code: str,
    Email: str,
    Emergency_Services: bool,
    Preventive_Care_and_Screenings: bool,
    Hospital_Stays_and_Treatments: bool,
    Prescription_Medication: bool,
    Smoking_Status: str,
    Pre_existing_Health_Conditions: bool
):
    try:
        # Find matching policies from the CSV
        matching_policies = df[
            (df['Budget in INR'] <= Budget_in_INR) &
            (df['Emergency Services'] == Emergency_Services) &
            (df['Preventive Care and Screening'] == Preventive_Care_and_Screenings) &
            (df['Hospital Stays and Treatments'] == Hospital_Stays_and_Treatments) &
            (df['Prescription Medication'] == Prescription_Medication) &
            (df['Smoking Status'] == Smoking_Status)
        ]

        # Convert matching policies to list of recommendations
        recommendations = []
        for _, policy in matching_policies.iterrows():
            recommendations.append({
                'Policy_Name': policy['Policy Name'],
                'Insurance_Company': policy['Insurance Company'],
                'Budget_in_INR': float(policy['Budget in INR'])
            })

        return {'recommendations': recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 