import requests

url = "https://rec-model2.onrender.com/recommend"
data = {
    "Full_Name": "John Doe",
    "Age": 35,
    "Budget_in_INR": 150000,
    "Zip_Code": "400001",
    "Email": "john@example.com",
    "Emergency_Services": True,
    "Preventive_Care_and_Screening": True,
    "Hospital_Stays_and_Treatments": False,
    "Prescription_Medication": True,
    "Smoking_Status": "Non-Smoker",
    "Pre_existing_Health_Conditions": False
}

response = requests.post(url, json=data)
print(response.json())