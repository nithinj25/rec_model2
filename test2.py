import requests

# Test Case 1
params = {
    'Full_Name': 'John Doe',
    'Age': 35,
    'Budget_in_INR': 1500,
    'Zip_Code': '400001',
    'Email': 'john@example.com',
    'Emergency_Services': 'true',
    'Preventive_Care_and_Screenings': 'true',
    'Hospital_Stays_and_Treatments': 'true',
    'Prescription_Medication': 'true',
    'Smoking_Status': 'Non-Smoker',
    'Pre_existing_Health_Conditions': 'false'
}

response = requests.get('https://rec-model2.onrender.com/recommend', params=params)
print(response.json())