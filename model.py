import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import io

# Function to convert CSV data to DataFrame
def load_data_from_csv_content(csv_content):
    return pd.read_csv(io.StringIO(csv_content))

# Mock CSV data - in a real scenario, this would be read from a file
# For this example, I'll use a simplified version of the data
csv_content = """Full Name,Insurance Company,Policy Name,Age,Budget in INR,Zip Code,Email,Emergency Services,Preventive Care and Screening,Hospital Stays and Treatments,Prescription Medication,Smoking Status,Pre-existing Health Conditions
John Doe,Max Life,Max Smart Plan,27,300000,400056,johndoe@gmail.com,False,False,False,True,Non-Smoker,False
William Harris,SBI Life Insurance,New Jeevan Pragati,56,60000,411005,williamharris@gmail.com,False,False,True,True,Non-Smoker,False
John Doe,Bajaj Allianz General Insurance,Critical Illness Insurance,29,250000,110067,johndoe@gmail.com,False,True,True,False,Non-Smoker,False
Mrs. Matilda N. Lemon,Bajaj Allianz Life Insurance Company Ltd,Bajaj Life Smart Protect Plan,59,50000,357001,matilda.lemon@gmail.com,True,True,True,True,Non-Smoker,False
Mike Brown,National Insurance Company Limited,Term Plan With Return Of Premium,30,200000,400001,mike1991@gmail.com,False,True,False,False,Smoker,True"""

# Clean and process the data
def clean_and_process_data(df):
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Fix the Insurance Company column for problematic entries
    problematic_patterns = ["[{", "[name]", "[", "30,"]
    mask = df['Insurance Company'].apply(lambda x: any(pattern in str(x) for pattern in problematic_patterns))
    
    # Extract company names where possible or set to 'Unknown'
    df.loc[mask, 'Insurance Company'] = df.loc[mask, 'Insurance Company'].apply(
        lambda x: x.split(',')[1].strip().replace("'", "") if isinstance(x, str) and ',' in x and not x.startswith('[name]')
        else ('Bajaj Allianz Life Insurance' if x == "[30, 'Bajaj Allianz Life Insurance']"
              else ('SBI Life Insurance Company Ltd' if x == "[99, 'SBI Life Insurance Company Ltd']"
              else 'Unknown')))
    
    # Handle budget values - some are very small which might be in lakhs
    # Assuming budgets under 1000 INR are actually in thousands or lakhs
    df['Budget in INR'] = df['Budget in INR'].apply(
        lambda x: x * 1000 if x < 1000 else x)
    
    # Convert boolean columns to boolean type
    bool_columns = ['Emergency Services', 'Preventive Care and Screening', 
                   'Hospital Stays and Treatments', 'Prescription Medication',
                   'Pre-existing Health Conditions']
    for col in bool_columns:
        df[col] = df[col].astype(bool)
    
    # Convert smoking status to boolean
    df['is_smoker'] = df['Smoking Status'] == 'Smoker'
    
    return df

def train_recommendation_model(df):
    """
    Train a recommendation model based on the insurance data
    
    Parameters:
    - df: DataFrame with insurance data
    
    Returns:
    - Trained model pipeline and processed dataframe
    """
    print("Training recommendation model...")
    
    # Clean and process the data
    processed_df = clean_and_process_data(df)
    
    # Features to consider for recommendations
    numerical_features = ['Age', 'Budget in INR']
    categorical_features = ['Insurance Company', 'Smoking Status']
    boolean_features = ['Emergency Services', 'Preventive Care and Screening', 
                        'Hospital Stays and Treatments', 'Prescription Medication',
                        'Pre-existing Health Conditions']
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Define preprocessing for boolean features
    boolean_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features)
        ])
    
    # Create the model pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=5, metric='euclidean'))
    ])
    
    # Fit the model
    pipe.fit(processed_df)
    
    print("Model training complete!")
    
    model_info = {
        'model_pipeline': pipe,
        'processed_data': processed_df,
        'features': {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'boolean': boolean_features
        }
    }
    
    return model_info

def get_recommendations(model_info, user_info, top_n=5):
    """
    Get insurance policy recommendations based on user info
    
    Parameters:
    - model_info: Dictionary with model pipeline and processed data
    - user_info: Dictionary with user information
    - top_n: Number of recommendations to return
    
    Returns:
    - List of recommended policies
    """
    # Create a dataframe for the user
    user_df = pd.DataFrame([user_info])
    
    # Process user data the same way as training data
    user_df = clean_and_process_data(user_df)
    
    # Extract model components
    pipe = model_info['model_pipeline']
    processed_df = model_info['processed_data']
    
    # Transform user data using the preprocessor
    user_features = pipe.named_steps['preprocessor'].transform(user_df)
    
    # Reshape in case we have a single sample
    if user_features.ndim == 1:
        user_features = user_features.reshape(1, -1)
    
    # Get nearest neighbors
    distances, indices = pipe.named_steps['knn'].kneighbors(user_features, n_neighbors=top_n)
    
    # Get recommendations
    recommendations = []
    for i, idx in enumerate(indices[0]):
        policy = {
            'Full Name': processed_df.iloc[idx]['Full Name'],
            'Insurance Company': processed_df.iloc[idx]['Insurance Company'],
            'Policy Name': processed_df.iloc[idx]['Policy Name'],
            'Budget in INR': processed_df.iloc[idx]['Budget in INR'],
            'Age': processed_df.iloc[idx]['Age'],
            'Features': {
                'Emergency Services': processed_df.iloc[idx]['Emergency Services'],
                'Preventive Care and Screening': processed_df.iloc[idx]['Preventive Care and Screening'],
                'Hospital Stays and Treatments': processed_df.iloc[idx]['Hospital Stays and Treatments'],
                'Prescription Medication': processed_df.iloc[idx]['Prescription Medication'],
                'Pre-existing Health Conditions': processed_df.iloc[idx]['Pre-existing Health Conditions']
            },
            'Smoking Status': processed_df.iloc[idx]['Smoking Status'],
            'Similarity Score': 1 - distances[0][i] / (1 + distances[0][i])  # Convert distance to similarity score
        }
        recommendations.append(policy)
    
    # Sort by similarity score (highest first)
    recommendations.sort(key=lambda x: x['Similarity Score'], reverse=True)
    
    return recommendations

# Function to actually train on the real data and get recommendations
def run_recommendation_model_with_real_data(csv_data_string, user_info=None):
    """
    Train the model using real CSV data and return recommendations for a user
    
    Parameters:
    - csv_data_string: String containing CSV data
    - user_info: Dictionary with user information (optional)
    
    Returns:
    - Dictionary with model info and recommendations (if user_info provided)
    """
    # Load data from CSV string
    df = pd.read_csv(io.StringIO(csv_data_string))
    
    # Print basic info about the dataset
    print(f"Loaded data with {len(df)} records")
    print(f"Insurance companies: {df['Insurance Company'].nunique()} unique values")
    print(f"Policy names: {df['Policy Name'].nunique()} unique values")
    print(f"Age range: {df['Age'].min()} to {df['Age'].max()}")
    
    # Train the model
    model_info = train_recommendation_model(df)
    
    result = {
        'model_info': {
            'num_records': len(df),
            'num_companies': df['Insurance Company'].nunique(),
            'num_policies': df['Policy Name'].nunique(),
            'model_features': model_info['features']
        }
    }
    
    # If user info is provided, get recommendations
    if user_info:
        recommendations = get_recommendations(model_info, user_info)
        result['recommendations'] = recommendations
    
    return result

# Example CSV data - this would be the actual data from the file
# Here I'm using the mock content for the example
csv_data = """Full Name,Insurance Company,Policy Name,Age,Budget in INR,Zip Code,Email,Emergency Services,Preventive Care and Screening,Hospital Stays and Treatments,Prescription Medication,Smoking Status,Pre-existing Health Conditions
John Doe,Max Life,Max Smart Plan,27,300000,400056,johndoe@gmail.com,False,False,False,True,Non-Smoker,False
William Harris,SBI Life Insurance,New Jeevan Pragati,56,60000,411005,williamharris@gmail.com,False,False,True,True,Non-Smoker,False
John Doe,Bajaj Allianz General Insurance,Critical Illness Insurance,29,250000,110067,johndoe@gmail.com,False,True,True,False,Non-Smoker,False
Mrs. Matilda N. Lemon,Bajaj Allianz Life Insurance Company Ltd,Bajaj Life Smart Protect Plan,59,50000,357001,matilda.lemon@gmail.com,True,True,True,True,Non-Smoker,False
Mike Brown,National Insurance Company Limited,Term Plan With Return Of Premium,30,200000,400001,mike1991@gmail.com,False,True,False,False,Smoker,True"""

# Example user
example_user = {
    'Full Name': 'Test User',
    'Insurance Company': 'Unknown',
    'Policy Name': 'Unknown',
    'Age': 35,
    'Budget in INR': 150000,
    'Zip Code': '400001',
    'Email': 'test@example.com',
    'Emergency Services': True,
    'Preventive Care and Screening': True,
    'Hospital Stays and Treatments': False,
    'Prescription Medication': True,
    'Smoking Status': 'Non-Smoker',
    'Pre-existing Health Conditions': False
}

# Run the model with example data
# In a real application, you would load the CSV file data here
# result = run_recommendation_model_with_real_data(csv_data, example_user)

# Function to use the model in a real application
def recommend_insurance_policies(user_info):
    """
    Recommend insurance policies for a user based on trained model
    
    Parameters:
    - user_info: Dictionary with user information
    
    Returns:
    - List of recommended policies
    """
    # Load the CSV data (in a real application)
    # with open('mock_insurance_data_100.csv', 'r') as f:
    #     csv_data = f.read()
    
    # For this example, we'll use our mock CSV data
    result = run_recommendation_model_with_real_data(csv_data, user_info)
    
    return result['recommendations']

# Example of how to call this function
# recommendations = recommend_insurance_policies(example_user)