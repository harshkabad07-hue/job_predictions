import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
print("Loading data...")
try:
    df = pd.read_csv('ds_salaries.csv')
except FileNotFoundError:
    print("Error: ds_salaries.csv not found.")
    exit(1)

# Filter for Indian region only
df = df[(df['employee_residence'] == 'IN') & (df['company_location'] == 'IN')]

# Add mock 'it_skills'
np.random.seed(42)
skills_list = [
    'Python, SQL, Excel', 
    'Python, Machine Learning, Deep Learning', 
    'Data Analysis, Tableau, PowerBI', 
    'Cloud (AWS/GCP), MLOps', 
    'R, Statistics, Python'
]
df['it_skills'] = np.random.choice(skills_list, size=len(df))

# Adjust salary based on skills to make the model learn it
skill_multipliers = {
    'Data Analysis, Tableau, PowerBI': 1.0,
    'Python, SQL, Excel': 1.2,
    'R, Statistics, Python': 1.1,
    'Python, Machine Learning, Deep Learning': 1.5,
    'Cloud (AWS/GCP), MLOps': 1.6
}
df['salary_multiplier'] = df['it_skills'].map(skill_multipliers)

# Convert salary to INR (assuming 1 USD = 83 INR) and apply multiplier
df['salary_in_inr'] = df['salary_in_usd'] * 83 * df['salary_multiplier']

# Drop redundant or target-leaking columns
cols_to_drop = ['salary', 'salary_currency', 'salary_in_usd', 'Unnamed: 0', 
                'employee_residence', 'company_location', 'salary_multiplier']
for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

# Features and target
X = df.drop('salary_in_inr', axis=1)
y = df['salary_in_inr']

# Identify categorical and numerical columns
categorical_features = ['experience_level', 'employment_type', 'job_title', 
                        'company_size', 'it_skills']
numeric_features = ['work_year', 'remote_ratio']

# Keep only existing columns
categorical_features = [col for col in categorical_features if col in X.columns]
numeric_features = [col for col in numeric_features if col in X.columns]

X = X[categorical_features + numeric_features]

# Preprocessing for numerical data
numeric_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
print("Training model...")
clf.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = clf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save the model and unique values for UI dropdowns
print("Saving model and metadata...")
joblib.dump(clf, 'salary_prediction_model.pkl')

# Save unique categorical values for the Streamlit UI to use
metadata = {
    'experience_level': X['experience_level'].unique().tolist() if 'experience_level' in X.columns else [],
    'employment_type': X['employment_type'].unique().tolist() if 'employment_type' in X.columns else [],
    'job_title': sorted(X['job_title'].unique().tolist()) if 'job_title' in X.columns else [],
    'it_skills': sorted(X['it_skills'].unique().tolist()) if 'it_skills' in X.columns else [],
    'company_size': X['company_size'].unique().tolist() if 'company_size' in X.columns else [],
    'work_year': sorted(X['work_year'].unique().tolist()) if 'work_year' in X.columns else [],
    'remote_ratio': sorted(X['remote_ratio'].unique().tolist()) if 'remote_ratio' in X.columns else [],
}
joblib.dump(metadata, 'model_metadata.pkl')

print("Done!")
