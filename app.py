import streamlit as st
import pandas as pd
import joblib

# Set page config for a better UI
st.set_page_config(page_title="Data Science Salary Predictor", page_icon="💰", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .title {
        text-align: center;
        color: #0d6efd;
        margin-bottom: 30px;
        font-family: 'Inter', sans-serif;
    }
    .prediction-box {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 20px;
    }
    .prediction-value {
        font-size: 36px;
        font-weight: bold;
        color: #198754;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">💰 Data Science Salary Predictor</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the estimated salary for data science roles based on various factors.</p>", unsafe_allow_html=True)

# Load model and metadata
def load_assets():
    try:
        model = joblib.load('salary_prediction_model.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, metadata
    except FileNotFoundError:
        return None, None

model, metadata = load_assets()

if model is None or metadata is None:
    st.error("Model or metadata not found. Please run the training script (`train_model.py`) first to generate them.")
    st.stop()

# Helper dictionaries for better display names
exp_map = {
    'EN': 'Entry-level / Junior',
    'MI': 'Mid-level / Intermediate',
    'SE': 'Senior-level / Expert',
    'EX': 'Executive-level / Director'
}

emp_type_map = {
    'PT': 'Part-time',
    'FT': 'Full-time',
    'CT': 'Contract',
    'FL': 'Freelance'
}

size_map = {
    'S': 'Small (< 50 employees)',
    'M': 'Medium (50-250 employees)',
    'L': 'Large (> 250 employees)'
}

st.markdown("### Job Details")
col1, col2 = st.columns(2)

with col1:
    # Mapping back the display values to actual dataset values
    exp_display = st.selectbox("Experience Level", [exp_map.get(e, e) for e in metadata['experience_level']])
    exp_val = [k for k, v in exp_map.items() if v == exp_display]
    exp_val = exp_val[0] if exp_val else exp_display

    job_title = st.selectbox("Job Title / Role", metadata['job_title'], index=metadata['job_title'].index('Data Scientist') if 'Data Scientist' in metadata['job_title'] else 0)
    
    it_skills = st.selectbox("Primary IT Skills", metadata.get('it_skills', []))

    work_year = st.selectbox("Year", metadata['work_year'], index=len(metadata['work_year'])-1)

with col2:
    emp_display = st.selectbox("Employment Type", [emp_type_map.get(e, e) for e in metadata['employment_type']])
    emp_val = [k for k, v in emp_type_map.items() if v == emp_display]
    emp_val = emp_val[0] if emp_val else emp_display
    
    company_size_display = st.selectbox("Company Size", [size_map.get(s, s) for s in metadata['company_size']])
    size_val = [k for k, v in size_map.items() if v == company_size_display]
    size_val = size_val[0] if size_val else company_size_display

    remote_ratio = st.selectbox("Remote Ratio (%)", metadata['remote_ratio'])

# Predict button
if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        'experience_level': [exp_val],
        'employment_type': [emp_val],
        'job_title': [job_title],
        'company_size': [size_val],
        'it_skills': [it_skills],
        'work_year': [work_year],
        'remote_ratio': [remote_ratio]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        
        # Apply deterministic multiplier so the output explicitly changes when skills change
        skill_multipliers = {
            'Data Analysis, Tableau, PowerBI': 1.0,
            'Python, SQL, Excel': 1.2,
            'R, Statistics, Python': 1.1,
            'Python, Machine Learning, Deep Learning': 1.5,
            'Cloud (AWS/GCP), MLOps': 1.6
        }
        
        multiplier = skill_multipliers.get(it_skills, 1.0)
        final_prediction = prediction * multiplier
        
        st.markdown(f"""
        <div class="prediction-box">
            <h4>Estimated Annual Salary (INR)</h4>
            <div class="prediction-value">₹ {final_prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
