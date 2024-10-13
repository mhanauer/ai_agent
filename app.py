import streamlit as st
import pandas as pd
import numpy as np
import openai
from faker import Faker

# Set OpenAI API Key (replace with your actual key)
openai.api_key =  st.secrets["OPENAI_API_KEY"]

# Generate Simulated Claims Data
def generate_claims_data(n=100):
    faker = Faker()
    claims_data = {
        'ClaimID': [faker.uuid4() for _ in range(n)],
        'MemberID': np.random.randint(1000, 1100, n),
        'ClaimAmount': np.round(np.random.uniform(100, 5000, n), 2),
        'DateOfClaim': [faker.date_this_year() for _ in range(n)],
        'ClaimStatus': np.random.choice(['Approved', 'Pending', 'Rejected'], n)
    }
    return pd.DataFrame(claims_data)

# Generate Simulated Enrollment Data
def generate_enrollment_data(n=100):
    faker = Faker()
    enrollment_data = {
        'MemberID': np.random.randint(1000, 1100, n),
        'MemberName': [faker.name() for _ in range(n)],
        'DateOfEnrollment': [faker.date_this_decade() for _ in range(n)],
        'PlanType': np.random.choice(['HMO', 'PPO', 'EPO', 'POS'], n)
    }
    return pd.DataFrame(enrollment_data)

# Load the simulated claims and enrollment datasets
@st.cache_data
def load_data():
    claims_df = generate_claims_data()
    enrollment_df = generate_enrollment_data()
    return claims_df, enrollment_df

claims_df, enrollment_df = load_data()

# Function to process the user's question using GPT-4
def ask_question(question, claims_df, enrollment_df):
    # Combine the two datasets into a prompt
    combined_data = f"Claims Data:\n{claims_df.head(10).to_string()}\n\nEnrollment Data:\n{enrollment_df.head(10).to_string()}\n"
    prompt = f"{combined_data}\n\nUser Question: {question}\nAnswer based on the data:"

    # Call the OpenAI API to get the response from GPT-4
    response = openai.Completion.create(
        engine="gpt-4",  # Using GPT-4
        prompt=prompt,
        max_tokens=150,
        temperature=0.2
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.title("AI Agent for Simulated Claims and Enrollment Data")

st.write("This app allows you to ask questions about claims and enrollment data and get answers.")

# Display the datasets
if st.checkbox("Show Claims Data"):
    st.dataframe(claims_df)

if st.checkbox("Show Enrollment Data"):
    st.dataframe(enrollment_df)

# Input box for the user to ask questions
question = st.text_input("Ask any question about the claims or enrollment data:")

if question:
    answer = ask_question(question, claims_df, enrollment_df)
    st.write("Answer:")
    st.write(answer)
