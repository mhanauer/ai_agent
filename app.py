import streamlit as st
import pandas as pd
import numpy as np
import openai

# Set OpenAI API Key
api_key = st.secrets["OPENAI_API_KEY"]

# Set up OpenAI client
client = openai.OpenAI(api_key=api_key)

# Generate Simulated Enrollment Data
def generate_enrollment_data(n=100):
    member_ids = np.random.choice(range(1000, 1100), n, replace=False)
    enrollment_data = {
        'MemberID': member_ids,
        'MemberName': [f"Member_{str(i).zfill(4)}" for i in range(1, n+1)],
        'DateOfEnrollment': pd.date_range(start='2015-01-01', periods=n).strftime('%Y-%m-%d'),
        'PlanType': np.random.choice(['HMO', 'PPO', 'EPO', 'POS'], n)
    }
    return pd.DataFrame(enrollment_data)

# Generate Simulated Claims Data
def generate_claims_data(enrollment_df, n=100):
    # Use MemberIDs from enrollment_df
    member_ids = np.random.choice(enrollment_df['MemberID'], n, replace=True)
    claims_data = {
        'ClaimID': [f"C{str(i).zfill(4)}" for i in range(1, n+1)],
        'MemberID': member_ids,
        'ClaimAmount': np.round(np.random.uniform(100, 5000, n), 2),
        'DateOfClaim': pd.date_range(start='2023-01-01', periods=n).strftime('%Y-%m-%d'),
        'ClaimStatus': np.random.choice(['Approved', 'Pending', 'Rejected'], n)
    }
    return pd.DataFrame(claims_data)

# Load the simulated claims and enrollment datasets
@st.cache_data
def load_data():
    enrollment_df = generate_enrollment_data()
    claims_df = generate_claims_data(enrollment_df)
    return claims_df, enrollment_df

claims_df, enrollment_df = load_data()

# Function to process the user's question using GPT-4
def ask_question(question, claims_df, enrollment_df):
    # Combine the two datasets into a prompt
    combined_data = (
        f"Claims Data:\n{claims_df.head(10).to_string(index=False)}\n\n"
        f"Enrollment Data:\n{enrollment_df.head(10).to_string(index=False)}\n"
    )
    prompt = f"{combined_data}\n\nUser Question: {question}\nAnswer based on the data:"

    # Call the OpenAI Chat API using the client object
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in healthcare claims and enrollment data. Use the data provided to answer the user's question."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        # Extract and return the response text
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

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
