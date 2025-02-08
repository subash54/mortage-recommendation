import pandas as pd
import numpy as np
import faiss
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.preprocessing import StandardScaler
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Generate Synthetic Mortgage Data
def generate_synthetic_data(num_records=1000):
    np.random.seed(42)
    
    banks = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Citibank", "US Bank", "PNC Bank", "Truist", "Capital One", "TD Bank", "Goldman Sachs"]
    credit_scores = np.random.randint(500, 850, num_records)
    loan_amounts = np.random.randint(100000, 1000000, num_records)
    salaries = np.random.randint(30000, 200000, num_records)
    down_payments = np.random.randint(5000, 200000, num_records)
    loan_terms = np.random.choice([15, 30], num_records)
    interest_rates = np.random.uniform(2.5, 7.5, num_records)
    bank_names = np.random.choice(banks, num_records)
    
    df = pd.DataFrame({
        "Credit Score": credit_scores,
        "Loan Amount": loan_amounts,
        "Salary": salaries,
        "Down Payment": down_payments,
        "Loan Term": loan_terms,
        "Interest Rate": interest_rates,
        "Bank Name": bank_names
    })
    return df

# Generate the dataset
data = generate_synthetic_data()

# 2. Preprocess Data for FAISS
scaler = StandardScaler()
embeddings = scaler.fit_transform(data[["Credit Score", "Loan Amount", "Salary", "Down Payment", "Loan Term"]])

# Create FAISS index
d = embeddings.shape[1]  # Number of features
index = faiss.IndexFlatL2(d)
index.add(embeddings.astype(np.float32))

# 3. Function to Find Top Banks Based on User Input
def get_top_banks(user_input):
    user_vector = scaler.transform(np.array([user_input]))
    _, indices = index.search(user_vector.astype(np.float32), 5)  # Get top 5 similar entries
    similar_data = data.iloc[indices[0]]
    
    # Pass data to LLM to determine the best bank
    bank_choices = "\n".join(
    f"{row['Bank Name']} - Interest Rate: {row['Interest Rate']:.2f}%, "
    f"Credit Score: {row['Credit Score']}, Loan Amount: ${row['Loan Amount']}, "
    f"Salary: ${row['Salary']}, Down Payment: ${row['Down Payment']}, Loan Term: {row['Loan Term']} years"
    for _, row in similar_data.iterrows())
    return bank_choices, similar_data

# Streamlit UI
st.title("Mortgage Recommendation System")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=750)
loan_amount = st.number_input("Loan Amount ($)", min_value=50000, max_value=1000000, value=500000)
salary = st.number_input("Annual Salary ($)", min_value=30000, max_value=500000, value=120000)
down_payment = st.number_input("Down Payment ($)", min_value=5000, max_value=500000, value=50000)
loan_term = st.radio("Loan Term (years)", [15, 30])

if st.button("Find Best Mortgage Offers"):
    user_query = [credit_score, loan_amount, salary, down_payment, loan_term]
    bank_choices, similar_data = get_top_banks(user_query)
    
    # LLM integration
    prompt_template = PromptTemplate(
        input_variables=["banks"],
        template="Based on the following mortgage options, suggest the top 3 best bank: {banks}. Consider interest rate and reliability and give the interest rate as well in response."
    )
    llm = OpenAI(streaming=True)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # st.subheader("Recommended Mortgage Options:")
    # st.text(bank_choices)
    st.subheader("AI Recommendation:")
    
    # Streaming response
    response = chain.stream({"banks": bank_choices})

    response_text = ""
    for chunk in response:
        if isinstance(chunk, dict) and "text" in chunk:
            st.write( chunk["text"])
        else:
            st.write(str(chunk))
        # st.write(response_text)