# Mortgage Recommendation System

## Overview
This project is a **Mortgage Recommendation System** that uses **FAISS for vector search** and **OpenAI's LLM** to suggest the top two banks offering the best mortgage rates based on user attributes like **credit score, loan amount, salary, down payment, and loan term**.

## Features
- **Synthetic Data Generation**: Creates sample mortgage data for testing.
- **Vector Search with FAISS**: Finds similar mortgage offers based on user input.
- **LLM Integration**: Uses OpenAI to refine and recommend the best mortgage options.
- **Streamlit UI**: Provides a user-friendly interface for input and recommendations.

## Technologies Used
- **Python 3.11**
- **FAISS** (Facebook AI Similarity Search)
- **OpenAI API**
- **LangChain**
- **Streamlit** (for UI)
- **Pandas & NumPy** (for data processing)
- **scikit-learn** (for data preprocessing)
- **dotenv** (for managing API keys)

---

# Local Setup

## Prerequisites
Ensure you have the following installed:
- Python (>= 3.8)
- pip (Python package manager)

## Setup Instructions
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo/mortgage-recommendation.git
   cd mortgage-recommendation
   ```

2. **Create a virtual environment**:
   ```sh
   python -m venv mortage-env
   source mortage-env/bin/activate  # On macOS/Linux
   mortage-env\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```ini
   OPENAI_API_KEY=your-api-key-here
   ```

5. **Run the application**:
   ```sh
   streamlit run main.py
   ```

## Testing Sample Inputs
To test, enter values such as:
- **Credit Score**: 750
- **Loan Amount**: 500,000
- **Salary**: 120,000
- **Down Payment**: 50,000
- **Loan Term**: 30 years

## Expected Output
- **Top 2 Recommended Banks**
- **Interest Rates & Loan Terms**
- **LLM-Powered Suggestions**

---

## Troubleshooting
- **ModuleNotFoundError: No module named 'langchain_community'**
  - Solution: Run `pip install langchain-community`
- **Error: Missing OpenAI API Key**
  - Solution: Ensure `.env` is set up correctly and run `source .env`
- **FAISS Index Not Found**
  - Solution: Ensure the synthetic data generation runs properly before search.

---

## Future Enhancements
- Integrate real-time mortgage rate APIs.
- Expand to more loan types (e.g., auto loans, personal loans).
- Improve LLM prompts for better recommendation quality.

---

## Contributors
- **Subash** - Developer & Maintainer
- Open for contributions! Fork and submit PRs.

## License
MIT License

