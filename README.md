# To run the project:

1. Install the requirements: pip install -r requirements.txt
2. Generate the mock data: python data/generate_mock_data.py
3. Start the FastAPI backend: uvicorn app.main:app --reload
4. In a new terminal, start the Streamlit frontend: streamlit run frontend/streamlit_app.py



This project includes:

1. Mock data generation
2. Data preparation and model fine-tuning
3. A FastAPI backend for serving predictions and financial analysis
4. A Streamlit frontend for user interaction
5. Integration of machine learning for expense categorization
6. Basic financial analysis and investment recommendations
