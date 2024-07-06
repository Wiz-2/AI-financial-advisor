from fastapi import FastAPI, File, UploadFile
from app.models import process_receipt, classify_query, accuracy
from app.financial_analysis import analyze_budget, get_investment_recommendation

app = FastAPI()

@app.post("/process_receipt")
async def process_receipt_endpoint(file: UploadFile = File(...)):
    category = await process_receipt(file)
    return {"category": category}

@app.post("/process_query")
async def process_query_endpoint(query: str):
    intent = classify_query(query)
    return {"intent": intent}

@app.get("/analyze_budget")
async def analyze_budget_endpoint(income: float):
    analysis = analyze_budget(income)
    return analysis

@app.get("/investment_recommendation")
async def investment_recommendation_endpoint(risk_tolerance: str, investment_horizon: int):
    recommendation = get_investment_recommendation(risk_tolerance, investment_horizon)
    return {"recommendation": recommendation}

@app.get("/model_accuracy")
async def model_accuracy_endpoint():
    return {"accuracy": accuracy}
