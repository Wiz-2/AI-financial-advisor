import pandas as pd

def analyze_budget(income):
    df = pd.read_csv('data/mock_expenses.csv')
    df['date'] = pd.to_datetime(df['date'])
    recent_expenses = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]
    
    total_expenses = recent_expenses['amount'].sum()
    savings = income - total_expenses
    savings_rate = (savings / income) * 100 if income > 0 else 0
    
    category_totals = recent_expenses.groupby('category')['amount'].sum().to_dict()
    
    return {
        "total_expenses": total_expenses,
        "savings": savings,
        "savings_rate": savings_rate,
        "category_totals": category_totals
    }

def get_investment_recommendation(risk_tolerance, investment_horizon):
    if risk_tolerance == "low":
        if investment_horizon < 5:
            return "Consider a conservative mix of bonds and high-yield savings accounts."
        else:
            return "Consider a balanced portfolio with a mix of bonds and some low-risk stocks."
    elif risk_tolerance == "medium":
        if investment_horizon < 5:
            return "Consider a balanced mix of stocks and bonds."
        else:
            return "Consider a growth-oriented portfolio with a higher allocation to stocks."
    else:  # high risk tolerance
        if investment_horizon < 5:
            return "Consider a growth-oriented portfolio with a focus on stocks."
        else:
            return "Consider an aggressive portfolio with a high allocation to stocks and possibly some alternative investments."
