import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer

def prepare_data():
    df = pd.read_csv('data/mock_expenses.csv')
    
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df, le

def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
