import torch
from PIL import Image
import pytesseract
from app.model_training import train_model

model, tokenizer, label_encoder, accuracy = train_model()

async def process_receipt(file):
    image = Image.open(file.file)
    text = pytesseract.image_to_string(image)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_category_id = outputs.logits.argmax().item()
    predicted_category = label_encoder.inverse_transform([predicted_category_id])[0]
    
    return predicted_category

def classify_query(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_category_id = outputs.logits.argmax().item()
    intent = label_encoder.inverse_transform([predicted_category_id])[0]
    return intent
