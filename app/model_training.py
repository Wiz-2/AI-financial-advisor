import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW
from app.data_preparation import prepare_data, tokenize_data
from sklearn.metrics import accuracy_score

def train_model():
    train_df, test_df, label_encoder = prepare_data()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))
    
    train_encodings = tokenize_data(train_df['description'].tolist(), tokenizer)
    test_encodings = tokenize_data(test_df['description'].tolist(), tokenizer)
    
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], 
                                  torch.tensor(train_df['category_encoded'].tolist()))
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], 
                                 torch.tensor(test_df['category_encoded'].tolist()))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    return model, tokenizer, label_encoder, accuracy
