import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialise the model
model_id = 'vinai/phobert-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained('/home4/bachpt/domain_classification/Models/saved_checkpoints/checkpoint-379600')

model.eval()
model.to('cuda')

# Loading the data
import pandas as pd
df = pd.read_csv('/home4/bachpt/domain_classification/Data/Test/All_test_segmented.csv')
df['pred_label'] = ''
df['softmax'] = ''

import torch.nn.functional as F

df_text = df['text']
predictions = []

for datum in df_text:
    inputs = tokenizer(datum, return_tensors='pt', truncation=True, max_length=256).to('cuda')
    

    with torch.no_grad():
        logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        
        model.config.id2label = {0: 'Chat', 1: 'Uni', 2: 'Others'}
        model.config.id2label[predicted_class_id]

        predictions.append(predicted_class_id)
    
df['pred_label'] = predictions
df.to_csv('/home4/bachpt/domain_classification/Pho_Bert_pred.csv', index=False)