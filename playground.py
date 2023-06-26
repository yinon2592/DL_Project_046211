import pandas as pd
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained sentiment classification model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 1: Dataset Preparation
file_path="data/training.1600000.processed.noemoticon.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
df = df[[0, 5]]
df.columns = ['label', 'text']
df = df.sample(100, random_state=1)
df['label'] = df['label'].replace({0: 'negative', 2: 'neutral', 4: 'positive'})
# Drop the rows with 'neutral' sentiment
df = df[df['label'] != 'neutral']
print(df.label.value_counts())
print(df.sample(5))