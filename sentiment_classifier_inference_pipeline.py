from miscs import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Load the pre-trained sentiment classification model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)

ds_df = get_clean_tweets_ds()