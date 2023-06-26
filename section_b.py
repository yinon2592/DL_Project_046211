from transformers import AutoModelForSequenceClassification, AutoTokenizer
from miscs import *

# Load the pre-trained sentiment classification model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)

ds_df = get_clean_tweets_ds()

# Step 3: Model Loading
model_name = 'gpt2'
model, tokenizer = get_model_and_tokenizer(model_name)

# Step 4: Sentence Generation
df_sanity_df = ds_df.sample(5)
res_df = generate_sentences(df_sanity_df, model, tokenizer)

pd.set_option('display.max_colwidth', 50)
print('\n\n'+ '-' * 20 + ' Sanity Check ' + '-' * 20)
print(df_sanity_df.sample(5))
print(res_df.sample(5))