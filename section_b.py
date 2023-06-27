import pandas as pd
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
df_sanity_df = ds_df.sample(5, random_state=seed)
prompts = create_prompts()
res_df = generate_sentences(df_sanity_df, model, tokenizer, prompts)

pd.set_option('display.max_colwidth', 50)
print('\n\n'+ '-' * 20 + ' Sanity Check ' + '-' * 20)
print(df_sanity_df.sample(5, random_state=seed))
print(res_df.sample(5, random_state=seed))


# head_prompts_list2 = ['change {changeable_words} word to make the sentence have {opposite_sentiment} sentiment: '.format(changeable_words, opposite_sentiment),
#                 'rephrase the sentence to {opposite_sentiment} sentiment: '.format(opposite_sentiment),
#                 'make the sentence have {opposite_sentiment} sentiment: '.format(opposite_sentiment),
#                 'make the sentence have {opposite_sentiment} sentiment by changing a single word: '.format(opposite_sentiment),
#                 'change up to two words to make the sentence {opposite_sentiment}: '.format(opposite_sentiment),
#                 'change up to two words to flip the sentence sentiment from {sentiment} to {opposite_sentiment}: '.format(sentiment, opposite_sentiment)]

# few shots example:
# ["I love this movie", "I hate this movie", "I don't know how I feel about this movie"]