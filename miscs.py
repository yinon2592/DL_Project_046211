import re
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

seed = 1

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove usernames
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # Remove special characters
    # Remove newlines and multiple whitespaces
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters and punctuations
    text = re.sub(r'[^\w\s]', '', text)

    text = text.lower().strip()
    return text

def get_clean_tweets_ds(file_path = "data/training.1600000.processed.noemoticon.csv", verbose = False):
    # Step 1: Dataset Preparation
    df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
    df = df[[0, 5]]
    df.columns = ['label', 'text']
    df = df.sample(100, random_state=1)
    df['label'] = df['label'].replace({0: 'negative', 2: 'neutral', 4: 'positive'})
    # Drop the rows with 'neutral' sentiment
    df = df[df['label'] != 'neutral']
    if verbose:
        print(df.label.value_counts())
        print(df.sample(5, random_state=seed))

    # Step 2: Data Preprocessing - clean the text
    df['text'] = df['text'].apply(clean_text)
    df['generated_sentence'] = ""

    return df

def get_model_and_tokenizer(model_name):
    # Step 3: Model Loading
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    return model, tokenizer

# Step 4: Sentence Generation
def generate_sentences(ds_df, model, tokenizer, prompts = []):
    # prompts = input prompts that containes {text} for where the text should be inserted,
    # {sentiment} for the sentiment, and {opposite_sentiment} for the opposite sentiment
    # Step 4: Sentence Generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, row in ds_df.iterrows():
            sentiment = row['label']
            opposite_sentiment = 'positive' if sentiment == 'negative' else 'negative'  # Determine the opposite sentiment
            input_prompt = f"rephrase the sentence to {opposite_sentiment} sentiment: {row['text']}"
            input_ids = tokenizer.encode(input_prompt, add_special_tokens=True, return_tensors='pt').to(device)

            # Check if input_ids is not None and has a valid shape
            if input_ids is not None and input_ids.shape[-1] > 0:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=input_ids.ne(0),  # Pass attention mask to the model (assuming pad_token_id is 0)
                    max_length=100,
                    num_return_sequences=1,
                    pad_token_id=0  # Set pad_token_id to 0
                )
                generated_sentence = tokenizer.decode(generated_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

                ds_df.at[i, 'generated_sentence'] = generated_sentence
            else:
                ds_df.at[i, 'generated_sentence'] = ""  # Assign an empty string if input_ids is None or has shape (0, )

    return ds_df

def create_model_querry(prompt, text, original_sentiment):
    sentiment = original_sentiment
    opposite_sentiment = 'positive' if sentiment == 'negative' else 'negative'  # Determine the opposite sentiment
    if '{sentiment}' in prompt and '{opposite_sentiment}' in prompt:
        input_prompt = prompt.format(text=text, sentiment=sentiment, opposite_sentiment=opposite_sentiment)
    elif '{sentiment}' in prompt and '{opposite_sentiment}' not in prompt:
        input_prompt = prompt.format(text=text, sentiment=sentiment)
    elif '{sentiment}' not in prompt and '{opposite_sentiment}' in prompt:
        input_prompt = prompt.format(text=text, opposite_sentiment=opposite_sentiment)
    else:
        input_prompt = prompt.format(text=text, sentiment=sentiment, opposite_sentiment=opposite_sentiment)

    return input_prompt

def create_prompts(preset_prompts = [], verbose = False):
    # preset_prompts: a list of prompts that contains the following strings in it for the text and possibly the existing sentiment
    tail_prompt_list = ['. keep the original sentence meaning', '\nlets think step by step: ', '']
    number_changeable_words = ['1', '2', '3']
    # QAs = [('Q: ', 'A: '), ('', '')]

    prompts = preset_prompts

    for changeable_words in number_changeable_words:
        for tail_prompt in tail_prompt_list:
            head_prompt = 'change {changeable_words}'.format(
                changeable_words=changeable_words) + ' words to change the following sentence sentiment from {sentiment} to {opposite_sentiment}. the sentence is: '
            str = head_prompt + '{text}' + tail_prompt
            prompts.append(str)
            if verbose:
                print(str)

    return prompts