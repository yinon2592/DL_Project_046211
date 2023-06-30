import re
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from transformers import BertTokenizer, TFBertModel

seed = 5
VERBOSE=False


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
    if 'gpt2' in model_name:
        configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        configuration.pad_token_id = tokenizer.eos_token_id + 1
        model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
        # configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
        # model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
        # tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    elif model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = TFBertModel.from_pretrained(model_name)

    return model, tokenizer


def generate_sentences(df, model, tokenizer, prompts=[], max_answer_length=100, verbose=VERBOSE):
    res_dicts = []

    # prompts = input prompts that contains {text} for where the text should be inserted,
    # {sentiment} for the sentiment, and {opposite_sentiment} for the opposite sentiment
    # Step 4: Sentence Generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, row in df.iterrows():
            for prompt in prompts:
                input_prompt = create_model_query(prompt, text=row['text'], original_sentiment=row['label'])
                input_ids = tokenizer.encode(input_prompt, add_special_tokens=True, return_tensors='pt').to(device)

                # Check if input_ids is not None and has a valid shape
                if input_ids is not None and input_ids.shape[-1] > 0:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        # attention_mask=input_ids.ne(0),  # Pass attention mask to the model (assuming pad_token_id is 0)
                        max_length=max_answer_length,
                        num_return_sequences=1
                        # pad_token_id=0  # Set pad_token_id to 0
                    )
                    generated_sentence = tokenizer.decode(generated_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                    if verbose:
                        print('in function generate_sentences')
                        print('i = ', i)
                        print('text = ', row['text'])
                        print('prompt = ', prompt)
                        print('generated sentence = ', generated_sentence)
                else:
                    generated_sentence = ""

                d = {'text': row['text'], 'label':  row['label'], 'prompt': prompt, 'input_prompt': input_prompt,
                     'generated_sentence': generated_sentence, 'generated_sentence_label': ''}
                res_dicts.append(d)

    return pd.DataFrame(res_dicts)


def create_model_query(prompt, text, original_sentiment, verbose=VERBOSE):
    if verbose:
        print('in function create_model_query')
        print('prompt = ', prompt)
        print('text = ', text)
        print('original sentiment = ', original_sentiment)

    sentiment = original_sentiment
    opposite_sentiment = 'positive' if sentiment == 'negative' else 'negative'  # Determine the opposite sentiment
    if '{sentiment}' in prompt and '{opposite_sentiment}' in prompt:
        input_prompt = prompt.format(text=text, sentiment=sentiment, opposite_sentiment=opposite_sentiment)
    elif '{sentiment}' in prompt and '{opposite_sentiment}' not in prompt:
        input_prompt = prompt.format(text=text, sentiment=sentiment)
    elif '{sentiment}' not in prompt and '{opposite_sentiment}' in prompt:
        input_prompt = prompt.format(text=text, opposite_sentiment=opposite_sentiment)
    elif '{sentiment}' not in prompt and '{opposite_sentiment}' not in prompt:
        input_prompt = prompt.format(text=text)
    else:
        input_prompt = prompt.format(text=text, sentiment=sentiment, opposite_sentiment=opposite_sentiment)

    if verbose:
        print('model input prompt = ', input_prompt)

    return input_prompt


def create_prompts(manual_prompts, add_default_prompts, verbose=VERBOSE):
    # manual_prompts: can be empty if only the default prompts are wanted.
    # a list of prompts that contains the following strings in it for the text and possibly the
    # existing sentiment
    tail_prompt_list = ['. keep the original sentence meaning', '\nlets think step by step: ', '']
    number_changeable_words = ['1', '2', '3']
    # QAs = [('Q: ', 'A: '), ('', '')]
    prompts = []
    for p in manual_prompts:
        if '{text}' in p:
            prompts.append(p)
        else:
            if verbose:
                print('manual prompt ' + p + 'doesnt contain "{text}", therefor it was left out')

    if add_default_prompts:
        for changeable_words in number_changeable_words:
            for tail_prompt in tail_prompt_list:
                head_prompt = 'change {changeable_words}'.format(
                    changeable_words=changeable_words) + 'words to change the following sentence sentiment from {' \
                                                         'sentiment} to {opposite_sentiment}. the sentence is:'
                s = head_prompt + '{text}' + tail_prompt
                prompts.append(s)
                if verbose:
                    print('in function: create_prompts')
                    print(s)

    return prompts
