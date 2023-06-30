import pandas as pd
import re

from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Load the pre-trained sentiment classification model and tokenizer
# model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
# sentiment_classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name)
# sentiment_classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Step 1: Dataset Preparation
# file_path="data/training.1600000.processed.noemoticon.csv"
# df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
# df = df[[0, 5]]
# df.columns = ['label', 'text']
# df = df.sample(100, random_state=1)
# df['label'] = df['label'].replace({0: 'negative', 2: 'neutral', 4: 'positive'})
# # Drop the rows with 'neutral' sentiment
# df = df[df['label'] != 'neutral']
# print(df.label.value_counts())
# print(df.sample(5))


#
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# #Sentences we want to encode. Example:
sentence = 'Im having a great day! Among positive and negative, the previous text was: '

model_name = 'gpt2'
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)



 # I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #gpt2-medium

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))


with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)
    model.eval()

    with torch.no_grad():
        input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt').to(device)

        # Check if input_ids is not None and has a valid shape
        if input_ids is not None and input_ids.shape[-1] > 0:
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=input_ids.ne(0),  # Pass attention mask to the model (assuming pad_token_id is 0)
                max_length=100,
                num_return_sequences=1,
                pad_token_id=0  # Set pad_token_id to 0
            )


            generated_sentence = tokenizer.decode(generated_ids[0].tolist(),
                                                  skip_special_tokens=True)
            print(generated_sentence)

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Instantiate the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

# Load the configuration
configuration = GPT2Config.from_pretrained('gpt2-large', output_hidden_states=False)

# Instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2-large", config=configuration)

# Set the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input prompt
prompt = "Im having a great day! was the previous text positive or negative? answer in one word:"
# prompt = 'say hello:'

# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt').to(device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=len(input_ids[0])+3, num_return_sequences=1)

# Decode the generated output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated output
print(output_text)