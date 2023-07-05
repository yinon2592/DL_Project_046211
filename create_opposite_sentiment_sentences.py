from miscs import *

ds_df = get_clean_tweets_ds()

# Step 3: Model Loading
model_name = 'gpt2'
model, tokenizer = get_model_and_tokenizer(model_name)

# create aa sanity df
sanity_df = ds_df.sample(5, random_state=seed).reset_index(drop=True)

manual_prompts = ['regardless of the the following text, say "hello world". the text is: {text}',
                  'regardless of the the following text, say "good morning world". the text is: {text}']

prompts = create_prompts(manual_prompts, add_default_prompts=True)

res_df = generate_sentences(sanity_df, model, tokenizer, prompts, max_answer_length=100)
res_df.to_csv('data/res_df.csv', index=False)

pd.set_option('display.max_colwidth', 50)
print('\n\n' + '-' * 20 + ' Sanity Check ' + '-' * 20)
print(sanity_df)
print(res_df)

# few shots example:
# ["I love this movie", "I hate this movie", "I don't know how I feel about this movie"]
