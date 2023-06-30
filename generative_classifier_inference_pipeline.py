from miscs import *
# import openai


ds_df = get_clean_tweets_ds()

model_name = 'gpt2-large'
model, tokenizer = get_model_and_tokenizer(model_name)

sanity_df = ds_df.sample(5, random_state=seed).reset_index(drop=True)

# manual_prompts = ["Im having a great day! was the previous text positive or negative? answer in one word:"]
manual_prompts = ["{text}. was the previous text positive or negative? answer in one word:"]

prompts = create_prompts(manual_prompts, add_default_prompts=False)

res_df = generate_sentences(sanity_df, model, tokenizer, prompts, max_answer_length=3)
res_df.to_csv('data/generative_cls_res_df.csv', index=False)

pd.set_option('display.max_colwidth', 50)
print('\n\n' + '-' * 20 + ' Sanity Check ' + '-' * 20)
print(sanity_df)
print(res_df)

# few shots example:
# ["I love this movie", "I hate this movie", "I don't know how I feel about this movie"]

# openai.api_key = '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'

# response = openai.Completion.create(
#   model="text-davinci-003",
#   prompt="say hello"
# )
# print(response)