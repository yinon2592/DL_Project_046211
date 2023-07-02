from miscs import *
# import openai
IS_DEBUG = False

model_name = 'gpt2-xl'  # optional model names: gpt2-xl, gpt2-large, gpt2-medium, gpt2
comment = 'zero_shot_CoT_prompt_' # a comment to describe the kind of prompt being used
model, tokenizer = get_model_and_tokenizer(model_name)

ds_df = get_clean_tweets_ds(file_path = "data/test_data.csv").reset_index(drop=True)[0:5000]
# ds_df = get_clean_tweets_ds().sample(10, random_state=seed).reset_index(drop=True)
sanity_df = ds_df.sample(5, random_state=seed).reset_index(drop=True)

# different kinds of prompts, can be used all at once, or one at a time using the comments
manual_prompts = [
    # "Q: between positive and negative, what is the sentiment of the next sentence: {text}. A: the sentiment is:",
    # "{text}. was the previous text positive or negative? answer in one word:",
    # "{text}. Q: was the previous text positive or negative? answer in one word: A:"
    # "I love this movie. was the previous text positive or negative? answer in one word: positive.\n"
    # "I hate this movie. was the previous text positive or negative? answer in one word: negative.\n"
    # "{text}. was the previous text positive or negative? answer in one word:"
    "{text}. was the previous text positive or negative? answer in one word: let's think step by step:"
    ]

prompts = create_prompts(manual_prompts, add_default_prompts=False)

input_df = sanity_df if IS_DEBUG else ds_df

res_df = generate_sentences(input_df, model, tokenizer, prompts, max_answer_length=10, out_csv_name=comment + model_name + '_test_' + 'generative_cls', save_results=True)
output_df_name = comment + model_name + '_generated_cls_sanity_df' if IS_DEBUG else comment + model_name + '_test_' + 'generative_cls'
if IS_DEBUG:
    res_df.to_csv('data/{}.csv'.format(output_df_name), index=False)

# few shots example:
# ["I love this movie", "I hate this movie", "I don't know how I feel about this movie"]

# openai.api_key = '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'

# response = openai.Completion.create(
#   model="text-davinci-003",
#   prompt="say hello"
# )
# print(response)