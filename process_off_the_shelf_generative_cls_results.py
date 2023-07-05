from glob import glob
import pandas as pd


# get all the csv files in the data folder
def get_generative_cls_csv_files():
    return glob("data/test results/generative_cls_results/*.csv")


def get_cls_csv_files():
    return glob("data/test results/cls_results/*.csv")


def get_generated_label_from_generated_sentence(generated_sentence):
    if 'positive' in generated_sentence:
        return 'positive'
    elif 'negative' in generated_sentence:
        return 'negative'
    else:
        return 'unknown'


def get_model_name_from_file_name(file_name):
    fn = file_name.split('.')[0]
    if 'gpt2-xl' in fn:
        return 'gpt2-xl'
    elif 'gpt2-large' in fn:
        return 'gpt2-large'
    elif 'gpt2-medium' in fn:
        return 'gpt2-medium'
    elif 'gpt2' in fn:
        return 'gpt2'
    return 'unknown'


def create_generative_cls_summary_line(df, f):
    model_name = get_model_name_from_file_name(f)
    accuracy = round(df[df['label'] == df['generated_sentence_label']].shape[0] / df.shape[0], 2)
    df_no_unknown = df[df['generated_sentence_label'] != 'unknown']
    accuracy_no_unknown = round(df_no_unknown[df_no_unknown['label'] == df_no_unknown['generated_sentence_label']].shape[0] / df_no_unknown.shape[0], 2)
    prompt = df['prompt_mask'].unique()[0]
    # the prompt comment is the part in the file name that comes before the model name
    fn = f.split('.')[0]
    # the prompt comment is the part in tn that comes before 'gpt2
    prompt_comment = fn.split('gpt2')[0]
    prompt_comment = prompt_comment.split('\\')[-1]
    if len(prompt_comment) > 0:
        prompt_comment = prompt_comment[:-1] if prompt_comment[-1] == '_' else prompt_comment
    comment = ''

    pd.DataFrame([{'model': model_name, 'prompt': prompt, 'prompt_comment': prompt_comment, 'accuracy': accuracy,
                   'accuracy_no_unknown': accuracy_no_unknown, 'comment': comment}]).to_csv('data/test results/summary.csv', mode='a', index=False, header=False)


def create_cls_summary_line(df, f):
    model_name = 'gpt2'
    accuracy = round(df[df['label'] == df['predicted_label']].shape[0] / df.shape[0], 2)
    df_no_unknown = df[df['predicted_label'] != 'unknown']
    accuracy_no_unknown = round(df_no_unknown[df_no_unknown['label'] == df_no_unknown['predicted_label']].shape[0] / df_no_unknown.shape[0], 2)
    prompt = '{text}'
    prompt_comment = ''
    comment = 'non-generative cls, ' + ('was fine-tuned without prompt' if 'option_1' in f else 'was fine-tuned with prompt: "{text} was the previous text positive or negative"')

    pd.DataFrame([{'model': model_name, 'prompt': prompt, 'prompt_comment': prompt_comment, 'accuracy': accuracy,
                   'accuracy_no_unknown': accuracy_no_unknown, 'comment': comment}]).to_csv('data/test results/summary.csv', mode='a', index=False, header=False)


generative_cls_csvs = get_generative_cls_csv_files()

# create a summary df header to add lines to later on
pd.DataFrame(columns=['model', 'prompt', 'prompt_comment', 'accuracy', 'accuracy_no_unknown', 'comment']).to_csv('data/test results/summary.csv', index=False)

# for each csv file, read it in, update the column names, and save it back
for f in generative_cls_csvs:
    df = pd.read_csv(f)

    # update the column names to: text, label, prompt_mask, input_prompt, generated_sentence, generated_sentence_label
    df.columns = ['text', 'label', 'prompt_mask', 'input_prompt', 'generated_sentence', 'generated_sentence_label']

    # update the value in the generated_sentence_label column to be a function of the generated_sentence
    df['generated_sentence_label'] = df['generated_sentence'].apply(get_generated_label_from_generated_sentence)

    print(df.columns)
    print(df.head())

    # save the updated csv file
    df.to_csv(f, index=False)
    print('saved to csv')

    # create a summary line for the csv file
    create_generative_cls_summary_line(df, f)

summary_df = pd.read_csv('data/test results/summary.csv')
print(summary_df.head())


cls_csvs = get_cls_csv_files()
for f in cls_csvs:
    df = pd.read_csv(f)

    print(df.columns)
    print(df.head())

    df.to_csv(f, index=False)
    print('saved to csv')

    # create a summary line for the csv file
    create_cls_summary_line(df, f)