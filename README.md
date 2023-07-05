# DL_Project_046211
## Project Goal :
  ### sentiment analysis with existing LLMs
  we performed experiments with GPT-2 different size models to assess 2 classifcation approaches: binary & genrative   

## Project Files Details:
| file | content | 
|----------|----------|
|   binary_classifier_test.ipynb            |   training 'gpt-2' model for sequence binary classification  |   
|   binary_classifier_training.ipynb        |   test our fine tuned 'gpt-2' model for sequence binary classification with 5,000 tweets |   
|   dataset_preprocessing.ipynb             |   clean and process sentiment140 dataset and divide it into 3 chunks (2 for classifiers training and 1 for test)  |   
|   generative_classifier_test.ipynb        |   test our fine tuned 'gpt-2 large' model for sequence generative classification with 5,000 tweets | 
|   generative_classifier_training.ipynb    |   training 'gpt-2 large' model for sequence generative classification  | 
|   (obsolete) opposite_sentiment_experiment.ipynb      |  generates opposite sintiment text using an off the shelf generative model  | 
|   miscs     |  contains help functions for the generative off the shelf classifiers scripts     | 
|   off_the_shelf_generative_cls_inference_pipeline.py      |  a script for using an off the shelf gpt2 model as a generative classifier     | 
|   off_the_shelf_generative_create_opposite_sentiment_sentences.py      |  a script for using an off the shelf gpt2 model as to create opposite sentiment text    | 
|   process_off_the_shelf_generative_cls_results.py      |  a script for processing the off the shelp generative classifier results    | 

 ## Project Technical Details: 
 in this project we used google-drive for:
 1) saving our 3 chunks cleaned and divided sentiment140 dataset
 2) saving the fine tuned models
 3) load the fine tuned models for testing purpose
 4) save the output generated sentences

  therefore if you want to train\test the models on your own please make sure
  to connect your goole-drive account ahead or replace the relevant paths as you wish

 ## Project Main Results
 we tested the model with 5,000 tweets and got the following results:
 | model | test accuracy | remarks
|----------|----------|----------|
|   fine tuned binary classifier           |   0.66  |   -|
|   fine tuned binary classifier           |   0.71  |   using {text} + 'was the previous text positive or negative' prompt during training|
|   off the shelf gpt2-xl generative classifier           |   0.7  |   using  "I love this movie. was the previous text positive or negative? answer in one word: positive.\nI hate this movie. was the previous text positive or negative? answer in one word: negative.\n{text}. was the previous text positive or negative? answer in one word:" prompt during inference|
|   fine tuned generative classifier           |   -  |   we tried diffrent prompts with 'gpt-2 large' model with no actual progress| 
 
## Conclusions And Future Work
we noticed that we got the best results using fine tuned with prompt binary classifier (we tried also feature extracting training method but got inferior results)
another promising direction is to use gpt2-xl backbone for binary/generative classification, after a finetune
regarding generative classifier training we assume that 'gpt-2 large' model is not big enough for this task and different modules can
be used for future experiment (such as 'GPT2ForQuestionAnswering')

## References
https://www.kaggle.com/code/abdeljalilouedraogo/twitter-sentiment-analysis-on-sentiment140-dataset
https://www.kaggle.com/code/baekseungyun/gpt-2-with-huggingface-pytorch
