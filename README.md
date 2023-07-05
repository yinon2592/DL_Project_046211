# DL_Project_046211
## project goal :
  sentiment analysis with existing LLMs 
  we performed experiments with GPT-2 different size models to assess 2 predication approaches: genrative and binary classifcation

## project files details:
| file | content | 
|----------|----------|
|   binary_classifier_test.ipynb            |   training 'gpt-2' model for sequence binary classification  |   
|   binary_classifier_training.ipynb        |   test our fine tuned 'gpt-2' model for sequence binary classification with 5,000 tweets |   
|   dataset_preprocessing.ipynb             |   clean and process sentiment140 dataset and divide it into 3 chunks (2 for classifiers training and 1 for test)  |   
|   generative_classifier_test.ipynb        |   test our fine tuned 'gpt-2 large' model for sequence generative classification with 5,000 tweets | 
|   generative_classifier_training.ipynb    |   training 'gpt-2 large' model for sequence generative classification  | 
|   reverse_sentiment_experiment.ipynb      |   perform reverse sentiment experiment  | 

 ## project technical details: 
 in this project we used goole-drive for:
 1) saving our 3 chunks cleaned and divided sentiment140 dataset
 2) saving the fine tuned models
 3) load the fine tuned models for testing purpose

  therefore if you want to train\test the models on your own please make sure
  to connect your goole-drive account ahead and replace the relevant paths as you wish

 

