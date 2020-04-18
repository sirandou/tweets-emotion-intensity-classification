# SemEval-2018 Task 1: Affect in Tweets, EI-oc subtask (an emotion intensity ordinal classification task) for English tweets

Given a tweet and an emotion E, we classify the tweet into one of four ordinal classes of intensity of E that best represents the mental state of the tweeter.
Different sentiments are anger, fear, joy, and sadness.

https://competitions.codalab.org/competitions/17751

This project uses different variations of BERT.

# Setup

1. Please make sure you have pip, venv and python installed.
2. Create the environment using these. 

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

3. Download GloVe vectors for twitter and add the file glove.twitter.27B.200d.txt to the data/models/glove folder.

4. Download the pretrained models from https://drive.google.com/open?id=1ZfqgYfAG9mwe-gU0qF7up5kswMhVB8VE into data/models/uncased-bert/models

5. If you want to train the models, run
```
python3 main.py --train True --train_model <Model you want to train>
```

Models are: 'lstm', 'bilstm', 'bert_uncased', 'bert_cased', 'bert_hybrid', and 'bert_ordinal'

6. To get the predicted intensities, and to evaluate them, make sure that the models are placed in the data/models/uncased-bert/models folder, and then run:
```
python3 main.py
```
The results generated will be for the best performing model.

