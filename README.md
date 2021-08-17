# sentence-level-domain-classification
Sentence-level domain classification with a feedforward neural network

NOTE: This repository is intended as a companion to the **Tutorial** section of the article [Domain Classification with Natural Language Processing](https://www.taus.net) as published on the [TAUS blog](https://blog.taus.net).

## Overview

Domain classification is the task of predicting a domain or category label given an input text. In this tutorial, we focus on sentence-level domain classification, which consists of assigning a category label to individual sentences in a data set.

## Getting started

This tutorial assumes basic knowledge of the Python programming language. The scripts in this repository use Python version 3.8.10.

  1. Clone this repository to the desired location using ``` git clone https://github.com/TAUSBV/sentence-level-domain-classification.git ```.
  2. Set up a [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)       for this project.
  3. Activate the environment and run ``` pip install -r requirements.txt ``` from the root directory in order to install the required packages.
  4. Run ``` python src/download_stanza.py ``` to download the model used for sentence splitting.
  5. Download the BBC News data set and save it into the root directory:
  
     ```
     wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
     unzip bbc-fulltext.zip
     rm bbc-fulltext.zip
     ```

## Data preprocessing

The BBC News Classification data set is a collection of 2225 English-language news documents that together cover five different domains (BUSINESS, ENTERTAINMENT, POLITICS, SPORT, and TECH) [[1]](#1). It is commonly used in text categorization research, where the main focus is usually on the document level. However, as we have already decided that we would be focusing on the sentence level for this tutorial, some preprocessing of this data set is needed.

We use [Stanza](https://stanfordnlp.github.io/stanza/index.html) to split each document in the data set into sentences. The algorithm is trained to identify sentence boundaries in text and split the contents of a file accordingly. Run ``` python src/preprocess.py "bbc/*/*.txt" ``` to split all documents in the data set into sentences. Make sure that you have downloaded the BBC data set and that it is located in a folder called ``` bbc ``` in the root directory - otherwise, this script is not going to work. Once the sentence splitter is running, it may take up to a few minutes for it to process all the files.

When this is finished, we can run ``` python src/classify.py ``` to perform the next preprocessing steps and then later the classification itself. First, we create the data set by loading the sentences and connecting them to their original labels based on which BBC folder they can be found in:

```
from utils import create_dataset
X, y = create_dataset()
```

    
Next, the labels must be encoded into a numerical representation so that the classifier can recognize them. For this, we rely on [scikit-learn's label encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) and the [to_categorical function from TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical):

```
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
y = to_categorical(le.fit_transform(y))
```


We can now proceed to splitting the data set into training, development, and test sets. The script does all the work for you, but you can also do this yourself using the [train_test_split function from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Below, we split the data set two times - first into TRAIN and DEV and then we further divide DEV into DEV and TEST. We use ``` random_state=42 ``` to make the results easily reproducible.

```
from sklearn.model_selection import train_test_split

# split data set into TRAIN, DEV, and TEST sets
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% for DEV and TEST

# split DEV further into DEV and TEST
X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.5, random_state=42)
```

The resulting sets should contain the following numbers of sentences:

| Category name | TRAIN      | DEV       | TEST      |
| ------------- | ---------- | --------- | --------- |
| BUSINESS      | 6889       | 826       | 863       |
| ENTERTAINMENT | 5416       | 702       | 712       |
| POLITICS      | 7385       | 932       | 911       |
| SPORT         | 7569       | 974       | 969       |
| TECH          | 8149       | 992       | 971       |


The next step is to convert each of our sentences into sentence embeddings, or in other words, vector representations that encode their semantics. There are many publicly available sentence embedding models out there, but for this tutorial, we use the [Universal Sentence Encoder made available on TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder/4) [[2]](#2). Find out more about sentence embeddings and their applications [in this article from the TAUS blog](https://blog.taus.net/what-are-sentence-embeddings-and-their-applications).

```
import tensorflow_hub as hub

# load embeddings model from Tensorflow Hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# convert sentences to embeddings
X_train = embed(X_train)
X_dev = embed(X_dev)
X_test = embed(X_test)
```

Note that by converting the sentences to embeddings we essentially discard the original strings. To keep the original sentences, it is recommended that you save the embeddings as new variables. 

## Classification

Having preprocessed the data, we are now ready to move on the classification. As discussed in the accompanying article, neural networks and deep learning are responsible for many major breakthroughs in recent NLP research and they have also been shown to perform well at various text classification tasks. In this tutorial, we employ a [simple feedforward neural network model](https://www.tensorflow.org/guide/keras/sequential_model) provided by TensorFlow.

We define a 3-layer network with a single hidden layer. The first layer consists of 32 nodes and the hidden layer is made up of 64. In classification tasks, the final output layer must always consist of a number of nodes that is equivalent to the number of classes in the data set, so we set this value to 5. We use the categorical cross-entropy loss function, the Adam optimizer, and accuracy as our primary evaluation metric during training.

```
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# build Sequential model with 3 layers
model = Sequential()
model.add(Dense(units=32, activation="relu"))  # input layer
model.add(Dense(units=64, activation="relu"))  # hidden layer
model.add(Dense(units=5, activation="softmax"))  # output layer, no. of units equals no. of classes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Next, we fit the model using our training data and our development set for validation. This step may take up to a couple of minutes.

```
model.fit(x=X_train, y=y_train,
          epochs=10,
          validation_data=(X_dev, y_dev),
          batch_size=32,
          verbose=1)
```

## Evaluation

The final step of the process is to evaluate the model's performance on previously unseen data - this is where the test set comes into play. We run the trained classifier on the test data and compare the results against the real labels, a.k.a. the gold standard. We can do this and generate a classification report in a few simple lines of code:

```
import numpy as np
from sklearn.metrics import classification_report

predictions = np.argmax(model.predict(X_test), axis=-1)
y_test = le.inverse_transform([np.argmax(y) for y in y_test])  # reconstruct original string labels
predictions = le.inverse_transform(predictions)
report = classification_report(y_test, predictions)
```

We can see that in terms of [precision, recall, and accuracy](https://datagroomr.com/precision-recall-and-f1-explained-in-plain-english/) our model performs rather well despite its relative simplicity. Sentences can sometimes be notoriously difficult to categorize (for example, what would you say is the domain of the sentence *It’s not good, but it is very understandable* ?), which makes the performance of this model all the more impressive.

|               | precision | recall | f1-score |
| ------------- | --------- | ------ | -------- |
| BUSINESS      | 0.83      | 0.82   | 0.83     |
| ENTERTAINMENT | 0.86      | 0.79   | 0.82     |
| POLITICS      | 0.78      | 0.83   | 0.81     |
| SPORT         | 0.86      | 0.89   | 0.88     |
| TECH          | 0.86      | 0.84   | 0.85     |
| accuracy      |           |        | 0.84     |
| macro avg     | 0.84      | 0.84   | 0.84     |
| weighted avg  | 0.84      | 0.84   | 0.84     |

## Challenge

- Clone this repository and see if you can replicate the results. Can you improve them?
- Save the test sentences along with their predicted labels to a file and inspect the results. Pay particular attention to the ones that are mislabeled by the model. Can you see any interesting patterns?

## References

<a id="1">[1]</a> 
D. Greene and P. Cunningham (2006). 
[Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering.](http://mlg.ucd.ie/files/publications/greene06icml.pdf) 
Proc. 23rd International Conference on Machine learning (ICML'06), 377--384.

<a id="2">[2]</a>
Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Céspedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil (2018).
[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
arXiv:1803.11175
