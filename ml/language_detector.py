"""Build a basic language detector model

We will train a classifier on text features of different languages
that represent sequences of up to N consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

The script saves the trained model to disk for later use.
"""
# Original author: Olivier Grisel
# Adapted by: Dimitar Pachov

import sys
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# The training data folder must be passed as first argument
dataset = load_files('./paragraphs')

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5, random_state=0)

# Build a an vectorizer that splits strings into sequence of 1 to n
# characters instead of word tokens
n = 3
vectorizer = TfidfVectorizer(ngram_range=(1,n), analyzer='char', use_idf=False)

# Build a vectorizer / classifier pipeline using the previous analyzer
pipeline = Pipeline([
    ('vec', vectorizer),
    ('classifier', DecisionTreeClassifier())
#    ('classifier', LogisticRegression())
    ])

# Fit the pipeline on the training set
model = pipeline.fit(docs_train, y_train)

# Predict the outcome on the testing set
y_predicted = pipeline.predict(docs_test)


# Print the classification report
print(metrics.classification_report(y_test, y_predicted,
                                    target_names=dataset.target_names))

# Print the confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)


# Save the trained model in memory/disk.
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump([model, dataset.target_names], f)




