# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:27:06 2019

@author: N827941
"""
# Import the libraries
import math
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm

# Load the data
df = pd.read_csv('spamsms-1.csv', encoding = 'latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
df.info()
df.head

# Exploring and Preparing the data
sns.countplot('type', data = df)

# Prepare the data by splitting the text documents into words and also create indicator feature for frequent words
#Data preparation – splitting text documents into words
def text_process(x):
    return x.split()
bow_tranformer = CountVectorizer(analyzer = text_process).fit(df['text'])
print(len(bow_tranformer.vocabulary_))
#y = df['type']
#X = df['text']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
#
## Data preparation – creating indicator features for frequent words
#messages_bow = bow_tranformer.transform(X_train)
#tfidf_tranformer = TfidfTransformer().fit(messages_bow)
#messages_tfidf = tfidf_tranformer.transform(messages_bow)
#print(messages_tfidf.shape)

# Create training and test datasets
df['length'] = df['text'].apply(lambda x: len(x))
df = df[df['length'] > 0]
df.info()
X_train = df[:4168]['text']
y_train = df[:4168]['type']
X_test = df[4168:]['text']
y_test = df[4168:]['type']

messages_bow = bow_tranformer.transform(X_train)
tfidf_tranformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_tranformer.transform(messages_bow)
print(messages_tfidf.shape)

# Train model on the data
spam_detect_model = MultinomialNB().fit(messages_tfidf, y_train)

#Evaluate the model
messages_bow = bow_tranformer.transform(X_test)
tfidf_tranformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_tranformer.transform(messages_bow)
print(messages_tfidf.shape)

# Predict the model
y_pred = spam_detect_model.predict(messages_tfidf)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))

df_table = confusion_matrix(y_test, y_pred)
a = (df_table[0,0] + df_table[1,1]) / (df_table[0,0] + df_table[0,1] + df_table[1,0] + df_table[1,1])
p = (df_table[1,1]) / (df_table[1,1] + df_table[0,1])
r = (df_table[1,1]) / (df_table[1,1] + df_table[1,0])
f = (2 * p * r) / (p + r)

print("accuracy : ",round(a,2))
print("precision: ",round(p,2))
print("recall   : ",round(r,2))
print("F1 score : ",round(f,2))

    
