#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import fastparquet
import pyarrow

import re
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import string
import numbers


# In[2]:


# importing training and test data
train_df=pd.read_parquet(r'drive/MyDrive/Colab Notebooks/DS_Phase_5/data/train-00000-of-00001.parquet')


# In[3]:


test_df=pd.read_parquet(r'drive/MyDrive/Colab Notebooks/DS_Phase_5/data/test-00000-of-00001.parquet')
test_df


# In[4]:


print(train_df['sentence'][1])


# In[5]:


nltk.download('punkt')


# In[6]:


# inspecting an example of initial tokenization
word_tokenize(train_df['sentence'][1])


# In[7]:


# inspecting the length of corpus and vocabulary from initial tokenization
corpus = [word_tokenize(doc) for doc in train_df['sentence']]
import itertools
flattenedcorpus_tokens = pd.Series(list(itertools.chain(*corpus)))
flattenedcorpus_tokens.shape


# In[8]:


len(flattenedcorpus_tokens.unique())


# In[9]:


# inspecting high frequency words from initial tokenization
flattenedcorpus_tokens.value_counts()[0:40]


# In[10]:


#initial tokenation results have high stopwords frequencies 
#we'd like to remove stopwords in preproccessing,however certain stopwords may be considerred key in differenciating auditors' sentiment, 
#examples below show up in a rather decent frequency in our corpus  
audit_nonstop=['under','above','below','up','down']
flattenedcorpus_tokens.value_counts()[audit_nonstop]


# In[11]:


# 's also has too high of a frequency and lack of useful meaning, to remove
flattenedcorpus_tokens.value_counts()[["'s"]]


# In[12]:


# custom stopwords to remove
audit_stopwords=stopwords.words('english')
for word in audit_nonstop:
        audit_stopwords.remove(word)
for punct in string.punctuation:
        audit_stopwords.append(punct)
audit_stopwords.append("'s")
audit_stopwords


# In[13]:


# aside from stopwords, numbers typically have a lack of useful meaning too
# inspecting tokenized example with numbers to build function in the custom preprocessing transformer later on to remove them
digits=['0','1','2','3','4','5','6','7','8','9']
for token in word_tokenize(train_df['sentence'][0]):
    print(token,token.lower()[0],token.lower()[0] in (digits))


# In[14]:


# building function to tokenize, remove stopwords, and remove tokens starting with numbers 
def pre_process(doc):
    doc_norm = [token.lower() for token in word_tokenize(doc) if (token.lower() not in audit_stopwords) and (token.lower()[0] not in digits)]
    return doc_norm


# In[15]:


corpus1=train_df['sentence'].apply(pre_process)


# In[16]:


corpus1[0]


# In[17]:


# comparing corpus and vocabulary sizes before/after removing stopwords and numbers, which are significantly reduced
flattenedcorpus_1 = pd.Series(list(itertools.chain(*corpus1)))
print(flattenedcorpus_1.shape)


# In[18]:


print(flattenedcorpus_tokens.shape)


# In[19]:


len(flattenedcorpus_1.unique())


# In[20]:


len(flattenedcorpus_tokens.unique())


# In[21]:


# next step of preprocessing is lemmatization, we will use wordnet lemmatization, 
# for this, we'd need the wordnet part of speach tags converted from nltk tags
# function to tag each nltk part of speech tag to wordnet
def wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None


# In[22]:


# creating function to lemmatize using wordnet
def lemmatize(doc_norm):
    wnl=WordNetLemmatizer()
    wn_tagged=list(map(lambda x: (x[0],wordnet_pos(x[1])),pos_tag(doc_norm)))
    lemmatized_norm=[wnl.lemmatize(token, pos) for (token, pos) in wn_tagged if pos is not None]
    return " ".join(lemmatized_norm)


# In[23]:


# inspecting examples after lemmatization
corpus2=train_df['sentence'].apply(pre_process).apply(lemmatize)
corpus2[0:5]


# In[24]:


# building custom preprocessing transformer to lower case, remove custom stopwords, and lemmatize
class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        pass
    
    def fit(self, data, y = 0):
        return self
    
    def process_doc(self, doc):
        doc_norm=pre_process(doc)
        lemmatized_norm=lemmatize(doc_norm)
        return lemmatized_norm
    def transform(self, data, y = 0):
        fully_normalized_corpus = data.apply(self.process_doc)
       
        return fully_normalized_corpus


# In[25]:


pre_proc=TextPreprocessor()
pre_processed=pre_proc.fit_transform(train_df['sentence'])


# In[26]:


pre_proc_split=[sent.split() for sent in pre_processed]


# In[27]:


# taking a look at the final corpus and vocab size
flattenedcorpus_3=pd.Series(itertools.chain(*pre_proc_split))
print(f"Final corpus contains {len(flattenedcorpus_3)} words, with {len(flattenedcorpus_3.unique())} unique values in the dictionary")


# In[ ]:




