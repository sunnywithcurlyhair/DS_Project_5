# Auditor Sentiments
This is a natural language processing and text classification project, using machine learning techniques to predict whether the sentiment of an audit documentation is bad, neutral, or good.

### Business background
In the audit landscape, many companies undergo audits for different reasons. Public companies are required to be audited by the regulator annually, and other circumstances occur that private companies would also need an audit requested by different stakeholders, such as when they take out a loan or go through a capital transaction. In most cases, due to stringent independence rules and other restrictions, audit clients typically do not have access to their audit documentation, which is usually only made available to certain regulators. However, it can be helpful for the audit clients to understand the audit sentiment for different areas of the company in order to make process improvements.  The auditors or the CPA firms could employ a machine learning model to first, classify the documentation into different sentiments, and then perform downstream tasks to provide summary materials to the client regarding different areas of the audit and related sentiments.
This project is designed to develop a machine learning model that aims to accurately classify  of an audit documentation into the sentiment categories.


### Dataset and Testing Methodology
The dataset consists of audit documentations of Finnish companies in the financial market with ~3.8K sentences/data points in the training set, and ~1k in testing set. ~4.8k data points in total labeled with bad, neutral, or good sentiments. It is sourced from [huggingface](https://huggingface.co/datasets/FinanceInc/auditor_sentiment?row=1)

The auditors' texts in the dataset discuss a range of topics on the company's state of financial metrics as well as updates and trends with their current operations. In terms of word distributions, there are a lot of overlaps in high count frequency words across different classes of sentiments given the nature of the dataset (e.g. Finnish) and that it can be on similar topics with different sentiments in different context (e.g. 'net sale' and 'operating profit'). 

**Bad reviews**

![image](https://github.com/sunnywithcurlyhair/DS_Project_5/assets/151488038/eff6712b-1ab1-4c6d-bdfb-e41d68715387) 

**Neutral reviews**

![image](https://github.com/sunnywithcurlyhair/DS_Project_5/assets/151488038/a72402de-2917-4f17-a7ed-f145593be57b) 

**Good reviews**

![image](https://github.com/sunnywithcurlyhair/DS_Project_5/assets/151488038/8e8c946e-fc84-43a5-bf8e-948dbb9bcc9b)


Both rule-based vectorization (TFIDF) and machine learning word embeddings (GloVe, BERT) were explored to extract text features to find the best performing model. Through the iterations of different models, cross validaition and tuning, our understanding was confirmed that it is important for a model to learn from the context of our subject corpus to develop an accurate predictions.  

### Repository navigation
The training and testing data are downloaded from Huggingface and saved in the data folder. The notebooks should be read in the order of data_wrangling --> Wordcloud --> Text Classification (TFIDF and GloVe models) --> BERT_Tensorflow (BERT models)

### Tested Models and Performance Metrics
1.  TFIDF
  - Mutinomial Naive Bayes - Accuracy ~ 68.8%
  - Complementary Naive Bayes - Accuracy ~64.2%
2.	GloVe
  - Random Forrest - Accuracy ~69.5%
  - Support Vector Machine - Accuracy ~74.3%
  - Extreme Gradient Boosting - Accuracy ~71.2%

There is a high degree of overlap of good reviews predicted as false neutrals from the best TFIDF and GloVe embedding, regardless of the models used. From inspecting the underlying documentation examples. It was clear that some reviews are ambuigously labled, but most are ambuigous due to the lack of context. The BERT models addressed the issue largely by incorporating the context of our specific corpus. 

3.	BERT
  - Pre-trained - Accuracy ~73.1%
  - Fine-tuned - Accuracy ~86.1%

**Best BERT fine tuned model confusion matrix**

![image](https://github.com/sunnywithcurlyhair/DS_Project_5/assets/151488038/5da364d5-8e16-4ea4-b360-df274ffa4541)


### Understanding BERT

BERT stands for Bidirectional Encoder Representations from Transformers. Unlike GloVe which is a non contextual embedding, BERT it is a contextualized from deeply bidirectional learning, masking each word in the sequence and consider words come both before and after to derive contextual meanings. Taking the examples from the wordclouds, it is likely that the fine tuned BERT model learned words squences like 'net sale decrease' and 'net sale increase' differently driving improvements in accuracy.

### Conclusions
- Contexual qualities are essential to analyze financial texts. Word sequence and word meaning in a specific context matter. The fine-tuned BERT model is our best performing model.
- Next steps:
  - Training the model on additional financial texts
  - Deployment - app that takes texts input and outputs labels, data pipeline to sort documents and texts into different folders


