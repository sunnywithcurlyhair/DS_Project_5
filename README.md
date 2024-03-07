# Auditor Sentiments
This is a natural language processing and text classification project, using machine learning techniques to predict whether the sentiment of an audit documentation is bad, neutral, or good.

### Business background
In the audit landscape, many companies undergo audits for different reasons. Public companies are required to be audited by the regulator annually, and other circumstances occur that private companies would also need an audit requested by different stakeholders, such as when they take out a loan or go through a capital transaction. In most cases, due to stringent independence rules and other restrictions, audit clients typically do not have access to their audit documentation, which is usually only made available to certain regulators. However, it can be helpful for the audit clients to understand the audit sentiment for different areas of the company in order to make process improvements.  The auditors or the CPA firms could employ a machine learning model to first, classify the documentation into different sentiments, and then perform downstream tasks to provide summary materials to the client regarding different areas of the audit and related sentiments.
This project is designed to develop a machine learning model that aims to accurately classify  of an audit documentation intothe sentiment categories.


### Dataset and Testing Methodology
The dataset consists of audit documentations of Finnish companies in the financial market with ~3.8K sentences/data points in the training set, and ~1k in testing set. ~4.8k data points in total labeled with bad, neutral, or good sentiments. It is sourced from [huggingface](https://huggingface.co/datasets/FinanceInc/auditor_sentiment?row=1)

The auditors' texts in the dataset discuss a range of topics on the company's state of financial metrics as well as updates and trends with their current operations. In terms of word distributions, there are a lot of overlaps in high count frequency words across different classes of sentiments given the nature of the dataset (like Finnish) and that it can be on similar topics with different sentiments in different context (like 'net sale' or 'operating profit'). 

Both rule-based vectorization and machine learning word embeddings were used to extract text features. TFIDF, then GloVe embedding and BERT embedding were explored to find the best performing model. 

### Tested Models and Performance Metrics
1.  TFIDF
  1a. 
2.	Random forrest
4.	Support Vector Machine
5.	
6.	

