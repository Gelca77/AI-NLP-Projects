## Phishing Email Classifier ##

## Introduction
This project is part of my learning journey in Python, AI, and natural language processing (NLP).  
I’ve been following free online tutorials and beginner courses on machine learning, and I wanted to apply what I’m learning to something connected to cybersecurity.  My idea was to build a small classifier that can tell the difference between phishing messages and normal ones.

## Method
The python script does a few simple steps:
Load the data from a CSV file
Clean the text a little (lowercase, remove extra spaces)
Turn the text into numbers using TF-IDF
Train two models (Logistic Regression and Linear SVM)
Print out accuracy and evaluation results

## Data
Right now I used only a very small dataset that I wrote (data/sample.csv) with only 4 messages.  
This was just to check that the code runs from start to finish.  
Next, I plan to try a real dataset from either the SMS Spam Collection or a phishing email dataset from Kaggle.

## Results
Logistic Regression: Accuracy = 0.50  
Linear SVM: Accuracy = 0.50  

Because the dataset was so small, I split it 50/50 so both classes (phishing and legit) showed up in training and testing.  
The results don’t mean much yet but it showed that the pipeline worked.

## What I Learned
How to set up a Python project with a virtual environment  
How to install and manage packages with requirements.txt
How to load and clean text data with pandas  
How TF-IDF works to turn text into numbers  
How to train and test simple models with scikit-learn  

## Next Steps
I want to use a larger real dataset then attempt to compare how the two models perform on that data.  
Then, try more advanced NLP methods (like BERT/transformers) as I continue learning

## How to Run

pip install -r requirements.txt

python baseline_classifier.py

## Context
This project is part of me building experience in AI, NLP, and cybersecurity.  

I want to keep improving over time as I continue to learn
-Angelica Shelman


