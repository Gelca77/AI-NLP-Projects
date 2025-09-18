# AI-NLP-Projects  

## Introduction  
This repository contains my beginner projects in artificial intelligence and natural language processing (AI/NLP).  
I am teaching myself how to apply Python, machine learning, and text analysis to real problems, especially those that connect to cybersecurity and human psychology and behavior.  

My goal is to build up from small experiments into more advanced projects that will support my long-term research vision of using AI and NLP to defend against social engineering attacks.  

## Projects  

### 1. Smishing Urgency Detector  
- **Goal:** Classify text messages as either phishing (smishing) or legitimate by focusing on fear and urgency words (such as “immediately,” “suspended,” “avoid permanent deactivation”).  
- **Techniques:**  
  - Python (pandas, scikit-learn)  
  - TF-IDF feature extraction  
  - Logistic Regression model  
- **Results:**  
  - Achieved working classification with a small dataset of 20 sample messages.  
  - Showed which words and features most strongly indicate phishing (like “urgent,” “immediately,” and “account suspended”).  
  - Accuracy ~80% on the tiny test set (not representative yet, but good proof-of-concept).  
- **Next Steps:**  
  - Collect a larger dataset of real smishing examples.  
  - Expand feature engineering beyond urgency/fear keywords.  
  - Compare traditional models with deep learning approaches (e.g., BERT).  

### 2. Phishing Email Classifier (Baseline)  
- **Goal:** Experiment with simple machine learning models to classify phishing vs. legitimate emails.  
- **Techniques:**  
  - TF-IDF for text representation  
  - Logistic Regression and Linear SVM models  
- **Results:**  
  - Accuracy around 50% with a tiny test dataset (only 4 messages).  
  - Proved that the pipeline works end-to-end, even though the dataset was too small to simulate real world.  
- **Next Steps:**  
  - Train on a real phishing email dataset from Kaggle or another public source.  
  - Compare performance of different models.  

## What I’ve Learned So Far  
- How to set up a Python project with clear file structure.  
- How to install and manage packages using `requirements.txt`.  
- How to load and clean text data with pandas.  
- How TF-IDF works to turn text into numbers.  
- How to train and test basic classifiers in scikit-learn.  
- How to interpret results in terms of precision, recall, and F1-score.  

## Next Steps  
- Collect larger, more realistic datasets for both projects.  
- Try modern NLP methods such as word embeddings or transformers.  
- Build more projects that directly combine AI/NLP with cybersecurity use cases.  

## Context  
This repo is part of my learning journey in AI, NLP, and cybersecurity.  
I am documenting my progress project by project, so others can see both my technical growth and my dedication to bridging human psychology, cybersecurity, and AI. 
