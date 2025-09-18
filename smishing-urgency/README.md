# Smishing Urgency Classifier  

A beginner-friendly machine learning project that shows how fear and urgency words in text messages can be used to detect smishing (SMS phishing).  

##  Introduction  
This project is part of my learning journey in Python, AI, and natural language processing (NLP).  
I wanted to connect what I’m learning with my interest in cybersecurity by building a small program that can spot smishing (SMS phishing) messages.  

A lot of smishing works by scaring people (“your account is suspended”) or making them feel rushed (“respond immediately”), so I thought it would be cool to try and measure those signals in my code.  

##  Method  
The Python script does a few main steps:  
1. Load the messages from a small CSV file (`data/messages.csv`).  
2. Clean up the text (make lowercase, strip extra spaces).  
3. Count how many urgency/fear words are in each message (like *immediately*, *suspended*, *avoid*, etc.).  
4. Turn all the words into numbers using TF-IDF (so the computer can work with them).  
5. Train a simple model (Logistic Regression).  
6. Test the model on a few messages it hasn’t seen before.  
7. Print out results and show which words/features the model thinks are most “phishy.”  

## Data  
- The dataset currently has 20 messages (10 phishing / 10 legit).
- These examples were collected from real smishing messages that were personally sent to me, combined with a few safe/legit messages for balance.
- Right now I only used a very tiny dataset (20 messages: half phishing, half legit).  
- I just wanted to make sure the whole process worked from start to finish.  
- Later, I plan to add more real examples from bigger public datasets so the model can actually learn better.  

## Results  
On my small test:  
- The model got about 80% accuracy (it guessed 4 out of 5 messages correctly).  
- It was very good at finding the “legit” ones (100% recall) but not perfect at spotting all “phish” ones (50% recall).  

What this means: with more training data, the model could get better at recognizing phishing, because right now it’s just guessing based on a tiny sample.  

The cool part is the top words/features the model thought were strong signals of phishing:  
- immediately  
- suspended  
- reactivate  
- permanent deactivation

This matches real-world smishing tricks, which often try to scare or rush people.  

It shows that even with just a handful of examples, the model is starting to “see” what makes phishing different from normal messages.  

## What I Learned    
- How to clean text and turn it into numbers with TF-IDF.  
- How to train and test a simple classifier in scikit-learn.  
- How to look at model outputs to understand why it makes certain predictions.  

## Next Steps  
- Add way more data so results are meaningful.  
- Try cross-validation(to test the model more fairly).  
- Add more lists of words (financial terms, scam patterns).  
- Experiment with advanced NLP methods (like BERT/transformers) when I’m ready.  

## How to Run  
- pip install -r requirements.txt
- python main.py

## Context
This project is part of me building experience in AI, NLP, and cybersecurity.  

I want to keep improving over time as I continue to learn
-Angelica Shelman
