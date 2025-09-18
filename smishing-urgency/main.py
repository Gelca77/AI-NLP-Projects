#!/usr/bin/env python3
# Simple script that:

# - loads data/messages.csv
# - creates lexicon features (urgency_count, fear_count)
# - trains TF-IDF + LogisticRegression
# - prints evaluation metrics and top indicative terms

import os
import sys

# trys imports and show helpful message if missing
try:
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from scipy.sparse import hstack
except Exception as e:
    print("A required package is missing. Run:")
    print("    pip install -r requirements.txt")
    sys.exit(1)

from src.utils import clean_text, count_hits, URGENCY_WORDS, FEAR_WORDS

DATA_PATH = os.path.join("data", "messages.csv")

def load_data(path):
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    return df

def featurize(df):
    df["clean"] = df["text"].apply(clean_text)
    df["urgency_count"] = df["clean"].apply(lambda x: count_hits(x, URGENCY_WORDS))
    df["fear_count"] = df["clean"].apply(lambda x: count_hits(x, FEAR_WORDS))
    return df

def train_and_eval(df):
    X_text = df["clean"].values
    y = df["label"].values

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_tfidf = tfidf.fit_transform(X_text)
    X_extra = df[["urgency_count", "fear_count"]].values
    X_all = hstack([X_tfidf, X_extra])

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.25, stratify=y, random_state=42)

    clf = LogisticRegression(max_iter=400)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print("*** Classification Report ***")
    print(classification_report(y_te, y_pred, zero_division=0))

    print("Confusion matrix (rows: true, cols: pred):")
    print(confusion_matrix(y_te, y_pred))

    # Show top positive coefficients for 'phish' (if labels are phish/legit)
        
    labels = list(clf.classes_)
    feature_names = list(tfidf.get_feature_names_out()) + ["urgency_count", "fear_count"]

    if len(labels) == 2:
        # Binary classification: one coefficient vector; positive direction = labels[1]
        pos_class = labels[1]
        coefs = clf.coef_[0]
        top_idx = np.argsort(coefs)[-15:][::-1]
        print(f"\nTop positive indicators for '{pos_class}' (term, weight):")
        for i in top_idx:
            print(feature_names[i], round(coefs[i], 3))
    else:
        # Multiclass (not your case, but safe to have)
        for c_idx, cls_name in enumerate(labels):
            coefs = clf.coef_[c_idx]
            top_idx = np.argsort(coefs)[-10:][::-1]
            print(f"\nTop positive indicators for '{cls_name}' (term, weight):")
            for i in top_idx:
                print(feature_names[i], round(coefs[i], 3))



def main():
    print("Loading data from", DATA_PATH)
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows. Class counts:")
    print(df["label"].value_counts())
    df = featurize(df)
    train_and_eval(df)

if __name__ == "__main__":
    main()
