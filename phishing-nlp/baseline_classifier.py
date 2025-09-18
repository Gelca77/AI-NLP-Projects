# Angelica Shelman
# #Exploring TF-IDF features and two simple models (LogReg and Linear SVM) so I can get a small pipeline running 

#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Where the starter data is (only a tiny toy CSV)
DATA_PATH = Path("data/sample.csv")   # starter dataset
TEXT_COL  = "text"
LABEL_COL = "label"

# loading the dataset and sanity check the columns
df = pd.read_csv(DATA_PATH)
print("Columns found:", df.columns.tolist())
df = df[[TEXT_COL, LABEL_COL]].dropna().copy()

# Mapping string labels to integers so sklearn is happy
unique_labels = sorted(df[LABEL_COL].unique())
label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
inv_label_map = {v: k for k, v in label_map.items()}
df["y"] = df[LABEL_COL].map(label_map)

# cleaning text where Lowercase and collapse extra spaces and wanted to keep it simple for now by not removing stopwards or puncuation
def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

df["x"] = df[TEXT_COL].apply(clean_text)

# trainand test split (50/50 since dataset is so small)
#Strarify makes sure both classes show up in each split
X_train, X_test, y_train, y_test = train_test_split(
    df["x"], df["y"], test_size=0.5, stratify=df["y"], random_state=42
)

# Using unigrams + bigrams (might change later)
#Wrap TF-IDF and model in a pipeline so its easier to use later
def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95, stop_words="english")),
        ("clf", clf),
    ])
#two simple baselines to compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "LinearSVM": LinearSVC()
}

# train and evaluate each model
for name, clf in models.items():
    pipe = make_pipeline(clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=[inv_label_map[i] for i in range(len(unique_labels))])
    cm = confusion_matrix(y_test, preds)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(report)
    print("Confusion matrix:\n", cm)
