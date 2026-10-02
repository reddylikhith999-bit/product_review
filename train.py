import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv("data/reviews.csv")

print("Dataset loaded")

print(df.head())

print("\nDataset shape:")
print(df.shape)


# -----------------------------
# 2. Remove missing values
# -----------------------------

df = df.dropna()


# -----------------------------
# 3. Text preprocessing
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["clean_review"] = df["review"].apply(
    clean_text
)


# -----------------------------
# 4. X and y
# -----------------------------

X = df["clean_review"]

y = df["sentiment"]


# -----------------------------
# 5. Train test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining data:", len(X_train))

print("Testing data:", len(X_test))


# -----------------------------
# 6. TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print("\nTF-IDF completed")


# -----------------------------
# 7. Logistic Regression
# -----------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_tfidf
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)


print("\nLogistic Regression Accuracy:")

print(logistic_accuracy)


# -----------------------------
# 8. SVM
# -----------------------------

svm_model = LinearSVC()

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_prediction = svm_model.predict(
    X_test_tfidf
)

svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)


print("\nSVM Accuracy:")

print(svm_accuracy)


# -----------------------------
# 9. Reports
# -----------------------------

print("\nLogistic Regression Report")

print(
    classification_report(
        y_test,
        logistic_prediction
    )
)


print("\nSVM Report")

print(
    classification_report(
        y_test,
        svm_prediction
    )
)


# -----------------------------
# 10. Select final model
# -----------------------------

if svm_accuracy >= logistic_accuracy:

    final_model = svm_model

    print("\nFinal Model: SVM")

else:

    final_model = logistic_model

    print("\nFinal Model: Logistic Regression")


# -----------------------------
# 11. Save model
# -----------------------------

with open(
    "model/model.pkl",
    "wb"
) as file:

    pickle.dump(
        final_model,
        file
    )


# -----------------------------
# 12. Save vectorizer
# -----------------------------

with open(
    "model/vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


print("\nModel saved successfully!")

print("Vectorizer saved successfully!")