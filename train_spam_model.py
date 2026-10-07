import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# Paths
# =========================================================

DATASET_PATH = "datasets/SMSSpamCollection"
MODEL_PATH = "models/spam_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# =========================================================
# Create models folder
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)


# =========================================================
# Load SMS Spam Dataset
# =========================================================

texts = []
labels = []


with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
    errors="ignore"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        parts = line.split(
            "\t",
            1
        )

        if len(parts) != 2:
            continue

        label, message = parts

        label = label.strip().lower()
        message = message.strip()

        if label == "ham":

            labels.append(0)
            texts.append(message)

        elif label == "spam":

            labels.append(1)
            texts.append(message)


# =========================================================
# Dataset information
# =========================================================

print()
print("Total messages:", len(texts))
print("Ham messages:", labels.count(0))
print("Spam messages:", labels.count(1))
print()


# =========================================================
# Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    texts,
    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print()


# =========================================================
# TF-IDF Vectorizer
# =========================================================

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.95,

    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print("TF-IDF training complete.")
print()


# =========================================================
# SVM Model
# =========================================================

model = SVC(

    kernel="linear",

    probability=True,

    class_weight="balanced",

    random_state=42
)


model.fit(

    X_train_tfidf,

    y_train
)


print("SVM training complete.")
print()


# =========================================================
# Model Evaluation
# =========================================================

predictions = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("==============================")
print(
    "MODEL ACCURACY:",
    round(accuracy * 100, 2),
    "%"
)
print("==============================")
print()


print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Ham",
            "Spam"
        ]
    )
)


# =========================================================
# Save Model
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


print()
print(
    "Spam model saved:"
)

print(
    MODEL_PATH
)

print(
    "TF-IDF vectorizer saved:"
)

print(
    VECTORIZER_PATH
)

print()
print(
    "Training completed successfully!"
)