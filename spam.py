# ==========================================================
# EMAIL SPAM DETECTION USING NLP
# ==========================================================

import os
import re
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

# Get the folder where spam.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
FILE_PATH = os.path.join(BASE_DIR, "EmailCollection")

print("========================================")
print("EMAIL SPAM DETECTION USING NLP")
print("========================================")

print("\nDataset Path:")
print(FILE_PATH)

# Check whether the dataset exists
if not os.path.isfile(FILE_PATH):
    print("\nERROR: EmailCollection file not found!")
    print("\nAvailable files in the folder:")
    print(os.listdir(BASE_DIR))

    raise FileNotFoundError(
        f"\nEmailCollection not found at:\n{FILE_PATH}"
    )


# Read dataset
messages = pd.read_csv( FILE_PATH, sep="\t",  names=["LABEL", "MESSAGES"], encoding="latin-1")

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Records:")
print(messages.head())

print("\nDataset Shape:")
print(messages.shape)

print("\nLabel Counts:")
print(messages["LABEL"].value_counts())


# # ==========================================================
# # 2. VISUALIZE HAM AND SPAM MESSAGES
# # ==========================================================

# plt.figure(figsize=(6, 4))

# sns.countplot(
#     x="LABEL",
#     data=messages
# )

# plt.title("Ham vs Spam Messages")

# plt.show()


# ==========================================================
# 3. DOWNLOAD REQUIRED NLTK DATA
# ==========================================================

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# ==========================================================
# 4. TEXT PREPROCESSING
# ==========================================================

print("\nProcessing messages...")

stemmer = PorterStemmer()

# Get stopwords once for better performance
stop_words = set(stopwords.words("english"))

corpus = []


for i in range(len(messages)):

    # Convert message to string
    review = str(messages["MESSAGES"].iloc[i])

    # Remove numbers and special characters
    review = re.sub("[^a-zA-Z]", " ", review)

    # Convert to lowercase
    review = review.lower()

    # Split into words
    review = review.split()

    # Remove stopwords and apply stemming
    review = [
        stemmer.stem(word)
        for word in review
        if word not in stop_words
    ]

    # Join words
    review = " ".join(review)

    # Add to corpus
    corpus.append(review)


print("\nText Processing Completed!")

print("\nFirst 5 Processed Messages:")

for message in corpus[:5]:
    print(message)


# ==========================================================
# 5. CREATE BAG OF WORDS MODEL
# ==========================================================

print("\nCreating Bag of Words model...")

cv = CountVectorizer(max_features=3500)

X = cv.fit_transform(corpus).toarray()

print("Feature Matrix Shape:", X.shape)


# ==========================================================
# 6. CONVERT LABELS TO NUMBERS
# ==========================================================

# ham = 0
# spam = 1

y = messages["LABEL"].map(
    {
        "ham": 0,
        "spam": 1
    }
).values

print("\nLabels converted successfully!")


# ==========================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING SETS
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)


print("\nTraining Data Shape:", X_train.shape)

print("Testing Data Shape:", X_test.shape)


# ==========================================================
# 8. TRAIN MULTINOMIAL NAIVE BAYES MODEL
# ==========================================================

print("\nTraining Multinomial Naive Bayes Model...")

mnb = MultinomialNB(alpha=0.8)

mnb.fit(X_train, y_train)

print("Model Training Completed!")


# ==========================================================
# 9. MAKE PREDICTIONS
# ==========================================================

y_pred_mnb = mnb.predict(X_test)


# ==========================================================
# 10. CALCULATE MODEL ACCURACY
# ==========================================================

mnb_acc = accuracy_score(
    y_test,
    y_pred_mnb
)

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print("MultinomialNB Accuracy:", mnb_acc)

print(
    "MultinomialNB Accuracy Percentage:",
    round(mnb_acc * 100, 2),
    "%"
)


# # ==========================================================
# 11. TEST A NEW MESSAGE
# ==========================================================

message = """
Go until jurong point, crazy.. Available only in bugis n
great world la e buffet... Cine there got amore wat...
"""


print("\n========================================")
print("TESTING NEW MESSAGE")
print("========================================")

print("\nOriginal Message:")
print(message)


# ==========================================================
# 12. PREPROCESS NEW MESSAGE
# ==========================================================

review = re.sub(
    "[^a-zA-Z]",
    " ",
    message
)

review = review.lower()

review = review.split()


review = [
    stemmer.stem(word)
    for word in review
    if word not in stop_words
]


processed_message = " ".join(review)


print("\nProcessed Message:")
print(processed_message)


# ==========================================================
# 13. CONVERT MESSAGE USING BAG OF WORDS
# ==========================================================

vect = cv.transform(
    [processed_message]
).toarray()


# ==========================================================
# 14. PREDICT MESSAGE
# ==========================================================

my_prediction = mnb.predict(vect)[0]


# ==========================================================
# 15. DISPLAY RESULT
# ==========================================================

print("\n========================================")
print("PREDICTION RESULT")
print("========================================")

if my_prediction == 0:

    print("Prediction: It's a HAM Mail")

else:

    print("Prediction: It's a SPAM Mail")


# ==========================================================
# 16. SAVE COUNT VECTORIZER
# ==========================================================

CV_PATH = os.path.join(
    BASE_DIR,
    "cv-transform.pkl"
)

with open(
    CV_PATH,
    "wb"
) as file:

    pickle.dump(
        cv,
        file
    )


print("\nCountVectorizer saved successfully!")


# ==========================================================
# 17. SAVE TRAINED MODEL
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        mnb,
        file
    )


print("Model saved successfully!")


