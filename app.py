import streamlit as st
import pickle
import re


# -----------------------------
# Load model
# -----------------------------

with open(
    "model/model.pkl",
    "rb"
) as file:

    model = pickle.load(file)


# -----------------------------
# Load vectorizer
# -----------------------------

with open(
    "model/vectorizer.pkl",
    "rb"
) as file:

    vectorizer = pickle.load(file)


# -----------------------------
# Text cleaning
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


# -----------------------------
# Streamlit page
# -----------------------------

st.title(
    "🛍️ Product Review Classification"
)

st.write(
    "Enter a product review to predict "
    "whether it is Positive or Negative."
)


# -----------------------------
# User input
# -----------------------------

review = st.text_area(
    "Enter your review:"
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        # Clean review
        cleaned_review = clean_text(
            review
        )

        # Convert text to TF-IDF
        review_vector = vectorizer.transform(
            [cleaned_review]
        )

        # Prediction
        prediction = model.predict(
            review_vector
        )[0]


        # Show result
        if prediction == "positive":

            st.success(
                "😊 Positive Review"
            )

        else:

            st.error(
                "😞 Negative Review"
            )