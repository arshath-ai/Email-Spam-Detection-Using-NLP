
# Email Spam Detection Using NLP

## Project Overview
An Email Spam Detection system built using Natural Language Processing (NLP), CountVectorizer, and Multinomial Naive Bayes. A Flask web application allows users to enter a message and predict whether it is spam or ham.

## Technologies Used
- Python
- Flask
- Natural Language Processing (NLP)
- NLTK
- Scikit-learn
- Pandas
- Multinomial Naive Bayes
- CountVectorizer
- HTML and CSS

## Features
- Text preprocessing
- Stopword removal
- Stemming using Porter Stemmer
- Bag of Words feature extraction
- Spam and ham classification
- Flask web interface

## Project Structure
- app.py: Flask web application
- spam.py: Model training and evaluation
- model.pkl: Trained classification model
- cv-transform.pkl: Saved CountVectorizer
- templates/: HTML templates
- requirements.txt: Python dependencies

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd email-spam-detection
```

Install dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Model
The model uses Multinomial Naive Bayes with Bag of Words features.

## Author
Seyyadh Arshath
