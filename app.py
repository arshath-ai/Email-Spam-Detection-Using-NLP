# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import os

# Get the folder where this Python file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to saved model and vectorizer
model_path = os.path.join(BASE_DIR, 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'cv-transform.pkl')

# Load the Multinomial Naive Bayes model
with open(model_path, 'rb') as file:
    classifier = pickle.load(file)

# Load CountVectorizer
with open(vectorizer_path, 'rb') as file:
    cv = pickle.load(file)

# Create Flask application
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    # Convert message into a list
    data = [message]

    # Transform text using CountVectorizer
    vect = cv.transform(data).toarray()

    # Predict spam or ham
    my_prediction = classifier.predict(vect)

    return render_template(
        'result.html',
        prediction=my_prediction
    )


if __name__ == '__main__':
    app.run(debug=True)