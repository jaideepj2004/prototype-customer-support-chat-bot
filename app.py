import string
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, jsonify, render_template

# Initialize Flask application
app = Flask(__name__)

# Configure logging to save to a file
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a string
    return ' '.join(tokens)

# Load the pre-trained TF-IDF vectorizer and Random Forest model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
rf_classifier = joblib.load('tfidf_random_forest_model.pkl')

# Route for home page
@app.route('/')
def home():
    logger.info("Rendering home page")
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            user_input = request.form['user_input']
            logger.info(f"Received input: {user_input}")

            # Process user_input and make predictions
            processed_input = preprocess_text(user_input)
            logger.info(f"Processed input: {processed_input}")

            input_vector = tfidf_vectorizer.transform([processed_input])
            prediction = rf_classifier.predict(input_vector)[0]
            logger.info(f"Prediction: {prediction}")

            # Directly return the prediction
            return jsonify({'response': prediction})
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return jsonify({'response': "Sorry, something went wrong."})

if __name__ == '__main__':
    app.run(debug=True)
