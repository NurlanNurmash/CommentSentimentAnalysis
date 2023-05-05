from flask import Flask, render_template, url_for, request 
import joblib 
import pickle 
import re 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from keras.utils import pad_sequences
from keras.models import load_model
 
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

# Define a function to preprocess the text input
def preprocess(text):
    words_removed = list(stopwords.words('english'))+list(punctuation)

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in words_removed] 
    
    return ' '.join(text)

# Define a function to predict the sentiment of a text input
def predict_sentiment(model, text):
    # Preprocess the text input
    text = preprocess(text)
    print(text)
    # Vectorize the preprocessed text using the same vectorizer used for training the model
    with open('Tweet Analysis/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    text_vectorized = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(text_vectorized, maxlen=34)
    print(padded_sequences)
    # Make the prediction using the loaded model
    prediction = model.predict(padded_sequences)[0]
    # Return the predicted sentiment label (0 or 1)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('Tweet Analysis/my_model.h5')
    if request.method == 'POST':
        comment = request.form['comment']
        new_text = preprocess(comment)
        my_prediction = round(predict_sentiment(model, new_text)[0])
    
    return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)