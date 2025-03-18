from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train model
df = pd.read_csv("indian_medicine_dataset_large.csv")
X = df['Symptoms']
y = df[['Disease Name', 'English Medicine Name', 'Ayurvedic Medicine Name', 'Diet Recommendation']]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
pipeline.fit(X, y)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    prediction = pipeline.predict([symptoms])[0]
    return render_template('index.html', 
                           symptoms=symptoms,
                           disease=prediction[0],
                           english_medicine=prediction[1],
                           ayurvedic_medicine=prediction[2],
                           diet=prediction[3])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

