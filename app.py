"""
classifier.py
-------------

To classify a given message as SPAM or HAM
"""

import pandas as pd

from flask import Flask,render_template,url_for,request

from sklearn.feature_extraction.text import CountVectorizer

import joblib as jl

from train import Trainer


app = Flask(__name__)

@app.route('/')
def enter_input():
    """
    Enter the message to classify
    """
    
    return render_template('input.html')

def prep_model():
    """
    Train the model on the data and store it
    """
    
    tr = Trainer("data/spam_text_messages.csv")
    tr.train_and_store() # stores the model as models/model.pkl
    
@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify the given input message as SPAM or HAM
    """
    
    # prepare the model
    prep_model()
    
    # load the vectorizer and model
    VEC = open("models/vectorizer.pkl", 'rb')
    MODEL = open("models/model.pkl", 'rb')
    
    # instantiate the vectorizer and estimator
    cv = jl.load(VEC)
    clf = jl.load(MODEL)
        
    if request.method == 'POST':
        sms = request.form['message']
        data = [sms]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    