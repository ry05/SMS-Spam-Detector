"""
train.py
--------

To train a model to classify a given message or SPAM or HAM
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import joblib as jl

class Trainer:
    """
    Trains a NB model and stores it as a .pkl file
    """
    
    def __init__(self, filename):
        """
        Initialise the dataframe
        """
        try:
            self.df = pd.read_csv(filename)
        except:
            print("File name not recognized...")
            
    def preprocess(self):
        """
        Prepares the dataset
        """
        
        # data prep
        self.df["label"] = self.df["Category"].map({'ham': 0, 'spam': 1})
        self.df = self.df.drop(["Category"], axis=1)
        
        # some basic NLP preprocessing
        corpus = self.df["Message"]
        # using BoW
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(corpus)
        jl.dump(vectorizer, "models/vectorizer.pkl") # store the vectorizer
        
        return (bow_matrix, self.df["label"])
        
    def modelling(self):
        """
        Modelling with traditional holdout method of data split
        -----
        
        X : Descriptors
        y : Target
        """
        
        (X, y) = self.preprocess()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=42)

        # modelling with holdout
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
                
        return clf
    
    def interpret_classifier(self):
        """
        Interpret the classifier
        -----
        
        This function can be called separately to see the misclassified samples
        """
        
        (X, y) = self.preprocess()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=42)

        # modelling with holdout
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        
        # make predictions
        preds = clf.predict(X_test)
        
        # classification report
        print("Classification Report")
        print(classification_report(y_test, preds,
                                    target_names=['ham', 'spam']))
        
        # find misclassified samples
        print("Misclassified Samples")
        misses = np.where(y_test != preds)
        misclassified = self.df.iloc[misses]
        print(misclassified)
        misclassified[misclassified['label']==0].\
            to_csv(r'misclassified_samples/Misclassified Ham Samples.txt', sep=' ')
        misclassified[misclassified['label']==1].\
            to_csv(r'misclassified_Samples/Misclassified Spam Samples.txt', sep=' ')
        
        return clf
    
    def store_model(self):
        """
        Store the model
        -----
        
        model : Trained model to be stored 
        """
        
        model = self.modelling()
        jl.dump(model, "models/model.pkl")
        
    def train_and_store(self):
        """
        Main function
        """
        
        self.store_model()
   

# To run this file as a standalone python script, uncomment the following code block

"""
if __name__ == "__main__":
    tr = Trainer("data/spam_text_messages.csv")
    tr.interpret_classifier()
"""


