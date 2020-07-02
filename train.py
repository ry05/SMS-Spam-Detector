"""
train.py
--------

To train a model to classify a given message or SPAM or HAM
"""

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

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
    

