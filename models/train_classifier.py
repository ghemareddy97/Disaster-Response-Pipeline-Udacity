import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import re
import pickle

def load_data(database_filepath):
    
    '''
        Doc String 
        Description: This function extracts data from the database creates train and test dataframes
        Input: String - path for the database file 
        Output: Dataframe - train dataframe, target dataframe, labels
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_table', engine)
    X = df['message']
    y = df.drop(['id','message', 'original', 'genre'], axis = 1)
    return X, y, y.columns

def tokenize(text):
    '''
        Doc String 
        Description: This function performs string operations that will remove punctuations, stopwords, numbers and also tokenized, lemmatized . 
        Input: String  
        Output: string with pure string characters
    '''
    
    stop_words = set(stopwords.words('english'))  
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'[0-9]', '', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(w) for w in text if not w in stop_words]


def build_model():
    '''
        Doc String 
        Description: This function creates pipline object which has a transformer and an estimator with tfidfvectorizer and classifier.  
        Input: None
        Output: pipeline model object
    '''
    
    pipeline = Pipeline([
        ('cntvect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {

    'clf__estimator__n_estimators':[25, 30],
}

    cv = GridSearchCV(pipeline, parameters,n_jobs=9, verbose=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
        Doc String 
        Description: This function uses pipeline object to predict on the test dataframe and extract classification report. 
        Input: pipeline object, test input, test target, test labels
        Output: None
    '''
    
    ypred = pd.DataFrame(model.predict(X_test), columns = category_names)
    for column in category_names:
        print(column)
        print(classification_report(Y_test[column], ypred[column]))


def save_model(model, model_filepath):
    
    '''
        Doc String 
        Description: This function saves the model to a pickle file.
        Input: pipline object, file path to save the model
        Output: None
    '''
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()