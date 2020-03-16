# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
import re
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = -1


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_colnames = list(df.columns[4:])
    return X,y,category_colnames


def tokenize(text):
     #normalize text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    #create tokens
    tokens = word_tokenize(text.lower().strip())
    #remove stopwords from tokens
    stop_words = set(stopwords.words('english')) 
    filtered_tokens = [token for token in tokens if token not in stop_words]   
    #apply stemming to convert words to their root
    ps = PorterStemmer()
    stem_tokens = [ps.stem(token) for token in filtered_tokens] 
    #apply lemmatizer to convert words to their lemma
    wnl = WordNetLemmatizer()
    lem_tokens = [wnl.lemmatize(token) for token in stem_tokens]
    return lem_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    # RandomForestClassifier
    'clf__estimator__n_estimators': [50,100],
    #'clf__estimator__min_samples_split': [2,5],
    #'clf__estimator__criterion': ['entropy', 'gini']
    
    # SVC
    #"clf__estimator__C": [0.001, 0.01, 0.1, 1, 10],
    #"clf__estimator__gamma":[0.001, 0.01, 0.1, 1]

    # DecisionTreeClassifier
    #"clf__estimator__criterion": ['entropy', 'gini'],
    #"clf__estimator__min_samples_split":[2,4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    #calculate f1 score, precision, recall, and accuracy for each output category
    y_pred = model.predict(X_test)
    y_true = Y_test.iloc[:,category_names].values
    classification_repo = classification_report(y_true, y_pred[:,category_names])
    accuracy = accuracy_score(y_true, y_pred[:,category_names])
    
    print("Classification Report: ", category_names, "\n", classification_repo)
    print("Accuracy:", accuracy, "\n")
    
    all_cat_accuracy = (y_pred == Y_test).mean().mean()
    print("Overall Model Accuracy:", all_cat_accuracy, "\n")


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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