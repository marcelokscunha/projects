# ------------------ IMPORTS  ------------------ #
import sys
import datetime
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from joblib import dump, load

from xgboost import XGBClassifier

import optuna

# ------------------ END IMPORTS  ------------------ #

# ------------------ SET CONFIGS  ------------------ #
nltk.download(['stopwords'])
pd.set_option('display.max_columns',40)
# ------------------ END CONFIGS  ------------------ #

def load_data(database_filepath):
    """
    Function for loading the data from a Database with SQLAlchemy to a Pandas dataframe.
    The database used is SQLite.
    
    Parameters
    ----------
    database_filepath : string containing the path of the SQLite database file

    Returns
    -------
    X: nx1 numpy array containing the messages. n is the number of messages 
    
    Y: nxm numpy array containing the messages. n is the number of messages and m is the number of labels
    
    category_names: list containing columns names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('disaster_data', engine)
    df.drop('child_alone', axis=1, inplace=True)
    X = df.iloc[:,1].values
    Y = df.iloc[:,4:].values
    category_names = df.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Function for tokenizing text with NLTK.

    Parameters
    ----------
    text : string containing the text to be tokenized

    Returns
    -------
    tokens: list of tokens processed. Each token is a string representing a tokenized word.
    """
    # normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    # remove leading and trailing spaces before lemmatizing
    tokens = [lemmatizer.lemmatize(word.strip()) for word in tokens if word not in stop_words]  

    return tokens


def build_model(model_name='gbc'):
    """
    Function for building the ML model with Scikit-Learn's Gradient Boosting Classifier or XGBoost Classifier.
    A pipeline including tokenization and the model is returned.
    
    Parameters
    ----------
    model_name : string of the model to be trained. 
        Possible values - 'gbc' for Gradient Boosting Classifier or 'xgb' for XGBoost Classifier...')
    
    Returns
    -------
    pipeline: sklearn.Pipeline object containing the pre-processing (TfidfVectorizer with the tokenizer function) 
        and the ML model.

    """
    if model_name=='gbc':
        print('Creating pipeline for Gradient Boosting Classifier...')
        pipeline = Pipeline([
                ('features', TfidfVectorizer(tokenizer=tokenize)),
                ('clf', MultiOutputClassifier(GradientBoostingClassifier()))
            ], 
            verbose=True)
        
    elif model_name=='bert':
        raise NotImplementedError("Scikit-Learn's wrapper for BERT is still being implemented. For more information of development view 'ML Pipeline Preparation.ipynb'")
    
    elif model_name=='xgb':
        print('Creating pipeline for XGBoost Classifier...')
        pipeline = Pipeline([
                ('features', TfidfVectorizer(tokenizer=tokenize)),
                ('clf', MultiOutputClassifier(XGBClassifier()))
            ], 
            verbose=True)        
    else:
        raise NotImplementedError("This type of model hasn't been implemented yet. Try 'gbc' for Scikit-Learn's Gradient Boosting Classifier or 'xgb' for XGBoost.")

    return pipeline


def set_hyperparameters(model_name=model_name)
    """
    Function for setting the grid hyperparameters for GridSearch.
    
    Parameters
    ----------
    model_name : string of the model to be trained. 
        Possible values - 'gbc' for Gradient Boosting Classifier or 'xgb' for XGBoost Classifier...')
    
    Returns
    -------
    parameters: dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    """
    if model_name=='gbc':
        print('Setting hyperparameter grid for Gradient Boosting Classifier...')
        parameters = {
            'clf__estimator__learning_rate': [0.03, 0.3],
            'clf__estimator__n_estimators': [100, 200],
            'clf__estimator__max_depth': [2,4]
        }
        
    elif model_name=='bert':
        raise NotImplementedError("Scikit-Learn's wrapper for BERT is still being implemented. For more information of development view 'ML Pipeline Preparation.ipynb'")
    
    elif model_name=='xgb':
        print('Setting hyperparameter grid for XGBoost Classifier...')
        parameters = {
            "clf__estimator__booster": ["gbtree", "dart"],
            "clf__estimator__lambda":  [5e-2, 5e-1],
            "clf__estimator__eta": [5e-2, 5e-1]
        }
                       
    else:
        raise NotImplementedError("This type of model hasn't been implemented yet. Try 'gbc' for Scikit-Learn's Gradient Boosting Classifier or 'xgb' for XGBoost.")
    
    return parameters
    
        

def create_reports(clfs_names, Y_test, Y_preds, column_names, verbose=True, df_reports=None):
    """
    Function for printing classification reports for each label (column) and generating a Pandas DataFrame with metrics for the whole model averaging labels. 
    
    Parameters
    ----------
    clfs_names : array of strings, shape = [n_classifiers]
        Names of each classifier
    
    Y_test : 2d array-like
        Ground truth (correct) target values. Each column is a label.

    Y_preds : n x 2d array-like
        Estimated targets as returned by a classifier. List of predictions for n classifiers.
    
    column_names: list of column names
    
    df_reports: Pandas DataFrame of the metrics for the whole multi-label model.
        If no DataFrame report if provided, create a new one. If it is provided, append metrics for new models.
    
    verbose: bool, default: True
        When set to True, prints classification_report for each column.
    
    Returns
    -------
    df_reports: Pandas DataFrame of the metrics for the whole multi-label models (if more than one)
    
    """
    if df_reports is None:
        df_reports = pd.DataFrame(columns=['mean accuracy','mean macro avg f1-score', 'mean weighted avg f1-score'])
    
    for clf_name, Y_preds in zip(clfs_names,Y_preds):#,Y_preds_hgbc,Y_preds_rfc]):
        clf_metrics = pd.DataFrame()
        if verbose:
            print(f'Metrics for each feature for model - {clf_name}\n')
        for col in range(Y_preds.shape[1]):
            report = classification_report(Y_test[:,col], Y_preds[:,col], output_dict=True)
            # model_reports[clf_name] = [report['accuracy'], report['macro avg']['f1-score'], report['weighted avg']['f1-score']]
            label_metrics = pd.DataFrame(data=[[report['accuracy'], report['macro avg']['f1-score'], report['weighted avg']['f1-score']]])
            clf_metrics = pd.concat([clf_metrics, label_metrics], axis=0)
            if verbose:
                print('Column:', column_names[col])
                print(classification_report(Y_test[:,col], Y_preds[:,col]),'\n   -----------------------------------------------\n')
        clf_metrics = clf_metrics.mean(axis=0).to_frame().transpose().rename(index={0:clf_name}, columns={0:'mean accuracy',1:'mean macro avg f1-score',2:'mean weighted avg f1-score'})
        df_reports = pd.concat([df_reports, clf_metrics],axis=0)
        
    return df_reports


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function for evaluating model and printing the classification reports for each label (column).
    
    Parameters
    ----------
    model: model to be evaluated. Must implement a predict method ('model.predict(X_test)')
    
    X_test: 1d array-like.
        Each item of the array is a text to be processed and predict the labels
    
    Y_test: 2d array-like
        Ground truth (correct) labels. Each column is a label.
    
    category_names: array containing the strings of the names of labels
    
    Returns
    -------
    None
    
    """
    model_name = model.__class__.__name__
    Y_preds = model.predict(X_test)
    
    final_report = create_reports([model_name], Y_test, [Y_preds], category_names, verbose=True, df_reports=None)
    print(final_report)

def save_model(model, model_filepath):
    """
    Function to save the model in the specified path.
    
    Parameters
    ----------
    model: object of the model that already trained
    
    model_filepath: string containing the path to save to model to
    
    Returns
    -------
    None
    
    """
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3 or len(sys.argv)==4:
        if len(sys.argv) == 4:
            database_filepath, model_filepath, model_name = sys.argv[1:]
            
        else:
            database_filepath, model_filepath = sys.argv[1:], 'gbc'
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(model_name=model_name)
        
        print('Configuring grid for training multiple models with GridSearch...')
        parameters = set_hyperparameters(model_name=model_name)
        cv_model = GridSearchCV(model, param_grid=parameters, n_jobs=-1 ,verbose=2)
        
        print('Training model...')
        cv_model.fit(X_train, Y_train)
                
        print('Evaluating model...')
        evaluate_model(cv_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument.'\
              '\nOptionally you can provide a third argument as the chosen model name that you want to train.'\
              '\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl'\
              '\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl xgb')


if __name__ == '__main__':
    main()