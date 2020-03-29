#!/usr/bin/env python
# coding: utf-8

# Converted notebook into python script

# # ML Pipeline Preparation
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database
# - Define feature and target variables X and Y

# In[1]:


# import libraries and set configurations
from IPython.display import display
import tqdm
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

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
# pd.set_option('display.max_columns',40)


# In[2]:


# load data from database
engine = create_engine('sqlite:///../data/DisasterResponse.db')

df = pd.read_sql('disaster_data', engine)

# display(df.head())

# def highlight_imbalanced(col):
#     distance_perfect_distribution = np.abs(col - (1/len(col)))
#     return ['background-color: red' if len(col)<2 else 'background-color: yellow' if dist > 0.1 else '' for dist in distance_perfect_distribution]


# print('Viewing distributions of values per label (class imbalances are highlighted):\n')
# df_X = df.iloc[:,4:]
# for col in df_X:
#     ct = pd.crosstab(index=df_X[col], columns='%freq', normalize='columns')
#     display(ct.style.apply(highlight_imbalanced))
#     if len(ct) < 2:
#         print('Column with less than 2 values!')


# As shown above, almost all classes are imbalanced.
# 
# We highlighted the labels that were considered imbalanced (with skewed distributions). Labels are considered imbalanced here when their values have a distance from the perfect balance that is higher than 10%. For example, a label that has 3 possible values (0,1,2) has a perfect balance of 33.33%. If a value occurs more than 43.33% or less than 23.33% of the time, than the label is considered imbalanced.
# 
# `child_alone` is always 0. Therefore we could choose not to predict this column (always predict 0 for example without an ML model). For this reason we choose to drop this column.

# In[3]:


# Drop `child_alone` column
df.drop('child_alone', axis=1, inplace=True)


# In[4]:


# df.head()


# In[5]:


X = df.iloc[:,1].values
Y = df.iloc[:,4:].values


# In[6]:


# X[:5]


# # In[7]:


# Y[:5,:]


# In[8]:


column_names = df.columns; column_names


# ### 2. Write a tokenization function to process your text data

# In[9]:


message = X[0]


# In[10]:


def tokenize(text):
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


# In[11]:

print("Test and load tokenizer")
print("Message:", message)
print("Result of tokenization",tokenize(message))


# ### 3. Build a machine learning pipeline
# - Use MultiOutputClassifier to for predict multiple target variables.

# In[12]:


# pipeline_gbc = Pipeline([
#                     ('features', TfidfVectorizer(tokenizer=tokenize)),
#                     ('clf', MultiOutputClassifier(GradientBoostingClassifier()))
#                ],
#                verbose=True)


# # In[13]:


# pipeline_gbc.get_params()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[12]:


# random_state to make it easier to reproduce
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1000)


# In[15]:


# # %%prun #%%mrun
# pipeline_gbc.fit(X_train,Y_train)


# # In[16]:


# dump(pipeline_gbc, '../scripts/models/pipeline_gbc.joblib')


# # In[13]:


# pipeline_gbc = load('../scripts/models/pipeline_gbc.joblib')


# In[14]:


# pipeline_gbc


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[15]:


# Y_preds_gbc = pipeline_gbc.predict(X_test)


# In[16]:


# def create_reports(clfs_names, Y_test, Y_preds, df_reports=None):
#     """
#     Function for printing classification reports for each label (column) and generating a Pandas DataFrame with metrics for the whole model averaging labels. 
    
#     clfs_names : array of strings, shape = [n_classifiers]
#     Names of each classifier
    
#     Y_test : 2d array-like
#     Ground truth (correct) target values. Each column is a label.

#     Y_preds : n x 2d array-like
#     Estimated targets as returned by a classifier. List of predictions for n classifiers.

#     df_reports: Pandas DataFrame of the metrics for the whole multi-label model.
#     If no DataFrame report if provided, create a new one. If it is provided, append metrics for new models.
#     """
#     if df_reports is None:
#         df_reports = pd.DataFrame(columns=['mean accuracy','mean macro avg f1-score', 'mean weighted avg f1-score'])
    
#     for clf_name, Y_preds in zip(clfs_names,Y_preds):#,Y_preds_hgbc,Y_preds_rfc]):
#         clf_metrics = pd.DataFrame()
#         print(f'Metrics for each feature for model - {clf_name}\n')
#         for col in range(Y_preds.shape[1]):
#             report = classification_report(Y_test[:,col], Y_preds[:,col], output_dict=True)
#             # model_reports[clf_name] = [report['accuracy'], report['macro avg']['f1-score'], report['weighted avg']['f1-score']]
#             label_metrics = pd.DataFrame(data=[[report['accuracy'], report['macro avg']['f1-score'], report['weighted avg']['f1-score']]])
#             clf_metrics = pd.concat([clf_metrics, label_metrics], axis=0)
#             print('Column:', column_names[col])
#             print(classification_report(Y_test[:,col], Y_preds[:,col]),'\n   -----------------------------------------------\n')
#         clf_metrics = clf_metrics.mean(axis=0).to_frame().transpose().rename(index={0:clf_name}, columns={0:'mean accuracy',1:'mean macro avg f1-score',2:'mean weighted avg f1-score'})
#         df_reports = pd.concat([df_reports, clf_metrics],axis=0)
        
#     return df_reports


# # In[17]:


# df_reports = create_reports(['gbc'], Y_test, [Y_preds_gbc])


# # In[18]:


# print('Mean metrics for all outputs of a model:')
# df_reports


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[22]:


# pipeline_gbc.get_params()


# # In[24]:


# parameters = {
#     'clf__estimator__learning_rate': [0.03, 0.3],
#     'clf__estimator__n_estimators': [100, 200],
#     'clf__estimator__max_depth': [2,4]
# }

# cv_gbc = GridSearchCV(pipeline_gbc, param_grid=parameters, n_jobs=-1 ,verbose=2)
# cv_gbc.fit(X_train, Y_train)


# In[25]:


# dump(cv_gbc, '../scripts/models/cv_gbc.joblib')


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  

# In[19]:


# cv_gbc = load('../scripts/models/cv_gbc.joblib')


# In[20]:


# Y_preds_gbc_tuned = cv_gbc.predict(X_test)


# # In[21]:


# df_reports = create_reports(['gbc_tuned'], Y_test, [Y_preds_gbc_tuned], df_reports=df_reports)


# In[22]:


# df_reports


# # In[23]:


# # Other way to calculate mean accuracy for all labels
# y = cv_gbc.best_estimator_.predict(X_test)
# (y==Y_test).mean()


# ### 8. Improving  model further

# In[13]:


# TODOs
# BERT - Modify head -> view "ML Pipeline Preparation.ipynb"
# Optuna gbc 


# In[14]:


def mean_accuracy_score(estimator, X, y):
    y_pred = estimator.predict(X)
    score = (y_pred==y).mean()
    return score


# In[15]:


def mean_macro_average_f1(estimator, X, y):
    y_pred = estimator.predict(X)
    
    f_tmp = 0
    for col in range(y.shape[1]):
        f_tmp += f1_score(y[:,col], y_pred[:,col], average='macro')
    
    score = f_tmp/y.shape[1]
    return score


# In[ ]:

class Objective(object):
    def __init__(self, data):
        self.data = data

    def __call__(self, trial):
        x, y = self.data

        classifier_name = trial.suggest_categorical("classifier", ["GradientBoostingClassifier", "XGB", "SVC"])
        
        if classifier_name == "GradientBoostingClassifier":
            gbc_lr = trial.suggest_loguniform("gbc_lr", 1e-2, 6e-1)
            gbc_estimators = int(trial.suggest_loguniform("gbc_estimators", 30, 300))
            gbc_depth = int(trial.suggest_uniform("gbc_depth", 3, 8))
            
            gbc = GradientBoostingClassifier(learning_rate=gbc_lr,
                                             n_estimators=gbc_estimators,
                                             max_depth=gbc_depth)
            transform = TfidfVectorizer(tokenizer=tokenize)
            
            classifier_obj = make_pipeline(transform, MultiOutputClassifier(gbc))
        
        elif classifier_name=="SVC":
            svc_c = trial.suggest_loguniform("svc_c", 1e-10, 1e10)
            svc_gamma = trial.suggest_categorical("svc_gamma", ["auto", "scale"])
            
            svc = SVC(C=svc_c, gamma=svc_gamma)
            transform = TfidfVectorizer(tokenizer=tokenize)
            
            classifier_obj = make_pipeline(transform, MultiOutputClassifier(svc))
            
        else:            
            param = {
                "silent": 1,
                "objective": "binary:logistic",
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            }

            if param["booster"] == "gbtree" or param["booster"] == "dart":
                param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
                param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
                param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
                param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)
            
            
            xgb = XGBClassifier(**param)    
            transform = TfidfVectorizer(tokenizer=tokenize)
            
            classifier_obj = make_pipeline(transform, MultiOutputClassifier(xgb))
        
        print(f"[{trial.number}] Cross-validating trial ...\n")
        print(f"[{trial.number}] Using {classifier_name}")
        # Use 1 CPU core per trial
        score = cross_val_score(classifier_obj, x, y, scoring=mean_macro_average_f1, n_jobs=1, cv=2)
        print(f"[{trial.number}] Finished cross-validation!")
        score = score.mean()
        return score
    
# TODO: Test CatBoost and LightGBM

# In[17]:


# %pdb off
# Load the dataset in advance for reusing it each trial execution.
data = (X, Y)
objective = Objective(data)

study = optuna.create_study(direction="maximize")
# Parallelize in all CPU cores
study.optimize(objective, n_trials=64, n_jobs=-1)#,  show_progress_bar=True)
print('study best_trial:', study.best_trial)
dump(study, f'study-{datetime.date.today()}.pkl')


# In[18]:


study = load('study-3.pkl')
print('Best trial until now:')
print(' Value: ', study.best_trial.value)
print(' Params: ')
for key, value in study.best_trial.params.items():
    print(f'    {key}: {value}')


# In[20]:


study.trials_dataframe()


# In[21]:


study.best_params


# # In[25]:


# bayesian_tuned = make_pipeline(transform, MultiOutputClassifier(GradientBoostingClassifier()))


# # In[ ]:


# bayesian_tuned.set_params(**study.best_params)
# bayesian_tuned.fit(X_train, Y_train)
# dump(bayesian_tuned, '../scripts/models/bayesian_tuned.joblib')


# # In[ ]:


# bayesian_tuned


# ### 9. Export your model as a pickle file

# In[31]:


# dump(clf, 'filename.joblib')

# pipeline_gbc.

