from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score, train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we not including hyperparameters.
    #parser.add_argument('--MY-HYPERPARM-NAME', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    data = pd.concat(raw_data)

    # move from a data frame into a numpy array
    X = data.iloc[:,1:].to_numpy()
    y = data['outcome'].to_numpy()
    
    # Cast to numeric types
    X = X.astype(float)
    y = y.astype(int)

    # Here we don't use hyperparameters. We could set this here
    # my_hyperparam = args.MY_HYPERPARM_NAME
    
    # split data in 80%-20% training/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # impute mean for missing values, since we have only numerical values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # evaluate a logistic regression with 5-fold cross-validation
    estimator = Pipeline([("imputer", SimpleImputer(missing_values=np.nan,
                                          strategy="mean")),
                          ('scaler', StandardScaler()),
                          ("regression", LogisticRegressionCV(cv=5,
                         
    # train pipeline
    estimator.fit(X_train, y_train)

    # predict class labels for the test set
    y_pred = model.predict(X_test)

    # generate class probabilities
    y_prob = model.predict_proba(X_test)

#     # generate evaluation metrics
#     print('Accuracy = {}'.format(accuracy_score(y_test, y_pred)))
#     print('AUROC = {}'.format(roc_auc_score(y_test, y_prob[:, 1])))

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(estimator, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    estimator = joblib.load(os.path.join(model_dir, "model.joblib"))
    return estimator
