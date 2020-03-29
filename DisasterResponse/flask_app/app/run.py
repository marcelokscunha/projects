import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
db_path = '../data/DisasterResponse.db'
engine = create_engine(f'sqlite:///{db_path}')

df = pd.read_sql('disaster_data', engine)
df.drop('child_alone', axis=1, inplace=True)
X = df.iloc[:,1].values
Y = df.iloc[:,4:].values
col_names = df.columns

# load model - best model was the Grid Search tuned Gradient Boosting Classifier
best_model = 'cv_gbc'
model = load(f'../models/{best_model}.joblib')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i in range(len(graphs))]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(col_names[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()